"""Load and apply LLM-generated auxiliary predicates.

Auxiliary predicates are extra molecule- or atom-level properties that the LLM
proposes for a specific ChEBI class (see ``generate_auxiliary_predicates.py``).
Each predicate is stored as a standalone Python program in
``data/ilp_problems/chebi_{id}/auxiliary_predicates/`` and follows this contract:

    \"\"\"<one-line description of what the predicate captures>\"\"\"
    PREDICATE_NAME = "carboxylic_acid"
    KIND = "atom_unary"   # "atom_unary" | "atom_binary" | "molecule"

    def extension(mol):
        # mol: an RDKit Mol object.
        # atom_unary  -> return an iterable of 0-based atom indices
        # atom_binary -> return an iterable of (i, j) 0-based atom-index pairs
        # molecule    -> return a bool (whether the property holds for the mol)
        ...

The predicate name that ends up in the Prolog background knowledge is
``aux_<sanitized PREDICATE_NAME>`` so that generated predicates can never
collide with the built-in ``mol_to_fol_atoms`` predicates or with reserved
Prolog atoms.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Literal

from rdkit import Chem

AuxKind = Literal["atom_unary", "atom_binary", "molecule"]
VALID_KINDS: set[str] = {"atom_unary", "atom_binary", "molecule"}


@dataclass
class AuxiliaryPredicate:
    """A single loaded auxiliary predicate ready to be applied to molecules."""

    name: str  # final Prolog predicate name, already ``aux_``-prefixed
    kind: AuxKind
    fn: Callable[[Chem.Mol], object]
    description: str = ""
    source_file: str = ""


def sanitize_predicate_name(raw_name: str) -> str:
    """Turn an arbitrary LLM-proposed name into a safe ``aux_``-prefixed Prolog atom."""
    name = raw_name.strip().lower().replace(" ", "_").replace("-", "_")
    name = "".join(c for c in name if c.isalnum() or c == "_")
    name = name.strip("_")
    if not name:
        name = "predicate"
    if not name.startswith("aux_"):
        name = "aux_" + name
    return name


def load_program_source(source: str, source_file: str = "<string>") -> AuxiliaryPredicate | None:
    """Compile one auxiliary-predicate program string into an ``AuxiliaryPredicate``.

    Returns ``None`` (and logs a warning) if the program does not satisfy the
    contract or fails to compile. RDKit's ``Chem``/``rdkit`` are injected into
    the namespace so generated programs can rely on them without importing.
    """
    namespace: dict = {"Chem": Chem}
    try:
        from rdkit.Chem import rdMolDescriptors, Descriptors  # noqa: F401

        namespace.update({"rdMolDescriptors": rdMolDescriptors, "Descriptors": Descriptors})
    except Exception:  # pragma: no cover - rdkit submodule import guard
        pass

    try:
        exec(compile(source, source_file, "exec"), namespace)
    except Exception as e:
        logging.warning("Auxiliary predicate %s failed to compile: %s", source_file, e)
        return None

    raw_name = namespace.get("PREDICATE_NAME")
    kind = namespace.get("KIND")
    fn = namespace.get("extension")

    if not isinstance(raw_name, str) or not raw_name.strip():
        logging.warning("Auxiliary predicate %s is missing a PREDICATE_NAME.", source_file)
        return None
    if kind not in VALID_KINDS:
        logging.warning("Auxiliary predicate %s has invalid KIND %r.", source_file, kind)
        return None
    if not callable(fn):
        logging.warning("Auxiliary predicate %s is missing an extension() function.", source_file)
        return None

    doc = (fn.__doc__ or namespace.get("__doc__") or "").strip().splitlines()
    description = doc[0].strip() if doc else ""

    return AuxiliaryPredicate(
        name=sanitize_predicate_name(raw_name),
        kind=kind,  # type: ignore[arg-type]
        fn=fn,
        description=description,
        source_file=source_file,
    )


def load_auxiliary_predicates(chebi_id, problem_dir: str | None = None) -> list[AuxiliaryPredicate]:
    """Load every auxiliary-predicate program for a ChEBI class.

    Programs live in ``<problem_dir>/chebi_{id}/auxiliary_predicates/*.py``.
    Missing / empty directories yield an empty list (with a warning) rather than
    an error, so ``build_bk`` degrades gracefully to the plain atom predicates.
    Names are de-duplicated (first file wins) to keep the Prolog BK consistent.
    """
    from chebILP.ilp_path_manager import get_aux_predicates_dir

    aux_dir = get_aux_predicates_dir(chebi_id, base_dir=problem_dir)
    py_files = sorted(
        os.path.join(aux_dir, f)
        for f in os.listdir(aux_dir)
        if f.endswith(".py") and not f.startswith("_")
    )
    if not py_files:
        logging.warning(
            "No auxiliary predicates found for CHEBI:%s in %s. "
            "Falling back to plain atom predicates. "
            "Run 'python -m chebILP.generate_auxiliary_predicates' first.",
            chebi_id,
            aux_dir,
        )
        return []

    predicates: list[AuxiliaryPredicate] = []
    seen: set[str] = set()
    for path in py_files:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        pred = load_program_source(source, source_file=path)
        if pred is None:
            continue
        if pred.name in seen:
            logging.warning(
                "Duplicate auxiliary predicate name %s (from %s) skipped.", pred.name, path
            )
            continue
        seen.add(pred.name)
        predicates.append(pred)

    return predicates


def apply_auxiliary_predicates(
    predicates: list[AuxiliaryPredicate], mol: Chem.Mol
) -> tuple[dict[str, list], set[str]]:
    """Evaluate auxiliary predicates on one molecule.

    Returns ``(atom_extensions, mol_extensions)`` in exactly the format produced
    by ``mol_to_fol_atoms`` so the results can be merged directly:
    - ``atom_extensions``: ``dict[name -> list[int]]`` (atom_unary) or
      ``dict[name -> list[tuple[int, int]]]`` (atom_binary).
    - ``mol_extensions``: ``set[name]`` for molecule-level predicates that hold.

    Any predicate that raises is skipped for this molecule (logged once) so a
    single buggy program cannot abort the whole background-knowledge build.
    """
    atom_extensions: dict[str, list] = {}
    mol_extensions: set[str] = set()

    for pred in predicates:
        try:
            result = pred.fn(mol)
        except Exception as e:
            logging.warning(
                "Auxiliary predicate %s crashed on a molecule, skipping: %s", pred.name, e
            )
            continue

        if pred.kind == "molecule":
            if result:
                mol_extensions.add(pred.name)
        elif pred.kind == "atom_unary":
            indices = [int(i) for i in result] if result else []
            if indices:
                atom_extensions[pred.name] = indices
        elif pred.kind == "atom_binary":
            pairs = [(int(a), int(b)) for a, b in result] if result else []
            if pairs:
                atom_extensions[pred.name] = pairs

    return atom_extensions, mol_extensions
