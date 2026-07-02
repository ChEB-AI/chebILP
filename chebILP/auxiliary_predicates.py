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
import threading
from dataclasses import dataclass
from typing import Callable, Literal

from rdkit import Chem

AuxKind = Literal["atom_unary", "atom_binary", "molecule"]
VALID_KINDS: set[str] = {"atom_unary", "atom_binary", "molecule"}

# Default wall-clock budget (seconds) for a single predicate's ``extension(mol)``
# call. LLM-generated programs occasionally contain pathological loops or
# exponential substructure searches that would otherwise hang ``build_bk``.
DEFAULT_AUX_TIMEOUT: float = 1.0


class AuxiliaryPredicateTimeout(Exception):
    """Raised when a predicate's ``extension(mol)`` exceeds its time budget."""


@dataclass
class AuxiliaryPredicate:
    """A single loaded auxiliary predicate ready to be applied to molecules."""

    name: str  # final Prolog predicate name, already ``aux_``-prefixed
    kind: AuxKind
    fn: Callable[[Chem.Mol], object]
    description: str = ""
    source_file: str = ""
    # Set to ``True`` once the predicate has exceeded its timeout, so it is
    # skipped for the remaining molecules of the same class instead of hanging
    # on every one of them.
    disabled: bool = False


def _call_with_timeout(fn: Callable, mol, timeout: float | None):
    """Run ``fn(mol)`` with a wall-clock ``timeout`` (seconds).

    Uses a daemon thread so the call works on Windows (where ``signal.alarm``
    is unavailable). If the call does not finish in time, ``AuxiliaryPredicateTimeout``
    is raised; the worker thread is abandoned (it cannot be force-killed) but,
    being a daemon, it will not keep the interpreter alive.
    """
    if timeout is None or timeout <= 0:
        return fn(mol)

    box: dict = {}

    def _worker():
        try:
            box["value"] = fn(mol)
        except BaseException as e:  # noqa: BLE001 - propagate to the caller thread
            box["error"] = e

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise AuxiliaryPredicateTimeout(f"timed out after {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


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
    Missing / empty directories yield an empty list (logged at debug level) rather than
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
        # Expected for classes that were never given auxiliary predicates; not a problem,
        # so log at debug level to avoid spamming when scanning every class of a split.
        logging.debug(
            "No auxiliary predicates found for CHEBI:%s in %s. "
            "Falling back to plain atom predicates.",
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
            logging.debug(
                "Duplicate auxiliary predicate name %s (from %s) skipped.", pred.name, path
            )
            continue
        seen.add(pred.name)
        predicates.append(pred)

    return predicates


def _normalize_result(pred: AuxiliaryPredicate, result):
    """Convert a predicate's raw ``extension`` result into BK-ready extensions.

    Returns ``("molecule", bool)``, ``("atom", list[int] | list[tuple])`` so the
    caller can slot it into either the molecule- or atom-level extension maps.
    """
    if pred.kind == "molecule":
        return "molecule", bool(result)
    if pred.kind == "atom_unary":
        return "atom", ([int(i) for i in result] if result else [])
    # atom_binary
    return "atom", ([(int(a), int(b)) for a, b in result] if result else [])


def compute_auxiliary_extensions(
    predicates: list[AuxiliaryPredicate],
    mols: list[tuple[object, Chem.Mol]],
    timeout: float | None = DEFAULT_AUX_TIMEOUT,
) -> dict[object, tuple[dict[str, list], set[str]]]:
    """Evaluate auxiliary predicates over many molecules at once.

    ``mols`` is a list of ``(mol_id, mol)`` pairs. Returns a mapping
    ``mol_id -> (atom_extensions, mol_extensions)`` in exactly the format produced
    by ``mol_to_fol_atoms`` so the results can be merged directly:
    - ``atom_extensions``: ``dict[name -> list[int]]`` (atom_unary) or
      ``dict[name -> list[tuple[int, int]]]`` (atom_binary).
    - ``mol_extensions``: ``set[name]`` for molecule-level predicates that hold.

    Each predicate's ``extension(mol)`` call is capped at ``timeout`` seconds
    (pass ``None`` or a non-positive value to disable). A predicate that exceeds
    the budget on *any* molecule is dropped **entirely**: it is marked
    ``disabled`` and none of its facts are returned for any molecule, not even
    the ones it had already been evaluated on successfully. This guarantees the
    background knowledge never contains a partially-computed extension. Any
    predicate that raises (without timing out) is merely skipped for the
    offending molecule so a single buggy input cannot abort the whole build.
    """
    atom_ext_by_mol: dict[object, dict[str, list]] = {mol_id: {} for mol_id, _ in mols}
    mol_ext_by_mol: dict[object, set[str]] = {mol_id: set() for mol_id, _ in mols}

    for pred in predicates:
        if pred.disabled:
            # Already timed out on an earlier batch (e.g. a previous split);
            # keep it dropped so the class stays internally consistent.
            continue

        # Buffer this predicate's results and only commit them once we know it
        # survived every molecule, so a late timeout discards earlier successes.
        pending_atom: dict[object, list] = {}
        pending_mol: set[object] = set()
        timed_out = False

        for mol_id, mol in mols:
            try:
                result = _call_with_timeout(pred.fn, mol, timeout)
            except AuxiliaryPredicateTimeout as e:
                pred.disabled = True
                timed_out = True
                # Expected for pathological predicates and repeated per molecule when
                # extensions are computed one molecule at a time; log at debug level.
                logging.debug(
                    "Auxiliary predicate %s %s; dropping it from the background knowledge "
                    "(including molecules where it already succeeded).",
                    pred.name,
                    e,
                )
                break
            except Exception as e:
                logging.debug(
                    "Auxiliary predicate %s crashed on a molecule, skipping: %s", pred.name, e
                )
                continue

            level, value = _normalize_result(pred, result)
            if level == "molecule":
                if value:
                    pending_mol.add(mol_id)
            elif value:
                pending_atom[mol_id] = value

        if timed_out:
            continue

        for mol_id, ext in pending_atom.items():
            atom_ext_by_mol[mol_id][pred.name] = ext
        for mol_id in pending_mol:
            mol_ext_by_mol[mol_id].add(pred.name)

    return {mol_id: (atom_ext_by_mol[mol_id], mol_ext_by_mol[mol_id]) for mol_id, _ in mols}
