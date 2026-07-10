"""Load and apply LLM-generated auxiliary predicates.

Auxiliary predicates are extra molecule- or atom-level properties that the LLM
proposes to help tell a ChEBI class apart from its siblings (see
``generate_auxiliary_predicates.py``).

Storage (EXP-014): predicates are **not** stored per class any more. They live once
in a shared library and each class records which of them it uses:

    <problem_dir>/_aux_predicate_library/
        programs/<aux_name>.py      # one canonical program per unique predicate
        class_map.json              # { "<chebi_id>": ["aux_name", ...] }
        generation_logs/chebi_<id>.md

``load_auxiliary_predicates(chebi_id)`` looks up the class's names in ``class_map.json``
and loads those programs from ``programs/``. Each program follows this contract:

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

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Callable, Literal

from rdkit import Chem

logger = logging.getLogger(__name__)

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
        logger.warning("Auxiliary predicate %s failed to compile: %s", source_file, e)
        return None

    raw_name = namespace.get("PREDICATE_NAME")
    kind = namespace.get("KIND")
    fn = namespace.get("extension")

    if not isinstance(raw_name, str) or not raw_name.strip():
        logger.warning("Auxiliary predicate %s is missing a PREDICATE_NAME.", source_file)
        return None
    if kind not in VALID_KINDS:
        logger.warning("Auxiliary predicate %s has invalid KIND %r.", source_file, kind)
        return None
    if not callable(fn):
        logger.warning("Auxiliary predicate %s is missing an extension() function.", source_file)
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


def aux_program_path(name: str, problem_dir: str | None = None) -> str:
    """Path of a predicate's canonical program in the shared library."""
    from chebILP.ilp_path_manager import get_aux_programs_dir

    return os.path.join(get_aux_programs_dir(base_dir=problem_dir), f"{sanitize_predicate_name(name)}.py")


def load_class_map(problem_dir: str | None = None) -> dict[str, list[str]]:
    """Read ``class_map.json`` (``chebi_id -> [aux_name, ...]``); ``{}`` if absent."""
    from chebILP.ilp_path_manager import get_aux_class_map_path

    path = get_aux_class_map_path(base_dir=problem_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_class_map(class_map: dict[str, list[str]], problem_dir: str) -> str:
    """Write ``class_map.json`` (keys sorted for stable diffs). Returns the path."""
    from chebILP.ilp_path_manager import get_aux_class_map_path

    path = get_aux_class_map_path(base_dir=problem_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: class_map[k] for k in sorted(class_map)}, f, indent=2)
    return path


def set_class_predicates(chebi_id, names: list[str], problem_dir: str) -> None:
    """Record (overwriting) the auxiliary predicates a class uses, de-duplicated in order."""
    class_map = load_class_map(problem_dir)
    seen: set[str] = set()
    ordered = [n for n in names if not (n in seen or seen.add(n))]
    class_map[str(chebi_id)] = ordered
    save_class_map(class_map, problem_dir)


def _rewrite_predicate_name(source: str, new_name: str) -> str:
    """Rewrite the program's ``PREDICATE_NAME`` string literal to ``new_name`` (first only)."""
    pattern = re.compile(r'(PREDICATE_NAME\s*=\s*)(["\']).*?\2')
    new_source, n = pattern.subn(lambda m: f'{m.group(1)}"{new_name}"', source, count=1)
    return new_source if n else source


def add_program_to_library(source: str, problem_dir: str, chebi_id=None):
    """Add a predicate program to the shared library, de-duplicating by content.

    Returns ``(stem, predicate)`` where ``stem`` is the library file name (without ``.py``)
    to record in ``class_map.json``, or ``None`` if the program is invalid.

    A byte-identical program already in the library is reused. If the program's sanitized
    name already backs a *different* program, the new one is disambiguated with the class
    id (``aux_has_carboxyl_group`` -> ``aux_has_carboxyl_group_<chebi_id>``). The rewrite is
    applied to the program's ``PREDICATE_NAME`` too, so the predicate is unique in the
    background knowledge, not just on disk — an existing predicate is never overwritten.
    """
    from chebILP.ilp_path_manager import get_aux_programs_dir

    pred = load_program_source(source)
    if pred is None:
        return None

    programs_dir = get_aux_programs_dir(base_dir=problem_dir)

    def _stem_path(stem: str) -> str:
        return os.path.join(programs_dir, f"{stem}.py")

    content = source.rstrip() + "\n"
    stem = pred.name  # already aux_-prefixed and sanitized

    # No file under this name yet: store it as-is.
    if not os.path.exists(_stem_path(stem)):
        with open(_stem_path(stem), "w", encoding="utf-8") as f:
            f.write(content)
        return stem, pred
    # Same name, identical code: reuse the existing library program.
    with open(_stem_path(stem), encoding="utf-8") as f:
        if f.read() == content:
            return stem, pred

    # Same name, different code: disambiguate with the class id and rewrite PREDICATE_NAME.
    desired = f"{pred.name}_{chebi_id}" if chebi_id is not None else f"{pred.name}__2"
    source = _rewrite_predicate_name(source, desired)
    pred = load_program_source(source) or pred
    stem = pred.name
    content = source.rstrip() + "\n"

    # In the rare event the disambiguated name is also taken by different code (e.g. a
    # re-run of the same class with changed output), fall back to a numeric suffix.
    suffix = 2
    while os.path.exists(_stem_path(stem)):
        with open(_stem_path(stem), encoding="utf-8") as f:
            if f.read() == content:
                return stem, pred  # identical already present
        candidate = f"{desired}__{suffix}"
        suffix += 1
        source = _rewrite_predicate_name(source, candidate)
        pred = load_program_source(source) or pred
        stem = pred.name
        content = source.rstrip() + "\n"

    with open(_stem_path(stem), "w", encoding="utf-8") as f:
        f.write(content)
    return stem, pred


def load_library_predicates(base_dir: str) -> list[AuxiliaryPredicate]:
    """Load every unique predicate in the shared library (for retrieval / inspection)."""
    from chebILP.ilp_path_manager import get_aux_programs_dir

    programs_dir = get_aux_programs_dir(base_dir=base_dir)
    predicates: list[AuxiliaryPredicate] = []
    for fname in sorted(os.listdir(programs_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(programs_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            pred = load_program_source(f.read(), source_file=path)
        if pred is not None:
            predicates.append(pred)
    return predicates


def load_auxiliary_predicates(chebi_id, problem_dir: str | None = None) -> list[AuxiliaryPredicate]:
    """Load the auxiliary-predicate programs a ChEBI class uses.

    Looks up the class's predicate names in ``class_map.json`` and loads each one's
    canonical program from the shared library's ``programs/`` directory. A class with no
    entry (or a missing library) yields an empty list — logged at debug level — so
    ``build_bk`` degrades gracefully to the plain atom predicates. Names are
    de-duplicated (first occurrence wins) to keep the Prolog BK consistent.
    """
    names = load_class_map(problem_dir).get(str(chebi_id), [])
    if not names:
        # Expected for classes that were never given auxiliary predicates; not a problem,
        # so log at debug level to avoid spamming when scanning every class of a split.
        logger.debug(
            "No auxiliary predicates recorded for CHEBI:%s. "
            "Falling back to plain atom predicates.",
            chebi_id,
        )
        return []

    predicates: list[AuxiliaryPredicate] = []
    seen: set[str] = set()
    for name in names:
        path = aux_program_path(name, problem_dir)
        if not os.path.exists(path):
            logger.warning(
                "Auxiliary predicate %s used by CHEBI:%s is missing from the library (%s).",
                name, chebi_id, path,
            )
            continue
        with open(path, "r", encoding="utf-8") as f:
            pred = load_program_source(f.read(), source_file=path)
        if pred is None:
            continue
        if pred.name in seen:
            logger.debug("Duplicate auxiliary predicate name %s skipped.", pred.name)
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
    failures: dict[str, str] | None = None,
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

    If ``failures`` is provided, predicate names that could not be computed are
    recorded into it as ``name -> "timeout" | "error"`` (``"timeout"`` wins, since
    a timed-out predicate is dropped entirely) so callers can report them.
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
                if failures is not None:
                    failures[pred.name] = "timeout"
                # Expected for pathological predicates and repeated per molecule when
                # extensions are computed one molecule at a time; log at debug level.
                logger.debug(
                    "Auxiliary predicate %s %s; dropping it from the background knowledge "
                    "(including molecules where it already succeeded).",
                    pred.name,
                    e,
                )
                break
            except Exception as e:
                if failures is not None:
                    failures.setdefault(pred.name, "error")
                logger.debug(
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
