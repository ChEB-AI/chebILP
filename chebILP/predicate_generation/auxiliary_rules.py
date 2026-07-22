"""Load and store LLM-generated auxiliary predicates written as ASP/clingo rules.

This is the rule-based counterpart of :mod:`chebILP.predicate_generation.auxiliary_predicates` (which stores
Python programs). Instead of a Python ``extension(mol)`` function, each auxiliary predicate
here is a small logic program: one head ``aux_<name>(...) :- <body>`` plus any helper clauses
it needs. During ``build_bk`` the program is grounded against the atom facts (and optional
computed facts) with Clingo and only the derived ``aux_*`` extension is written to ``bk.pl``
(see ``ilp_problem_builder.build_bk`` for the ``llm_generated_rules`` branch).

The head may be of any arity, mirroring the kinds the Python pipeline supports: a molecule
predicate ``aux_x(M)``, an atom one ``aux_x(A)``, an atom pair ``aux_x(A1,A2)``, or any mix.
A derived atom applies to a molecule when one of its arguments is that molecule or one of the
molecule's atoms — see :func:`derive_rule_extensions`, which is how both the generator's
validate-and-reject step and ``build_bk`` attribute an extension to molecules.

A class's programs are grounded TOGETHER, so one program may use a predicate another defines
and helpers are shared. The flip side is that names are shared too: two programs of the same
class defining one name differently merge into a single extension. Programs of *different*
classes must therefore be name-qualified before they meet in one grounding.

Storage reuses the shared-library layout of the Python pipeline, but with ``.pl`` files and a
separate default directory so the two libraries never mix::

    <library_dir>/
        programs/<aux_name>.pl     # one canonical rule program per predicate
        class_map.json             # { "<chebi_id>": ["aux_name", ...] }
        generation_logs/chebi_<id>.md

A rule program file carries a two-line header naming the predicate and describing it::

    % PREDICATE_NAME: aux_at_least_three_rings
    % DESCRIPTION: molecule has at least three rings
    aux_at_least_three_rings(M) :- N = #count{ S : ring_size(M,S) }, N >= 3.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from chebILP.predicate_generation.auxiliary_predicates import sanitize_predicate_name
from chebILP.ilp_path_manager import get_aux_class_map_path, get_aux_programs_dir

logger = logging.getLogger(__name__)

# Rule library, kept separate from the Python-program library (data/llm_generated_predicates).
DEFAULT_AUX_RULE_LIBRARY_DIR: str = os.path.join("data", "llm_generated_rules")

_NAME_RE = re.compile(r"^%\s*PREDICATE_NAME\s*:\s*(.+?)\s*$", re.IGNORECASE)
_DESC_RE = re.compile(r"^%\s*DESCRIPTION\s*:\s*(.+?)\s*$", re.IGNORECASE)


@dataclass
class RuleProgram:
    """A single auxiliary predicate expressed as an ASP/clingo program."""

    name: str  # final Prolog predicate name, already ``aux_``-prefixed
    description: str = ""
    source: str = ""  # verbatim program text: head clause + any helper clauses
    source_file: str = ""


def parse_rule_program(source: str, source_file: str = "<string>") -> RuleProgram | None:
    """Parse a rule-program string into a :class:`RuleProgram`, or ``None`` if invalid.

    Only the ``%`` annotation headers are read; the program body is carried verbatim and
    handed to clingo whole, so multi-line clauses, aggregates and ranges never pass
    through a parser of our own. A missing ``% PREDICATE_NAME:`` header makes the program
    unusable, since the name is what the derived extension is emitted under. Whether the
    name actually heads a clause is settled by grounding, not by inspecting the text.
    """
    name_raw = None
    description = ""
    for line in source.splitlines():
        m = _NAME_RE.match(line.strip())
        if m and name_raw is None:
            name_raw = m.group(1)
        m = _DESC_RE.match(line.strip())
        if m and not description:
            description = m.group(1)

    if name_raw is None:
        logger.warning("Rule program %s has no PREDICATE_NAME header.", source_file)
        return None
    if not source.strip():
        logger.warning("Rule program %s is empty.", source_file)
        return None

    return RuleProgram(
        name=sanitize_predicate_name(name_raw),
        description=description,
        source=source,
        source_file=source_file,
    )


_HAS_ATOM_RE = re.compile(r"^\s*has_atom\(\s*([^,\s]+)\s*,\s*([^)\s]+)\s*\)\s*\.\s*$")


def atom_to_molecule(facts: list[str]) -> dict[str, str]:
    """Map every atom id to its molecule, read from the ``has_atom/2`` facts."""
    mapping: dict[str, str] = {}
    for line in facts:
        m = _HAS_ATOM_RE.match(line)
        if m:
            mapping[m.group(2)] = m.group(1)
    return mapping


def rule_program_error(source: str) -> str | None:
    """Clingo's complaint if ``source`` is not a well-formed program, else ``None``.

    Grounded alone, with no facts and no sibling programs, so this reports only what is wrong
    with the program itself — syntax and unsafe variables. A predicate it references but does
    not define is *not* an error here: a sibling program grounded alongside it may define it
    (see :func:`derive_rule_extensions`).
    """
    import clingo

    messages: list[str] = []
    ctl = clingo.Control(logger=lambda code, msg: messages.append(msg))
    try:
        ctl.add("base", [], source)
        ctl.ground([("base", [])])
    except RuntimeError as e:
        detail = next((m.strip() for m in messages if "error" in m.lower()), "")
        return detail or str(e).strip().splitlines()[0]
    return None


def derive_rule_extensions(progs, facts: list[str], mol_ids, timeout: float | None = None) -> dict[str, dict[str, list[tuple[str, ...]]]]:
    """Ground ``progs`` together and group each one's derived atoms by molecule.

    All programs go into a single clingo instance, so one may build on predicates another
    defines. Names are consequently shared across ``progs``: two programs that define the same
    predicate differently contribute to a single extension. Within one class that is the
    intent; across classes the names must be qualified first (see ``test._qualify_aux_rules``).

    A head may have any arity: a molecule predicate ``aux_x(M)``, an atom one ``aux_x(A)``, a
    pair ``aux_x(A1,A2)``, or any mix. A derived atom belongs to a molecule when one of its
    arguments *is* that molecule or one of its atoms — resolved through the ``has_atom`` facts
    rather than the spelling of the atom id. Arguments that are neither (plain numbers, say)
    attach the atom to nothing on their own.

    Returns ``{program_name: {mol_id: [arg_tuple, ...]}}``, with one entry per program.
    """
    from chebILP.evaluation.clingo_eval import ground_extensions

    names = [p.name for p in progs]
    derived = ground_extensions([p.source for p in progs], facts, names, timeout=timeout)
    atom2mol = atom_to_molecule(facts)
    known = {str(m) for m in mol_ids}

    extensions: dict[str, dict[str, list[tuple[str, ...]]]] = {}
    for name in names:
        by_mol: dict[str, list[tuple[str, ...]]] = {}
        for args in derived.get(name, []):
            molecules = {atom2mol[a] for a in args if a in atom2mol} | {a for a in args if a in known}
            for mol in molecules & known:
                by_mol.setdefault(mol, []).append(args)
        extensions[name] = by_mol
    return extensions


def _rewrite_rule_name(source: str, old: str, new: str) -> str:
    """Rename the primary predicate ``old`` to ``new`` (header + word-boundary heads)."""
    source = _NAME_RE.sub(f"% PREDICATE_NAME: {new}", source, count=1) if _NAME_RE.search(source) else source
    return re.sub(rf"\b{re.escape(old)}\b", new, source)


def aux_rule_path(name: str, library_dir: str) -> str:
    """Path of a rule program's canonical ``.pl`` file in the library."""
    return os.path.join(get_aux_programs_dir(base_dir=library_dir), f"{sanitize_predicate_name(name)}.pl")


def add_rule_to_library(source: str, library_dir: str, chebi_id=None):
    """Add a rule program to the library, de-duplicating by content.

    Returns ``(stem, program)`` where ``stem`` is the library file name (without ``.pl``)
    to record in ``class_map.json``, or ``None`` if the program is invalid. A byte-identical
    program already present is reused. If the sanitized name backs *different* content, the
    new program is disambiguated with the class id (and the head/​header rewritten), so an
    existing predicate is never overwritten.
    """
    prog = parse_rule_program(source)
    if prog is None:
        return None

    programs_dir = get_aux_programs_dir(base_dir=library_dir)

    def _stem_path(stem: str) -> str:
        return os.path.join(programs_dir, f"{stem}.pl")

    content = source.rstrip() + "\n"
    stem = prog.name

    if not os.path.exists(_stem_path(stem)):
        with open(_stem_path(stem), "w", encoding="utf-8") as f:
            f.write(content)
        return stem, prog
    with open(_stem_path(stem), encoding="utf-8") as f:
        if f.read() == content:
            return stem, prog

    desired = f"{prog.name}_{chebi_id}" if chebi_id is not None else f"{prog.name}__2"
    source = _rewrite_rule_name(source, prog.name, desired)
    prog = parse_rule_program(source) or prog
    stem = prog.name
    content = source.rstrip() + "\n"

    suffix = 2
    while os.path.exists(_stem_path(stem)):
        with open(_stem_path(stem), encoding="utf-8") as f:
            if f.read() == content:
                return stem, prog
        candidate = f"{desired}__{suffix}"
        suffix += 1
        source = _rewrite_rule_name(source, prog.name, candidate)
        prog = parse_rule_program(source) or prog
        stem = prog.name
        content = source.rstrip() + "\n"

    with open(_stem_path(stem), "w", encoding="utf-8") as f:
        f.write(content)
    return stem, prog


def load_library_rules(base_dir: str) -> list[RuleProgram]:
    """Load every rule program in the library (for retrieval / inspection)."""
    programs_dir = get_aux_programs_dir(base_dir=base_dir)
    programs: list[RuleProgram] = []
    for fname in sorted(os.listdir(programs_dir)):
        if not fname.endswith(".pl") or fname.startswith("_"):
            continue
        path = os.path.join(programs_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            prog = parse_rule_program(f.read(), source_file=path)
        if prog is not None:
            programs.append(prog)
    return programs


def load_class_rules(chebi_id, library_dir: str | None = None) -> list[RuleProgram]:
    """Load the rule programs a ChEBI class uses (from ``class_map.json``).

    ``library_dir`` defaults to :data:`DEFAULT_AUX_RULE_LIBRARY_DIR`. Raises
    ``FileNotFoundError`` if the library has no ``class_map.json``, rather than silently
    building empty background knowledge. A class with no recorded rules yields ``[]``.
    """
    import json

    library_dir = library_dir or DEFAULT_AUX_RULE_LIBRARY_DIR
    class_map_path = get_aux_class_map_path(base_dir=library_dir)
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(
            f"No auxiliary-rule library found at {library_dir!r} (missing class_map.json). "
            "Generate rules first, or pass --predicate_dir with the library location."
        )
    with open(class_map_path, "r", encoding="utf-8") as f:
        names = json.load(f).get(str(chebi_id), [])
    if not names:
        logger.debug("No auxiliary rules recorded for CHEBI:%s.", chebi_id)
        return []

    programs: list[RuleProgram] = []
    seen: set[str] = set()
    for name in names:
        path = aux_rule_path(name, library_dir)
        if not os.path.exists(path):
            logger.warning(
                "Auxiliary rule %s used by CHEBI:%s is missing from the library (%s).",
                name, chebi_id, path,
            )
            continue
        with open(path, "r", encoding="utf-8") as f:
            prog = parse_rule_program(f.read(), source_file=path)
        if prog is None or prog.name in seen:
            continue
        seen.add(prog.name)
        programs.append(prog)
    return programs
