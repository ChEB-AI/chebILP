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
and helpers are shared. Names are shared too, but that is safe across classes: a name in the
library backs exactly one program (:func:`add_rule_to_library` disambiguates content that
would collide with the class id), so programs of different classes can meet in one grounding
as they are.

Storage reuses the shared-library layout of the Python pipeline, but with ``.pl`` files and a
separate default directory so the two libraries never mix::

    <library_dir>/
        programs/<aux_name>.pl     # one canonical rule program per predicate
        class_map.json             # { "<chebi_id>": ["aux_name", ...] }
        generation_logs/chebi_<id>.md

A rule program file carries a two-line header naming the predicate and describing it::

    % PREDICATE_NAME: aux_at_least_three_ring_atoms
    % DESCRIPTION: molecule has at least three atoms lying in a ring
    aux_at_least_three_ring_atoms(M) :- has_atom(M,_), 3 <= #count{ A : has_atom(M,A), in_ring(A) }.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from chebILP.predicate_generation.auxiliary_predicates import sanitize_predicate_name
from chebILP.ilp_path_manager import (
    get_aux_class_map_path,
    get_aux_hypotheses_path,
    get_aux_programs_dir,
)

logger = logging.getLogger(__name__)

# Rule library, kept separate from the Python-program library (data/llm_generated_predicates).
DEFAULT_AUX_RULE_LIBRARY_DIR: str = os.path.join("data", "llm_generated_rules")

# Wall-clock ceiling for a whole ``derive_rule_extensions`` call, across all of its batches.
# Generous: it is a backstop against a program that never finishes, not a performance budget.
DEFAULT_GROUNDING_TIMEOUT: float = 300.0

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
_FACT_RE = re.compile(r"^\s*[a-z_][A-Za-z0-9_]*\s*\((.*)\)\s*\.\s*$")


def atom_to_molecule(facts: list[str]) -> dict[str, str]:
    """Map every atom id to its molecule, read from the ``has_atom/2`` facts."""
    mapping: dict[str, str] = {}
    for line in facts:
        m = _HAS_ATOM_RE.match(line)
        if m:
            mapping[m.group(2)] = m.group(1)
    return mapping


def group_facts_by_molecule(facts: list[str], mol_ids) -> list[list[str]]:
    """Split ``facts`` into one fact list per molecule, in ``mol_ids`` order.

    Which molecule a fact belongs to is resolved through the ``has_atom`` facts, not the
    spelling of the atom id — the same rule :func:`derive_rule_extensions` uses to attribute
    a derived atom. A fact naming neither an atom nor a known molecule (there are normally
    none) is unattributable and is copied into *every* group, so an unrecognised fact shape
    can only ever be over-supplied, never silently dropped.
    """
    atom2mol = atom_to_molecule(facts)
    known = {str(m) for m in mol_ids}

    by_mol: dict[str, list[str]] = {m: [] for m in known}
    shared: list[str] = []
    for line in facts:
        m = _FACT_RE.match(line)
        args = [a.strip() for a in m.group(1).split(",")] if m else []
        owner = next(
            (atom2mol[a] for a in args if a in atom2mol),
            next((a for a in args if a in known), None),
        )
        if owner is None:
            shared.append(line)
        elif owner in by_mol:
            by_mol[owner].append(line)

    ordered = [str(m) for m in mol_ids if str(m) in by_mol]
    return [by_mol[m] + shared for m in dict.fromkeys(ordered) if by_mol[m] or shared]


def rule_program_error(source: str) -> str | None:
    """Clingo's complaint if ``source`` is not a well-formed program, else ``None``.

    Grounded alone, with no facts and no sibling programs, so this reports only what is wrong
    with the program itself — syntax and unsafe variables. A predicate it references but does
    not define is *not* an error here: a sibling program grounded alongside it may define it
    (see :func:`derive_rule_extensions`).
    """
    import clingo

    from chebILP.evaluation.clingo_eval import _patch_clingo_logger_decode

    _patch_clingo_logger_decode()
    messages: list[str] = []
    ctl = clingo.Control(logger=lambda code, msg: messages.append(msg))
    try:
        ctl.add("base", [], source)
        ctl.ground([("base", [])])
    except RuntimeError as e:
        detail = next((m.strip() for m in messages if "error" in m.lower()), "")
        return detail or str(e).strip().splitlines()[0]
    return None


_COMMENT_RE = re.compile(r"%.*")
_PRED_RE = re.compile(r"\b([a-z_][A-Za-z0-9_]*)\s*\(")
# ``L = L0 + 1`` and friends: a variable assigned another variable offset by a constant.
_INCREMENT_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\s*=\s*([A-Z][A-Za-z0-9_]*)\s*[+-]\s*\d+")
# the same growth written straight into a head argument: ``aux_p(A, D, N+1)``
_ARG_ARITH_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\s*[+-]\s*\d+")


def _clauses(source: str) -> list[tuple[str, str]]:
    """``(head, body)`` for every clause in ``source``; a fact gets an empty body."""
    parts = re.split(r"\.(?=\s|$)", _COMMENT_RE.sub("", source))
    clauses = []
    for chunk in parts:
        chunk = chunk.strip()
        if chunk:
            head, _, body = chunk.partition(":-")
            clauses.append((head.strip(), body.strip()))
    return clauses


def _recursive_predicates(progs) -> set[str]:
    """Predicate names that depend on themselves, directly or through other clauses."""
    edges: dict[str, set[str]] = {}
    for prog in progs:
        for head, body in _clauses(prog.source):
            m = _PRED_RE.match(head)
            if m is not None and body:
                edges.setdefault(m.group(1), set()).update(_PRED_RE.findall(body))

    reach = {name: set(deps) for name, deps in edges.items()}
    changed = True
    while changed:  # transitive closure; a class's dependency graph is a handful of nodes
        changed = False
        for name, deps in reach.items():
            grown = deps.union(*(reach.get(d, set()) for d in deps)) if deps else deps
            if grown != deps:
                reach[name] = grown
                changed = True
    return {name for name, deps in reach.items() if name in deps}


def _bounded_above(clause: str, variables: set[str]) -> bool:
    """Whether any of ``variables`` is capped by an integer constant in ``clause``."""
    return any(
        re.search(rf"\b{re.escape(v)}\s*(?:<=|<)\s*\d+", clause)
        or re.search(rf"\b\d+\s*(?:>=|>)\s*{re.escape(v)}\b", clause)
        for v in variables
    )


def unbounded_recursion_errors(progs) -> dict[str, str]:
    """Programs whose grounding would never terminate, keyed by program name.

    A recursive clause that increments a counter derives a *new* ground atom at every step,
    so its fixpoint is infinite and clingo grounds until memory runs out. The molecule graph
    is always cyclic here — ``has_bond_to`` is symmetric, so a walk can step A -> B -> A even
    in an acyclic molecule — and the counter keeps each lap distinct. Dropping the counter
    makes the very same rule saturate, which is why plain reachability is safe and this is
    not. Only a constant upper bound in the SAME clause stops the grounder; one applied by a
    consumer clause comes too late.

    ``progs`` is inspected as a set, since the cycle may run through a sibling or a reused
    library program, but only the clause carrying the increment is blamed.
    """
    recursive = _recursive_predicates(progs)
    errors: dict[str, str] = {}
    for prog in progs:
        for head, body in _clauses(prog.source):
            m = _PRED_RE.match(head)
            if m is None or not body or m.group(1) not in recursive:
                continue
            clause = f"{head} :- {body}"
            growing = {v for pair in _INCREMENT_RE.findall(clause) for v in pair}
            growing |= set(_ARG_ARITH_RE.findall(head[head.find("(") + 1:] if "(" in head else ""))
            if growing and not _bounded_above(clause, growing):
                errors[prog.name] = (
                    f"non-terminating recursion: {m.group(1)} recurses while incrementing "
                    f"{'/'.join(sorted(growing))}, with no constant upper bound in that clause"
                )
                break
    return errors


_VAR_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")
_NOT_RE = re.compile(r"^not\s+")
_AGG_BRACES_RE = re.compile(r"\{[^{}]*\}")
# A literal with no predicate call: ``A != C``, ``R >= 5``, ``L = L0 + 1``, ``M = M``.
_COMPARISON_OPS_RE = re.compile(r"!=|<=|>=|<|>|=")


def _body_literals(body: str) -> list[str]:
    """Top-level comma-separated literals of ``body``; commas inside ``(``/``{`` don't split."""
    literals, depth, current = [], 0, ""
    for ch in body:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if ch == "," and depth == 0:
            literals.append(current)
            current = ""
        else:
            current += ch
    literals.append(current)
    return [lit.strip() for lit in literals if lit.strip()]


class _Union:
    """Minimal union-find over variable names."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, var: str) -> None:
        self.parent.setdefault(var, var)

    def find(self, var: str) -> str:
        while self.parent[var] != var:
            self.parent[var] = self.parent[self.parent[var]]
            var = self.parent[var]
        return var

    def union(self, vars_) -> None:
        vars_ = [v for v in vars_ if v in self.parent]
        for other in vars_[1:]:
            a, b = self.find(vars_[0]), self.find(other)
            if a != b:
                self.parent[a] = b

    def components(self) -> dict[str, set[str]]:
        groups: dict[str, set[str]] = {}
        for var in self.parent:
            groups.setdefault(self.find(var), set()).add(var)
        return groups


def _clause_components(head: str, body: str) -> tuple[dict[str, set[str]], list[str], set[str]]:
    """``(variable components, negative literals, aggregate result variables)`` for a clause.

    Variables are joined only by *positive* predicate literals and by plain ``=``. A negated
    literal cannot bind anything, so it does not join; nor does an inequality. Inside an
    aggregate the element variables are local, but a variable that also occurs outside it is
    global — so an aggregate joins exactly those of its variables that appear elsewhere. That
    is what separates a molecule-scoped count from an unscoped one: ``#count{ A : has_atom(M,A),
    … }`` mentions ``M`` inside and so ties its result to ``M``, while ``#count{ A : p(A) }``
    leaves the result standing alone.
    """
    literals = _body_literals(body)
    union = _Union()
    for var in _VAR_RE.findall(f"{head} {body}"):
        union.add(var)

    aggregate_results: set[str] = set()
    for i, literal in enumerate(literals):
        if _NOT_RE.match(literal):
            continue
        if "#" in literal:
            elsewhere = set(_VAR_RE.findall(head))
            elsewhere.update(v for j, o in enumerate(literals) if j != i for v in _VAR_RE.findall(o))
            union.union([v for v in _VAR_RE.findall(literal) if v in elsewhere])
            # variables outside the braces are the aggregate's result, bound by it
            outer = _VAR_RE.findall(_AGG_BRACES_RE.sub(" ", literal))
            aggregate_results.update(outer)
            union.union(outer)
        elif "(" in literal:
            union.union(_VAR_RE.findall(literal))
        elif _COMPARISON_OPS_RE.search(literal):
            if re.fullmatch(r"[^!<>=]*=[^=]*", literal):  # plain equality does bind
                union.union(_VAR_RE.findall(literal))

    return union.components(), [lit for lit in literals if _NOT_RE.match(lit)], aggregate_results


# Stable identifiers for the static rejections, so a caller — a repair prompt for a second
# LLM pass, say — can branch on the error type instead of parsing the message.
ERROR_UNBOUNDED_RECURSION = "unbounded_recursion"
ERROR_CROSS_PRODUCT = "cross_product"
ERROR_NEGATION_SCOPE = "negation_scope"
ERROR_UNSCOPED_AGGREGATE = "unscoped_aggregate"


def cross_product_errors(progs) -> dict[str, tuple[str, str]]:
    """Programs with an unjoined-variable clause, as ``{name: (code, message)}``.

    Such a clause's body splits into groups of variables that no positive literal connects,
    so clingo grounds their **cartesian product** — over every atom in the instance, not just
    the molecule's. Two harms: the extension is wrong (a molecule-level head fires because of
    a *different* molecule's atoms) and the cost is quadratic in the molecules ground
    together, which is what exhausted memory in ``build_bk``. Batching bounds the cost
    (see :func:`derive_rule_extensions`); only rejection fixes the meaning.

    Three codes are reported, most specific first:

    - :data:`ERROR_UNSCOPED_AGGREGATE` — ``N = #count{ A : p(A) }`` with no ``has_atom(M,A)``
      inside, so every molecule is assigned the same instance-wide total. Cheap to ground and
      therefore invisible to any memory guard; only wrong.
    - :data:`ERROR_NEGATION_SCOPE` — ``not p(X), q(X)`` written for ``not (p(X), q(X))``, which
      ``_SYSTEM_PROMPT`` already forbids. The clause stays *safe*, so clingo accepts it and
      only this check catches it.
    - :data:`ERROR_CROSS_PRODUCT` — anything else, normally a head variable never joined to the
      atoms the body tests.

    The last two overlap where a pair predicate leaves its two atoms unjoined; the message
    names the offending variables either way.

    Unlike :func:`unbounded_recursion_errors` this is per-program: a cross product is local to
    one clause and cannot be closed by a sibling.
    """
    errors: dict[str, tuple[str, str]] = {}
    for prog in progs:
        for head, body in _clauses(prog.source):
            if not body or _PRED_RE.match(head) is None:
                continue
            components, negatives, aggregate_results = _clause_components(head, body)
            # A variable no positive literal mentions is not a component of its own; only
            # groups that actually generate atoms count.
            positive_vars = {
                v
                for lit in _body_literals(body)
                if not _NOT_RE.match(lit) and "(" in lit
                for v in _VAR_RE.findall(_AGG_BRACES_RE.sub(" ", lit) if "#" in lit else lit)
            }
            generating = [c for c in components.values() if c & positive_vars]
            if len(generating) < 2:
                continue

            spanned = next(
                (
                    lit
                    for lit in negatives
                    if sum(1 for c in generating if c & set(_VAR_RE.findall(lit))) > 1
                ),
                None,
            )
            unscoped = next((c for c in generating if c <= aggregate_results), None)
            groups = " and ".join("{" + ", ".join(sorted(c)) + "}" for c in generating)
            if unscoped is not None:
                errors[prog.name] = (
                    ERROR_UNSCOPED_AGGREGATE,
                    f"aggregate not scoped to the molecule: {', '.join(sorted(unscoped))} is "
                    f"counted over every molecule at once, so each molecule gets the same total. "
                    f"Bind the element to the molecule inside the aggregate, as in "
                    f"#count{{ A : has_atom(M,A), ... }}",
                )
            elif spanned is not None:
                errors[prog.name] = (
                    ERROR_NEGATION_SCOPE,
                    f"negation over a conjunction: `{spanned.strip()}` is the only thing linking "
                    f"{groups}, but a negated literal cannot bind. Define a helper predicate for "
                    f"the whole conjunction and negate that helper instead",
                )
            else:
                errors[prog.name] = (
                    ERROR_CROSS_PRODUCT,
                    f"unjoined variables: {groups} are never linked by a positive literal, so "
                    f"this clause grounds as their cross product over every molecule. Join them "
                    f"— usually via has_atom(M,...) for each atom variable",
                )
            break
    return errors


def static_rule_errors(progs) -> dict[str, tuple[str, str]]:
    """All static rejections for ``progs``, as ``{program name: (code, message)}``.

    The single entry point for checks that must run *before* clingo sees a program, because
    the failure they describe is one grounding cannot survive (non-termination) or one it
    cannot detect (a cross product grounds "successfully", just wrongly and enormously).
    """
    errors = {
        name: (ERROR_UNBOUNDED_RECURSION, message)
        for name, message in unbounded_recursion_errors(progs).items()
    }
    for name, entry in cross_product_errors(progs).items():
        errors.setdefault(name, entry)
    return errors


def derive_rule_extensions(progs, facts: list[str], mol_ids, timeout: float | None = DEFAULT_GROUNDING_TIMEOUT, batch_size: int | None = None) -> dict[str, dict[str, list[tuple[str, ...]]]]:
    """Ground ``progs`` together and group each one's derived atoms by molecule.

    All programs go into a single clingo instance, so one may build on predicates another
    defines. Names are consequently shared across ``progs``: two programs that define the same
    predicate differently would contribute to a single extension. The library rules that out
    by construction — one name, one program — so programs from several classes may be passed
    together.

    A head may have any arity: a molecule predicate ``aux_x(M)``, an atom one ``aux_x(A)``, a
    pair ``aux_x(A1,A2)``, or any mix. A derived atom belongs to a molecule when one of its
    arguments *is* that molecule or one of its atoms — resolved through the ``has_atom`` facts
    rather than the spelling of the atom id. Arguments that are neither (plain numbers, say)
    attach the atom to nothing on their own.

    The facts are split per molecule and ground in batches of ``batch_size`` rather than all
    at once. For a well-formed program this changes nothing — every variable is joined to its
    molecule, so no clause could span two of them anyway — but it bounds what a clause with
    *unjoined* variables costs: such a clause grounds as a cross product over every atom in
    the instance, which at class scale (650+ molecules) reaches 10^8 ground atoms and exhausts
    memory. See :data:`~chebILP.evaluation.clingo_eval.GROUNDING_BATCH_SIZE`.

    ``timeout`` is the wall-clock ceiling for the whole call.

    Returns ``{program_name: {mol_id: [arg_tuple, ...]}}``, with one entry per program.
    """
    from chebILP.evaluation.clingo_eval import GROUNDING_BATCH_SIZE, ground_extensions_isolated

    if batch_size is None:
        batch_size = GROUNDING_BATCH_SIZE
    names = [p.name for p in progs]
    per_molecule = group_facts_by_molecule(facts, mol_ids)
    fact_groups = [
        [line for group in per_molecule[i:i + batch_size] for line in group]
        for i in range(0, len(per_molecule), batch_size)
    ] or [facts]
    derived = ground_extensions_isolated(
        [p.source for p in progs], fact_groups, names, timeout=timeout, total_timeout=timeout,
    )
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


def _defined_predicates(progs) -> set[str]:
    """Predicate names appearing in the head of any clause of ``progs``."""
    return {
        m.group(1)
        for prog in progs
        for head, _ in _clauses(prog.source)
        for m in [_PRED_RE.match(head)]
        if m is not None
    }


def _referenced_predicates(prog) -> set[str]:
    """Predicate names appearing in the body of any clause of ``prog``."""
    return {name for _, body in _clauses(prog.source) for name in _PRED_RE.findall(body)}


_DEFINITION_INDEX_CACHE: dict[str, dict[str, list[str]]] = {}


def _definition_index(library_dir: str) -> dict[str, list[str]]:
    """``{predicate: [program stem, ...]}`` over every clause head in the library.

    Memoised per directory: the index costs a full parse of the library (1600+ files) and
    ``build_bk`` resolves against it once per class.
    """
    if library_dir in _DEFINITION_INDEX_CACHE:
        return _DEFINITION_INDEX_CACHE[library_dir]

    programs_dir = get_aux_programs_dir(base_dir=library_dir)
    index: dict[str, list[str]] = {}
    for fname in sorted(os.listdir(programs_dir)):
        if not fname.endswith(".pl") or fname.startswith("_"):
            continue
        with open(os.path.join(programs_dir, fname), "r", encoding="utf-8") as f:
            source = f.read()
        stem = fname[:-3]
        for head, _ in _clauses(source):
            m = _PRED_RE.match(head)
            if m is not None and stem not in index.setdefault(m.group(1), []):
                index[m.group(1)].append(stem)
    _DEFINITION_INDEX_CACHE[library_dir] = index
    return index


def resolve_rule_dependencies(progs, library_dir: str | None = None) -> list[RuleProgram]:
    """Library programs defining ``aux_*`` predicates ``progs`` reference but do not define.

    ``class_map.json`` records only the predicates a class *chose*, never the ones those
    programs build on — the layering the generator asks for is invisible to it. So a program
    like ``aux_amino_nitrogen(N) :- aux_organic_amine_nitrogen(N), not aux_n_acylated_amino(N).``
    loads with both dependencies missing, derives nothing, and silently costs the class a
    predicate. Resolving them here fixes existing libraries without regenerating anything.

    The closure is transitive and cycle-safe. A predicate defined by several programs is
    resolved to the one whose file *is* that predicate, else one already loaded, else the
    first stem alphabetically (with a warning) — the library has no other notion of which
    definition is canonical.

    Returns only the programs that had to be added; the caller keeps emitting extensions for
    its own programs alone.
    """
    library_dir = library_dir or DEFAULT_AUX_RULE_LIBRARY_DIR
    index = _definition_index(library_dir)

    resolved: list[RuleProgram] = []
    defined = _defined_predicates(progs)
    loaded_stems: set[str] = set()
    pending = list(progs)

    while pending:
        for name in _referenced_predicates(pending.pop()):
            if not name.startswith("aux_") or name in defined:
                continue
            stems = index.get(name)
            if not stems:
                continue  # nothing in the library defines it; it just grounds as empty
            if name in stems:
                stem = name  # the program whose file IS this predicate is the canonical one
            else:
                stem = next((s for s in stems if s in loaded_stems), stems[0])
                if len(stems) > 1:
                    logger.warning(
                        "Auxiliary predicate %s is defined by %d library programs; using %s.",
                        name, len(stems), stem,
                    )
            if stem in loaded_stems:
                defined.add(name)  # already pulled in under a different predicate's name
                continue
            loaded_stems.add(stem)
            path = aux_rule_path(stem, library_dir)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                dep = parse_rule_program(f.read(), source_file=path)
            if dep is None:
                continue
            resolved.append(dep)
            pending.append(dep)
            defined |= _defined_predicates([dep])
    return resolved


def _head_args(head: str) -> list[str]:
    """Top-level argument list of a clause head, e.g. ``aux_x(O,C2)`` -> ``['O', 'C2']``."""
    m = re.match(r"\s*[a-z_][A-Za-z0-9_]*\s*\((.*)\)\s*$", head)
    return _body_literals(m.group(1)) if m else []


_HAS_ATOM_MOL_RE = re.compile(r"has_atom\(\s*([A-Z_][A-Za-z0-9_]*)\s*,")


def predicate_arg_sorts(prog: RuleProgram) -> list[str]:
    """Classify each argument of ``prog``'s own predicate as ``"mol"`` or ``"atom"``.

    A head argument is a molecule when it appears as the *first* argument of a ``has_atom/2``
    literal in one of the clauses defining ``prog.name`` (that is the shape every
    molecule-level rule uses, ``aux_x(M) :- has_atom(M,_), ...``); otherwise it is an atom.
    Across several defining clauses ``"mol"`` wins for a position. Returns ``[]`` when the
    program does not define its own predicate as a clause head.

    This is what tells :func:`~chebILP.ilp_classifier.build_seed_hypothesis` whether a chosen
    predicate is used as ``aux_x(Molecule)`` or needs a ``has_atom(Molecule, Atom)`` link.
    """
    sorts_by_pos: dict[int, str] = {}
    arity: int | None = None
    for head, body in _clauses(prog.source):
        m = _PRED_RE.match(head)
        if m is None or m.group(1) != prog.name:
            continue
        args = _head_args(head)
        arity = len(args)
        mol_vars = set(_HAS_ATOM_MOL_RE.findall(body))
        for i, arg in enumerate(args):
            sort = "mol" if arg in mol_vars else "atom"
            if sorts_by_pos.get(i) != "mol":
                sorts_by_pos[i] = sort
    if arity is None:
        return []
    return [sorts_by_pos.get(i, "atom") for i in range(arity)]


# A body literal is "datalog-simple" — usable in a Popper hypothesis — when it is a plain
# positive predicate call: no negation, no aggregate (#), no comparison operator.
_SIMPLE_LITERAL_RE = re.compile(r"[a-z_][A-Za-z0-9_]*\s*\(.*\)$")


def analyze_hypothesis(clause_source: str) -> dict | None:
    """Structural summary of an LLM class hypothesis ``chebi_<id>(A) :- <body>.``.

    Returns ``{head: (name, arity), body_preds: {(name, arity), ...}, body_pred_names: [...],
    n_vars, n_body, datalog_ok}`` or ``None`` when no clause with a body can be read.
    ``datalog_ok`` is True only when every body literal is a plain positive predicate call —
    the fragment a Popper seed / prefer_body_pred hint can consume; an aggregate, negation or
    comparison sets it False. ``body_pred_names`` keeps first-seen order for stable hints.
    """
    clause = next((c for c in _clauses(clause_source) if c[1]), None)
    if clause is None:
        return None
    head, body = clause
    hm = _PRED_RE.match(head.strip())
    if hm is None:
        return None

    literals = _body_literals(body)
    body_preds: set[tuple[str, int]] = set()
    names_in_order: list[str] = []
    datalog_ok = True
    for lit in literals:
        stripped = lit.strip()
        if _NOT_RE.match(stripped) or "#" in stripped or not _SIMPLE_LITERAL_RE.match(stripped):
            datalog_ok = False
            continue
        m = _PRED_RE.match(stripped)
        if m is None:
            datalog_ok = False
            continue
        args = _head_args(stripped)
        body_preds.add((m.group(1), len(args)))
        if m.group(1) not in names_in_order:
            names_in_order.append(m.group(1))

    return {
        "head": (hm.group(1), len(_head_args(head))),
        "clause": f"{head.strip()} :- {body.strip()}.",
        "body_preds": body_preds,
        "body_pred_names": names_in_order,
        "n_vars": len(set(_VAR_RE.findall(f"{head} {body}"))),
        "n_body": len(literals),
        "datalog_ok": datalog_ok,
    }


def save_class_hypothesis(chebi_id, entry: dict, library_dir: str | None = None) -> None:
    """Record one class's LLM hypothesis (clause text + train-F1 metrics) in the library."""
    import json

    library_dir = library_dir or DEFAULT_AUX_RULE_LIBRARY_DIR
    path = get_aux_hypotheses_path(base_dir=library_dir)
    store: dict = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)
    store[str(chebi_id)] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def load_class_hypothesis(chebi_id, library_dir: str | None = None) -> dict | None:
    """The saved LLM hypothesis entry for a class, or ``None`` if none was recorded."""
    import json

    library_dir = library_dir or DEFAULT_AUX_RULE_LIBRARY_DIR
    path = get_aux_hypotheses_path(base_dir=library_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get(str(chebi_id))


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
