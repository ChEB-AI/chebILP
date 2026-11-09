"""A small IR for learned rules and aux programs, plus aux unfolding.

Parsing goes through clingo.ast, so aggregates, negation, comparisons and arithmetic never
pass through a hand-written parser. Terms are kept as strings (``str`` of the clingo AST).

Body items:
  Lit(pred, args, neg)          p(X,Y) / not p(X,Y)
  Cmp(left, op, right)          X != Y, L = L0 + 1, K < 9
  Agg(func, elems, lguard, rguard)
      func in {"count","sum","min","max",...}; elems = [AggElem(terms, cond)], cond a list
      of Lit/Cmp; guards are (op, term) or None, e.g. 22 = #count{...} -> lguard ("=", "22")
  NegGroup(alternatives)        not aux(X) after unfolding: "none of these bodies holds";
                                each alternative is a list of body items whose variables
                                not shared with the enclosing clause are existential.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from clingo.ast import AggregateFunction, ASTType, ComparisonOperator, parse_string

_OPS = {
    ComparisonOperator.Equal: "=",
    ComparisonOperator.NotEqual: "!=",
    ComparisonOperator.LessThan: "<",
    ComparisonOperator.LessEqual: "<=",
    ComparisonOperator.GreaterThan: ">",
    ComparisonOperator.GreaterEqual: ">=",
}
_AGG_FUNCS = {
    AggregateFunction.Count: "count",
    AggregateFunction.Sum: "sum",
    AggregateFunction.SumPlus: "sum+",
    AggregateFunction.Min: "min",
    AggregateFunction.Max: "max",
}
_VAR_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class Lit:
    pred: str
    args: tuple[str, ...]
    neg: bool = False

    def __str__(self):
        a = f"{self.pred}({','.join(self.args)})" if self.args else self.pred
        return f"not {a}" if self.neg else a


@dataclass(frozen=True)
class Cmp:
    left: str
    op: str
    right: str

    def __str__(self):
        return f"{self.left} {self.op} {self.right}"


@dataclass(frozen=True)
class AggElem:
    terms: tuple[str, ...]
    cond: tuple  # of Lit / Cmp

    def __str__(self):
        return f"{','.join(self.terms)} : {', '.join(map(str, self.cond))}"


@dataclass(frozen=True)
class Agg:
    func: str
    elems: tuple[AggElem, ...]
    lguard: tuple[str, str] | None = None  # (op, term): "term op #agg"
    rguard: tuple[str, str] | None = None  # (op, term): "#agg op term"

    def __str__(self):
        s = f"#{self.func}{{ {'; '.join(map(str, self.elems))} }}"
        if self.lguard:
            s = f"{self.lguard[1]} {self.lguard[0]} {s}"
        if self.rguard:
            s = f"{s} {self.rguard[0]} {self.rguard[1]}"
        return s


@dataclass(frozen=True)
class NegGroup:
    alternatives: tuple[tuple, ...]  # each a tuple of body items
    source_pred: str = ""

    def __str__(self):
        alts = " ; ".join("(" + ", ".join(map(str, a)) + ")" for a in self.alternatives)
        return f"not [{self.source_pred}: {alts}]"


@dataclass
class Clause:
    head: Lit
    body: list = field(default_factory=list)

    def __str__(self):
        return f"{self.head} :- {', '.join(map(str, self.body))}."

    def variables(self) -> list[str]:
        return ordered_vars([self.head] + list(self.body))


def ordered_vars(items) -> list[str]:
    seen, out = set(), []
    for it in items:
        for v in _VAR_RE.findall(str(it)):
            if v != "_" and v not in seen:
                seen.add(v)
                out.append(v)
    return out


# ── parsing ────────────────────────────────────────────────────────────────


def _lit_from_ast(lit) -> Lit | Cmp | Agg:
    atom = lit.atom
    neg = lit.sign != 0
    if atom.ast_type == ASTType.SymbolicAtom:
        sym = atom.symbol
        if sym.ast_type == ASTType.Function:
            return Lit(sym.name, tuple(str(a) for a in sym.arguments), neg)
        return Lit(str(sym), (), neg)
    if atom.ast_type == ASTType.Comparison:
        # clingo >= 5.6: Comparison(term, guards); one guard in all our programs
        left = str(atom.term)
        g = atom.guards[0]
        op = _OPS[g.comparison]
        if neg:
            op = {"=": "!=", "!=": "=", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}[op]
        return Cmp(left, op, str(g.term))
    if atom.ast_type == ASTType.BodyAggregate:
        elems = []
        for e in atom.elements:
            cond = tuple(_lit_from_ast(c) for c in e.condition)
            elems.append(AggElem(tuple(str(t) for t in e.terms), cond))
        lg = (
            (_OPS[atom.left_guard.comparison], str(atom.left_guard.term))
            if atom.left_guard
            else None
        )
        rg = (
            (_OPS[atom.right_guard.comparison], str(atom.right_guard.term))
            if atom.right_guard
            else None
        )
        func = _AGG_FUNCS[atom.function]
        return Agg(func, tuple(elems), lg, rg)
    raise ValueError(f"unsupported body atom: {lit}")


def parse_program(text: str) -> list[Clause]:
    """All clauses of an ASP program (comments, #show etc. ignored)."""
    clauses: list[Clause] = []

    def cb(stm):
        if stm.ast_type != ASTType.Rule:
            return
        head = stm.head
        if head.ast_type != ASTType.Literal:
            return  # constraints / choice heads: not used by learned rules or aux programs
        if head.atom.ast_type == ASTType.BooleanConstant:
            return
        h = _lit_from_ast(head)
        body = []
        for b in stm.body:
            if b.ast_type == ASTType.Literal:
                body.append(_lit_from_ast(b))
            else:
                raise ValueError(f"unsupported body element: {b}")
        clauses.append(Clause(h, body))

    parse_string(text, cb)
    return clauses


def definitions(programs) -> dict[str, list[Clause]]:
    """``{pred: [clauses]}`` for every head in ``programs`` (strings or objects with .source)."""
    defs: dict[str, list[Clause]] = {}
    for p in programs:
        src = (
            p
            if isinstance(p, str)
            else (p["source"] if isinstance(p, dict) else p.source)
        )
        for c in parse_program(src):
            defs.setdefault(c.head.pred, []).append(c)
    return defs


def recursive_preds(defs: dict[str, list[Clause]]) -> set[str]:
    """Predicates that (transitively) depend on themselves."""

    def deps(p):
        out = set()
        for c in defs.get(p, []):
            for it in c.body:
                out |= _preds_in(it)
        return out & set(defs)

    rec = set()
    for p in defs:
        stack, seen = list(deps(p)), set()
        while stack:
            q = stack.pop()
            if q == p:
                rec.add(p)
                break
            if q in seen:
                continue
            seen.add(q)
            stack.extend(deps(q))
    return rec


def _preds_in(item) -> set[str]:
    if isinstance(item, Lit):
        return {item.pred}
    if isinstance(item, Agg):
        return {p for e in item.elems for c in e.cond for p in _preds_in(c)}
    if isinstance(item, NegGroup):
        return {p for alt in item.alternatives for c in alt for p in _preds_in(c)}
    return set()


# ── substitution / unfolding ──────────────────────────────────────────────


def _subst_term(term: str, sub: dict[str, str]) -> str:
    return _VAR_RE.sub(lambda m: sub.get(m.group(1), m.group(1)), term)


def substitute(item, sub: dict[str, str]):
    if isinstance(item, Lit):
        return Lit(item.pred, tuple(_subst_term(a, sub) for a in item.args), item.neg)
    if isinstance(item, Cmp):
        return Cmp(_subst_term(item.left, sub), item.op, _subst_term(item.right, sub))
    if isinstance(item, Agg):
        elems = tuple(
            AggElem(
                tuple(_subst_term(t, sub) for t in e.terms),
                tuple(substitute(c, sub) for c in e.cond),
            )
            for e in item.elems
        )
        lg = (item.lguard[0], _subst_term(item.lguard[1], sub)) if item.lguard else None
        rg = (item.rguard[0], _subst_term(item.rguard[1], sub)) if item.rguard else None
        return Agg(item.func, elems, lg, rg)
    if isinstance(item, NegGroup):
        return NegGroup(
            tuple(tuple(substitute(c, sub) for c in alt) for alt in item.alternatives),
            item.source_pred,
        )
    raise TypeError(item)


class _Fresh:
    def __init__(self, taken):
        self.taken = set(taken)
        self.n = 0

    def __call__(self, base: str) -> str:
        base = "X" if base == "_" else re.sub(r"\d+$", "", base) or "X"
        while True:
            self.n += 1
            cand = f"{base}_{self.n}"
            if cand not in self.taken:
                self.taken.add(cand)
                return cand


def _rename_apart(clause: Clause, fresh: _Fresh) -> Clause:
    """Copy of ``clause`` with every variable (and each ``_``) renamed to a fresh one."""
    sub = {v: fresh(v) for v in clause.variables()}

    def anon(term):  # each "_" occurrence is its own variable
        return re.sub(r"(?<![A-Za-z0-9_])_(?![A-Za-z0-9_])", lambda m: fresh("_"), term)

    def ren(item):
        item = substitute(item, sub)
        if isinstance(item, Lit):
            return Lit(item.pred, tuple(anon(a) for a in item.args), item.neg)
        return item

    return Clause(ren(clause.head), [ren(b) for b in clause.body])


def _unify_head(head: Lit, call: Lit) -> tuple[dict[str, str], list[Cmp]] | None:
    """Bind head variables to call arguments; constants/repeats become equality constraints."""
    if len(head.args) != len(call.args):
        return None
    sub, extra = {}, []
    for h, c in zip(head.args, call.args):
        if _VAR_RE.fullmatch(h):
            if h in sub and sub[h] != c:
                extra.append(Cmp(sub[h], "=", c))
            else:
                sub[h] = c
        elif h != c:
            extra.append(Cmp(h, "=", c))
    return sub, extra


@dataclass
class UnfoldPolicy:
    """When inlining an aux definition stops being reasonable."""

    max_clauses: int = 4  # after distributing multi-clause definitions
    max_body: int = 30  # body items per resulting clause
    inline_negation: bool = (
        True  # not aux(X) -> NegGroup when the definition is inlinable
    )
    max_neg_alternatives: int = 3
    keep: frozenset = frozenset()  # predicates never inlined


@dataclass
class UnfoldResult:
    clauses: list[Clause]
    inlined: list[str]  # predicates replaced by their definition
    kept: dict[str, str]  # predicate -> reason it stayed opaque


def unfold(
    rule_clauses: list[Clause],
    defs: dict[str, list[Clause]],
    policy: UnfoldPolicy | None = None,
) -> UnfoldResult:
    """Replace aux literals by their definitions as far as ``policy`` allows.

    Positive literals of a multi-clause definition distribute into several clauses (an OR).
    Recursive predicates are never inlined. Negated literals become a NegGroup.
    Aggregates are carried along: their conditions are unfolded too when they stay conjunctive.
    """
    policy = policy or UnfoldPolicy()
    rec = recursive_preds(defs)
    inlined: list[str] = []
    kept: dict[str, str] = {}

    def opaque_reason(pred):
        if pred not in defs:
            return None if not pred.startswith("aux_") else "no definition in library"
        if pred in policy.keep:
            return "kept by policy"
        if pred in rec:
            return "recursive definition"
        return None

    def expand_body(body: list, fresh: _Fresh, depth=0) -> list[list] | None:
        """All alternative bodies for ``body`` after unfolding; None if it exceeds the policy."""
        results = [[]]
        for item in body:
            options = expand_item(item, fresh, depth)
            results = [r + o for r in results for o in options]
            if len(results) > policy.max_clauses:
                return None
        return results

    def expand_item(item, fresh: _Fresh, depth) -> list[list]:
        if depth > 12:
            return [[item]]
        if isinstance(item, Lit) and item.pred in defs:
            reason = opaque_reason(item.pred)
            if reason:
                kept.setdefault(item.pred, reason)
                return [[item]]
            alts = []
            for d in defs[item.pred]:
                d = _rename_apart(d, fresh)
                u = _unify_head(d.head, Lit(item.pred, item.args))
                if u is None:
                    continue
                sub, extra = u
                body = [substitute(b, sub) for b in d.body] + extra
                sub_alts = expand_body(body, fresh, depth + 1)
                if sub_alts is None:
                    kept.setdefault(item.pred, "too many alternatives")
                    return [[item]]
                alts.extend(sub_alts)
            if not item.neg:
                if len(alts) > policy.max_clauses:
                    kept.setdefault(item.pred, "too many alternatives")
                    return [[item]]
                if item.pred not in inlined:
                    inlined.append(item.pred)
                return alts
            if not policy.inline_negation or len(alts) > policy.max_neg_alternatives:
                kept.setdefault(item.pred, "negated, too many alternatives")
                return [[item]]
            if item.pred not in inlined:
                inlined.append(item.pred)
            return [[NegGroup(tuple(tuple(a) for a in alts), item.pred)]]
        if isinstance(item, Agg):
            new_elems = []
            for e in item.elems:
                mark = len(inlined)
                alts = expand_body(list(e.cond), fresh, depth + 1)
                if alts is None or len(alts) != 1:
                    del inlined[mark:]
                    new_elems.append(e)  # a disjunctive condition stays as written
                else:
                    new_elems.append(AggElem(e.terms, tuple(alts[0])))
            return [[Agg(item.func, tuple(new_elems), item.lguard, item.rguard)]]
        return [[item]]

    out: list[Clause] = []
    for c in rule_clauses:
        fresh = _Fresh(c.variables())
        alts = expand_body(list(c.body), fresh)
        if alts is None or any(len(a) > policy.max_body for a in alts):
            # fall back: unfold nothing in this clause
            out.append(c)
            continue
        out.extend(Clause(c.head, a) for a in alts)
    if len(out) > policy.max_clauses * max(1, len(rule_clauses)):
        return UnfoldResult(rule_clauses, [], {"*": "too many clauses after unfolding"})
    return UnfoldResult(out, inlined, kept)


def simplify(clause: Clause) -> Clause:
    """Drop duplicate items and ``has_atom(M,X)`` guards whose X is used nowhere else.

    Every molecule has atoms, so such a guard is always true; it only keeps the clause safe.
    For display only: the result may be unsafe ASP (use to_asp on unsimplified clauses).
    """
    seen, body = set(), []
    for b in clause.body:
        if str(b) not in seen:
            seen.add(str(b))
            body.append(b)
    mol = clause.head.args[0] if clause.head.args else None
    counts: dict[str, int] = {}
    for b in [clause.head] + body:
        for v in _VAR_RE.findall(str(b)):
            counts[v] = counts.get(v, 0) + 1

    def lonely(b):
        return (
            isinstance(b, Lit)
            and b.pred == "has_atom"
            and not b.neg
            and b.args[0] == mol
            and (b.args[1] == "_" or counts.get(b.args[1], 0) == 1)
        )

    pruned = [b for b in body if not lonely(b)]
    return Clause(clause.head, pruned if pruned else body)


def to_asp(clauses: list[Clause]) -> str:
    """Executable ASP for unfolded clauses; each NegGroup becomes a helper predicate."""
    lines, counter = [], itertools.count()

    def emit(item, outer_vars):
        if not isinstance(item, NegGroup):
            return str(item)
        k = next(counter)
        alt_vars = ordered_vars([c for alt in item.alternatives for c in alt])
        shared = [v for v in alt_vars if v in outer_vars]
        head = f"_neg{k}({','.join(shared)})" if shared else f"_neg{k}"
        for alt in item.alternatives:
            inner_outer = set(ordered_vars(alt))
            lines.append(f"{head} :- {', '.join(emit(c, inner_outer) for c in alt)}.")
        return f"not {head}"

    for c in clauses:
        outer = set()
        for b in [c.head] + [b for b in c.body if not isinstance(b, NegGroup)]:
            outer |= set(ordered_vars([b]))
        lines.append(f"{c.head} :- {', '.join(emit(b, outer) for b in c.body)}.")
    return "\n".join(lines)
