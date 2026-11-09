"""Display-oriented unfolding of aux predicates, executable ASP for it, and witnesses.

Differs from ``common.rule_ir.unfold`` in what it produces for drawing:
- a positive multi-clause aux becomes an ``OrGroup`` inside the clause (distributed into
  separate clauses only when the rest of the clause is trivial);
- alternatives that differ only in one element literal merge into ``elem_in(X, "n|o")``;
- unary predicates defined only by element literals (``aux_o_or_n``) stay as literals and
  are read as element sets by the pattern builder;
- recursive predicates with an integer counter argument are unrolled when the call gives
  the counter as a constant (``aux_dec_lin(A,_,9)``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import clingo
from rdkit import Chem

from chebILP.explainability.rule_ir import (
    Agg,
    AggElem,
    Clause,
    Cmp,
    Lit,
    NegGroup,
    _Fresh,
    _rename_apart,
    _VAR_RE,
    ordered_vars,
    recursive_preds,
    substitute,
)

_pt = Chem.GetPeriodicTable()
ELEMENTS = {_pt.GetElementSymbol(z).lower() for z in range(1, 119)}
_INT = re.compile(r"-?\d+")
_PLUS = re.compile(r"^\(?\s*([A-Z][A-Za-z0-9_]*)\s*\+\s*(\d+)\s*\)?$")


@dataclass(frozen=True)
class OrGroup:
    alternatives: tuple[tuple, ...]
    source_pred: str = ""

    def __str__(self):
        alts = " ; ".join("(" + ", ".join(map(str, a)) + ")" for a in self.alternatives)
        return f"one of [{self.source_pred}: {alts}]"


def is_var(t: str) -> bool:
    return bool(re.fullmatch(r"[A-Z_][A-Za-z0-9_]*", t))


def item_vars(item) -> list[str]:
    return [v for v in ordered_vars([item]) if v != "_"]


def elemset_preds(defs) -> dict[str, frozenset]:
    """Unary predicates whose every clause is ``p(X) :- e(X).`` for an element e."""
    out = {}
    for p, cls in defs.items():
        els = set()
        for c in cls:
            b = c.body[0] if len(c.body) == 1 else None
            if not (
                len(c.head.args) == 1
                and isinstance(b, Lit)
                and not b.neg
                and b.pred in ELEMENTS
                and b.args == c.head.args
            ):
                break
            els.add(b.pred)
        else:
            if els:
                out[p] = frozenset(els)
    return out


def _eval_ground(expr: str):
    if re.fullmatch(r"[\d\s+\-*()]+", expr):
        try:
            return int(eval(expr, {"__builtins__": {}}))  # digits and + - * only
        except Exception:
            return None
    return None


def eval_cmp(c: Cmp):
    """True/False for a ground integer comparison, None otherwise."""
    a, b = _eval_ground(c.left), _eval_ground(c.right)
    if a is None or b is None:
        return None
    return {
        "=": a == b,
        "!=": a != b,
        "<": a < b,
        "<=": a <= b,
        ">": a > b,
        ">=": a >= b,
    }[c.op]


def counter_values(pred: str, defs, limit: int = 60) -> dict[int, set[int]]:
    """Derivable values of each integer counter position of a recursive predicate."""
    clauses = defs[pred]
    out = {}
    for pos in range(len(clauses[0].head.args)):
        kinds = []
        for c in clauses:
            h = c.head.args[pos]
            if _INT.fullmatch(h):
                kinds.append(("const", int(h)))
                continue
            m = _PLUS.match(h)
            rec = [
                b for b in c.body if isinstance(b, Lit) and b.pred == pred and not b.neg
            ]
            if m and len(rec) == 1 and rec[0].args[pos] == m.group(1):
                v = m.group(1)
                guards = [
                    b for b in c.body if isinstance(b, Cmp) and set(item_vars(b)) == {v}
                ]
                kinds.append(("step", (v, int(m.group(2)), guards)))
                continue
            break
        else:
            vals: set[int] = set()
            for _ in range(limit):
                new = set(vals)
                for kind, data in kinds:
                    if kind == "const":
                        new.add(data)
                    else:
                        v, k, guards = data
                        for x in vals:
                            if all(
                                eval_cmp(substitute(g, {v: str(x)})) for g in guards
                            ):
                                new.add(x + k)
                if new == vals or len(new) > limit:
                    break
                vals = new
            if vals and len(vals) <= limit:
                out[pos] = vals
    return out


def _unify(head: Lit, call: Lit):
    if len(head.args) != len(call.args):
        return None
    sub, extra = {}, []
    for h, c in zip(head.args, call.args):
        m = _PLUS.match(h)
        if is_var(h):
            if h in sub and sub[h] != c:
                extra.append(Cmp(sub[h], "=", c))
            else:
                sub[h] = c
        elif m and _INT.fullmatch(c):
            v, val = m.group(1), str(int(c) - int(m.group(2)))
            if v in sub and sub[v] != val:
                extra.append(Cmp(sub[v], "=", val))
            else:
                sub[v] = val
        elif h != c:
            extra.append(Cmp(h, "=", c))
    return sub, extra


def _ground_simplify(body: list):
    """Drop true ground comparisons; None if one is false."""
    out = []
    for b in body:
        if isinstance(b, Cmp):
            r = eval_cmp(b)
            if r is False:
                return None
            if r is True:
                continue
        out.append(b)
    return out


def _same_modulo_vars(a, b, bij: dict) -> bool:
    """Items equal up to a consistent variable renaming (extends ``bij``)."""
    sa, sb = str(a), str(b)
    va, vb = _VAR_RE.findall(sa), _VAR_RE.findall(sb)
    if len(va) != len(vb) or _VAR_RE.sub("V", sa) != _VAR_RE.sub("V", sb):
        return False
    for x, y in zip(va, vb):
        if bij.setdefault(x, y) != y:
            return False
    return True


def merge_element_alternatives(alts: list[list]):
    """``[..., e1(X), ...] ; [..., e2(X'), ...]`` -> one body with ``elem_in(X, "e1|e2")``."""
    if len(alts) < 2 or len({len(a) for a in alts}) != 1:
        return None
    base = alts[0]
    diff_pos = None
    els = []
    for alt in alts[1:]:
        bij: dict = {}
        diffs = [
            i
            for i, (x, y) in enumerate(zip(base, alt))
            if not _same_modulo_vars(x, y, bij)
        ]
        if len(diffs) != 1 or (diff_pos is not None and diffs[0] != diff_pos):
            return None
        i = diffs[0]
        x, y = base[i], alt[i]
        if not all(
            isinstance(z, Lit) and not z.neg and z.pred in ELEMENTS and len(z.args) == 1
            for z in (x, y)
        ):
            return None
        if bij.get(x.args[0], y.args[0]) != y.args[0]:
            return None  # the element literals must sit on corresponding variables
        diff_pos = i
        els.append(y.pred)
    x = base[diff_pos]
    allels = sorted({x.pred, *els})
    merged = list(base)
    merged[diff_pos] = Lit("elem_in", (x.args[0], "|".join(allels)))
    return merged


@dataclass
class DisplayUnfold:
    clauses: list[Clause]
    inlined: list[str]
    kept: dict[str, str]
    elemsets: dict[str, frozenset]


def display_unfold(
    rule_clauses: list[Clause], defs, max_alts: int = 4, max_neg_alts: int = 3
) -> DisplayUnfold:
    elemsets = elemset_preds(defs)
    rec = recursive_preds(defs)
    counters = {p: counter_values(p, defs) for p in rec}
    inlined: list[str] = []
    kept: dict[str, str] = {}

    def expand_lit(lit: Lit, fresh, depth):
        """Alternatives (list of bodies) for a positive call; None keeps the literal opaque."""
        p = lit.pred
        if p in elemsets or p not in defs:
            if p not in defs and p.startswith("aux_"):
                kept.setdefault(p, "no definition in library")
            return None
        if depth > 40:
            kept.setdefault(p, "nesting too deep")
            return None
        if p in rec:
            cpos = [i for i in counters.get(p, {}) if _INT.fullmatch(lit.args[i])]
            if not cpos:
                kept.setdefault(p, "recursive definition")
                return None
            if any(int(lit.args[i]) not in counters[p][i] for i in cpos):
                return []
        alts = []
        for d in defs[p]:
            d = _rename_apart(d, fresh)
            u = _unify(d.head, Lit(p, lit.args))
            if u is None:
                continue
            sub, extra = u
            body = _ground_simplify([substitute(b, sub) for b in d.body] + extra)
            if body is None:
                continue
            body = _ground_simplify(expand_body(body, fresh, depth + 1))
            if body is not None:
                alts.append(body)
        if p not in inlined:
            inlined.append(p)
        return alts

    def expand_body(body, fresh, depth):
        out = []
        for item in body:
            out.extend(expand_item(item, fresh, depth))
        return out

    def expand_item(item, fresh, depth) -> list:
        if isinstance(item, Lit):
            alts = expand_lit(item, fresh, depth)
            if alts is None:
                return [item]
            if not item.neg:
                if len(alts) == 0:
                    return [Cmp("0", "=", "1")]
                if len(alts) == 1:
                    return alts[0]
                merged = merge_element_alternatives(alts)
                if merged is not None:
                    return merged
                if len(alts) > max_alts:
                    inlined.remove(item.pred) if item.pred in inlined else None
                    kept.setdefault(item.pred, "too many alternatives")
                    return [item]
                return [OrGroup(tuple(tuple(a) for a in alts), item.pred)]
            if len(alts) == 0:
                return []
            if len(alts) > max_neg_alts:
                inlined.remove(item.pred) if item.pred in inlined else None
                kept.setdefault(item.pred, "negated, too many alternatives")
                return [item]
            merged = merge_element_alternatives(alts)
            if merged is not None:
                alts = [merged]
            return [NegGroup(tuple(tuple(a) for a in alts), item.pred)]
        if isinstance(item, Agg):
            elems = tuple(
                AggElem(e.terms, tuple(expand_body(list(e.cond), fresh, depth + 1)))
                for e in item.elems
            )
            return [Agg(item.func, elems, item.lguard, item.rguard)]
        return [item]

    out: list[Clause] = []
    for c in rule_clauses:
        fresh = _Fresh(c.variables())
        out.extend(
            distribute_trivial(
                Clause(c.head, expand_body(list(c.body), fresh, 0)), elemsets
            )
        )
    return DisplayUnfold(out, inlined, kept, elemsets)


MOL_PREDS = {
    "aromatic",
    "aliphatic",
    "net_charge_positive",
    "net_charge_negative",
    "net_charge_neutral",
}


def atom_vars_outside(body: list, mol: str, skip=None) -> set[str]:
    """Atom variables used by items of ``body`` other than ``skip`` (has_atom guards excluded)."""
    vs = set()
    for b in body:
        if b is skip:
            continue
        if isinstance(b, Lit) and b.pred == "has_atom" and not b.neg:
            continue
        if isinstance(b, Lit) and b.pred in MOL_PREDS:
            continue
        vs |= set(item_vars(b))
    vs.discard(mol)
    return vs


def distribute_trivial(clause: Clause, elemsets) -> list[Clause]:
    """Split a top-level OrGroup into clauses when the rest of the clause has at most one atom."""
    mol = clause.head.args[0] if clause.head.args else None
    for i, b in enumerate(clause.body):
        if isinstance(b, OrGroup):
            rest = atom_vars_outside(clause.body, mol, skip=b)
            if len(rest) <= 1:
                out = []
                for alt in b.alternatives:
                    body = clause.body[:i] + list(alt) + clause.body[i + 1 :]
                    out.extend(distribute_trivial(Clause(clause.head, body), elemsets))
                return out
    return [clause]


def distribute_all(clause: Clause) -> list[Clause]:
    """Every OrGroup distributed (for validation against the original rule)."""
    for i, b in enumerate(clause.body):
        if isinstance(b, OrGroup):
            out = []
            for alt in b.alternatives:
                out.extend(
                    distribute_all(
                        Clause(
                            clause.head,
                            clause.body[:i] + list(alt) + clause.body[i + 1 :],
                        )
                    )
                )
            return out
    return [clause]


# ── executable ASP ─────────────────────────────────────────────────────────


def to_asp2(clauses: list[Clause], head_override: Lit | None = None) -> str:
    """ASP for display clauses; NegGroup / OrGroup / elem_in become helper predicates."""
    lines: list[str] = ["_dom(X) :- has_atom(X,_).", "_dom(X) :- has_atom(_,X)."]
    k = [0]

    def scope(items) -> set[str]:
        out = set()
        for it in items:
            if isinstance(it, Agg):
                out |= {
                    v
                    for g in (it.lguard, it.rguard)
                    if g
                    for v in _VAR_RE.findall(g[1])
                }
            elif not isinstance(it, (NegGroup, OrGroup)):
                out |= set(ordered_vars([it]))
        return out

    def helper(prefix):
        k[0] += 1
        return f"_{prefix}{k[0]}"

    def emit(item, outer: set[str]) -> str:
        if isinstance(item, (NegGroup, OrGroup)):
            gvars = ordered_vars([c for alt in item.alternatives for c in alt])
            shared = [v for v in gvars if v in outer]
            name = helper("neg" if isinstance(item, NegGroup) else "or")
            head = f"{name}({','.join(shared)})" if shared else name
            for alt in item.alternatives:
                inner = set(shared) | scope(alt)
                dom = [f"_dom({v})" for v in shared]
                lines.append(
                    f"{head} :- {', '.join(dom + [emit(c, inner) for c in alt]) or '#true'}."
                )
            return f"not {head}" if isinstance(item, NegGroup) else head
        if isinstance(item, Lit) and item.pred == "elem_in":
            name = helper("el")
            for e in item.args[1].split("|"):
                lines.append(f"{name}(X) :- {e}(X).")
            return f"{'not ' if item.neg else ''}{name}({item.args[0]})"
        if isinstance(item, Agg):
            elems = []
            for e in item.elems:
                inner = outer | scope(e.cond)
                elems.append(
                    f"{','.join(e.terms)} : {', '.join(emit(c, inner) for c in e.cond)}"
                )
            s = f"#{item.func}{{ {'; '.join(elems)} }}"
            if item.lguard:
                s = f"{item.lguard[1]} {item.lguard[0]} {s}"
            if item.rguard:
                s = f"{s} {item.rguard[0]} {item.rguard[1]}"
            return s
        return str(item)

    for c in clauses:
        head = head_override or c.head
        outer = set(ordered_vars([head])) | scope(c.body)
        lines.append(
            f"{head} :- {', '.join(emit(b, outer) for b in c.body) or '#true'}."
        )
    return "\n".join(lines)


def bound_vars(clause: Clause) -> list[str]:
    """Variables bound by positive literals of the clause body (plus the head)."""
    items = [clause.head] + [b for b in clause.body if isinstance(b, Lit) and not b.neg]
    return [v for v in ordered_vars(items) if v != "_"]


def witnesses(
    clause: Clause,
    facts: list[str],
    extra_program: str = "",
    distinct: list[str] | None = None,
    limit: int = 50,
) -> list[dict]:
    """Bindings of the clause's bound variables satisfying its body; ``distinct`` vars pairwise different."""
    vs = bound_vars(clause)
    body = list(clause.body)
    if distinct:
        d = [v for v in distinct if v in vs]
        body += [Cmp(a, "!=", b) for i, a in enumerate(d) for b in d[i + 1 :]]
    prog = to_asp2([Clause(clause.head, body)], head_override=Lit("_wit", tuple(vs)))
    ctl = clingo.Control(["--warn=none", "--models=1"])
    ctl.add(
        "base",
        [],
        "\n".join(facts)
        + "\n"
        + extra_program
        + "\n"
        + prog
        + f"\n#show _wit/{len(vs)}.",
    )
    ctl.ground([("base", [])])
    out = []
    with ctl.solve(yield_=True) as h:
        for m in h:
            for s in sorted(m.symbols(shown=True), key=str)[:limit]:
                out.append(dict(zip(vs, (str(a) for a in s.arguments))))
    return out


def fires(clauses: list[Clause], facts: list[str], extra_program: str = "") -> bool:
    prog = to_asp2(clauses, head_override=Lit("_fires", ()))
    ctl = clingo.Control(["--warn=none", "--models=1"])
    ctl.add(
        "base",
        [],
        "\n".join(facts) + "\n" + extra_program + "\n" + prog + "\n#show _fires/0.",
    )
    ctl.ground([("base", [])])
    res = []
    with ctl.solve(yield_=True) as h:
        for m in h:
            res = list(m.symbols(shown=True))
    return bool(res)
