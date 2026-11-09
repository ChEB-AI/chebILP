"""Near misses: the binding of a clause that satisfies the most body items, found with clingo optimisation.

Every body item becomes ``sat(i)`` under a chosen binding ``val(Var, Atom)``; the solver maximises the
weighted number of satisfied items, then "almost satisfied" items (an atom with a property of the
same kind, e.g. another steroid position), then the number of distinct atoms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import clingo

from chebILP.explainability.molecule import MOL_ID
from chebILP.explainability.rule_ir import (
    Agg,
    Clause,
    Cmp,
    Lit,
    NegGroup,
    ordered_vars,
    substitute,
    to_asp,
)
from chebILP.explainability.molecule import BOND_PREDS, MOL_PREDS, MolContext, is_var
from chebILP.explainability.tagged import TClause

FAMILIES = [
    ("steroid", re.compile(r"steroid_\d+$")),
    ("ring", re.compile(r"in_ring\d*$")),
    ("cip", re.compile(r"cip_code_[RS]$")),
    ("charge", re.compile(r"charge(_m)?\d+$")),
    ("hcount", re.compile(r"has_\d+_hs$")),
]


def family(pred: str):
    for name, rx in FAMILIES:
        if rx.match(pred):
            return name
    return None


@dataclass
class NearMiss:
    tclause: TClause
    binding: dict
    sat: list  # indices into tclause.body
    unsat: list
    score: float  # satisfied weight / total weight
    anchored: set = field(default_factory=set)  # variables fixed by a satisfied item

    @property
    def holds(self):
        return not self.unsat


def weight(item, molvar) -> int:
    if isinstance(item, Lit) and item.pred == "has_atom" and not item.neg:
        return 0
    if isinstance(item, Lit) and (
        item.pred in BOND_PREDS or re.fullmatch(r"steroid_\d+", item.pred)
    ):
        return 2
    if isinstance(item, Cmp) and item.op not in ("!=", "="):
        return 0  # orderings such as A < B only break symmetry
    return 1


def _outer_vars(tc: TClause) -> list[str]:
    return [
        v
        for v in ordered_vars([b for b, _ in tc.body if isinstance(b, (Lit, Cmp))])
        if v != tc.mol_var
    ]


def near_miss(tc: TClause, ctx: MolContext, timeout: float = 8.0) -> NearMiss:
    molvar = tc.mol_var
    vars_ = _outer_vars(tc)
    sub_m = {molvar: MOL_ID}
    asp = ["dom(A) :- has_atom(m,A)."]
    for k, v in enumerate(vars_):
        asp.append(f"1 {{ val({k},A) : dom(A) }} 1.")
    vidx = {v: k for k, v in enumerate(vars_)}
    weights = []
    helpers = []
    for i, (item, _) in enumerate(tc.body):
        w = weight(item, molvar)
        weights.append(w)
        item = substitute(item, sub_m)
        if isinstance(item, (Lit, Cmp)):
            vs = [v for v in ordered_vars([item]) if v in vidx]
            guard = [f"val({vidx[v]},{v})" for v in vs]
            asp.append(f"sat({i}) :- {', '.join(guard + [str(item)])}.")
            if (
                isinstance(item, Lit)
                and not item.neg
                and len(vs) == 1
                and len(item.args) == 1
                and family(item.pred)
            ):
                fam = family(item.pred)
                asp.append(
                    f"near({i}) :- {guard[0]}, fam_{fam}({vs[0]}), not sat({i})."
                )
            if (
                isinstance(item, Lit)
                and not item.neg
                and item.pred in BOND_PREDS
                and len(vs) == 2
            ):
                a, b = item.args
                asp.append(
                    f"near({i}) :- {', '.join(guard)}, has_bond_to({a},{b}), not sat({i})."
                )
        elif isinstance(item, NegGroup):
            shared = [
                v
                for v in ordered_vars([c for alt in item.alternatives for c in alt])
                if v in vidx
            ]
            head = f"_ng{i}({','.join(shared)})" if shared else f"_ng{i}"
            for alt in item.alternatives:
                helpers.append(
                    to_asp([Clause(Lit(f"_ng{i}", tuple(shared)), list(alt))])
                )
            guard = [f"val({vidx[v]},{v})" for v in shared]
            asp.append(
                f"sat({i}) :- {', '.join(guard + ['not ' + head]) if guard else 'not ' + head}."
            )
        elif isinstance(item, Agg):
            body = to_asp([Clause(Lit(f"_ag{i}", ()), [item])])
            helpers.append(body)
            asp.append(f"sat({i}) :- _ag{i}.")
        else:
            asp.append(f"sat({i}).")
    fams = []
    for name, rx in FAMILIES:
        for p in ctx.model:
            if rx.match(p):
                fams.append(f"fam_{name}(A) :- {p}(A).")
        fams.append(f"fam_{name}(_dummy) :- #false.")
    asp.append("#maximize { W@2,I : sat(I), w(I,W) }.")
    asp.append("#maximize { 1@1,I : near(I) }.")
    asp.append("#maximize { 1@0,A : val(_,A) }.")
    asp += [f"w({i},{w})." for i, w in enumerate(weights)]
    prog = (
        ctx.model_text
        + "\n"
        + "\n".join(fams)
        + "\n"
        + "\n".join(helpers)
        + "\n"
        + "\n".join(asp)
        + "\n#show val/2. #show sat/1."
    )
    ctl = clingo.Control(["--warn=none", "--opt-mode=opt"])
    ctl.add("base", [], prog)
    ctl.ground([("base", [])])
    best = {"syms": None}

    def on_model(m):
        best["syms"] = [str(s) for s in m.symbols(shown=True)]

    with ctl.solve(on_model=on_model, async_=True) as h:
        if not h.wait(timeout):
            h.cancel()
        h.get() if h.wait(0) else None
    syms = best["syms"] or []
    binding = {molvar: MOL_ID}
    sat = set()
    for s in syms:
        t = clingo.parse_term(s)
        if t.name == "val":
            binding[vars_[t.arguments[0].number]] = str(t.arguments[1])
        elif t.name == "sat":
            sat.add(t.arguments[0].number)
    total = sum(weights) or 1
    sat_w = sum(weights[i] for i in sat)
    unsat = [i for i in range(len(tc.body)) if i not in sat and weights[i] > 0]
    anchored = set()
    for i in sat:
        item = tc.body[i][0]
        if weights[i] == 0 or isinstance(item, Cmp):
            continue
        if isinstance(item, Lit):
            anchored.update(a for a in item.args if is_var(a) and a != molvar)
    return NearMiss(tc, binding, sorted(sat), unsat, sat_w / total, anchored)


def rank_key(nm: NearMiss):
    """Fewer failing conditions first, then a larger satisfied share of the clause."""
    return (len(nm.unsat), -nm.score)


def rule_holds(tclauses, ctx: MolContext) -> bool:
    from chebILP.explainability.molecule import enumerate_bindings

    return any(
        enumerate_bindings(tc.clause(), ctx, limit=1, timeout=10) for tc in tclauses
    )


def mol_level(item, molvar) -> bool:
    return isinstance(item, Lit) and (
        item.pred in MOL_PREDS or set(item.args) <= {molvar}
    )
