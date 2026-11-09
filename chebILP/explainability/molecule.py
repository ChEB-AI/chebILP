"""One molecule as a logic model: BK facts, aux extensions, clause bindings, proof traces, counted sets."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import clingo
import numpy as np
import pandas as pd
from clingo.ast import ASTType, Sign, Transformer, parse_string
from rdkit import Chem

from chebILP.explainability.rule_ir import (
    Agg,
    Clause,
    Lit,
    NegGroup,
    ordered_vars,
    substitute,
    to_asp,
)
from chebILP.preprocessing.chebi.mol_to_fol import (
    build_background_chemlog,
    build_computed_facts,
)

MOL_ID = "m"  # molecule id in the BK of an explained molecule; its atoms are am_<idx+1>
ELEMENTS = {Chem.GetPeriodicTable().GetElementSymbol(z).lower() for z in range(1, 119)}
BOND_PREDS = {
    "bSINGLE",
    "bDOUBLE",
    "bTRIPLE",
    "bAROMATIC",
    "has_bond_to",
    "bSTEREOE",
    "bSTEREOZ",
    "bSTEREOANY",
}
MOL_PREDS = {
    "aromatic",
    "aliphatic",
    "net_charge_positive",
    "net_charge_negative",
    "net_charge_neutral",
}
_VAR = re.compile(r"[A-Z_][A-Za-z0-9_]*")


def bk_facts(mol: Chem.Mol) -> list[str]:
    """BK and computed facts of one molecule, as the dataset builds them (atom ids ``am_<idx+1>``)."""
    df = pd.DataFrame([{"mol": mol}], index=[MOL_ID])
    lines, _ = build_background_chemlog(df)
    return lines + build_computed_facts(df)


def atom_index(term: str) -> int | None:
    """0-based RDKit index of a BK atom id, or None for other terms."""
    m = re.fullmatch(rf"a{MOL_ID}_(\d+)", term)
    return int(m.group(1)) - 1 if m else None


def is_var(t: str) -> bool:
    return bool(_VAR.fullmatch(t))


def is_atom_term(t: str) -> bool:
    return atom_index(t) is not None


@dataclass
class MolContext:
    mol: Chem.Mol
    defs: dict
    facts: list[str] = field(default_factory=list)
    model: dict = field(default_factory=dict)  # pred -> set of arg tuples
    model_text: str = ""
    dist: np.ndarray | None = None
    xprogram: str = ""  # aux programs as xclingo reads them
    traces: dict = field(default_factory=dict)

    @classmethod
    def build(cls, mol, defs, programs: list[str]):
        facts = bk_facts(mol)
        ctx = cls(mol, defs, facts)
        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], "\n".join(facts) + "\n" + "\n".join(programs))
        ctl.ground([("base", [])])
        atoms = []
        with ctl.solve(yield_=True) as h:
            for m in h:
                atoms = [str(s) for s in m.symbols(atoms=True)]
                break
        model: dict[str, set] = {}
        for s in atoms:
            pred, args = split_atom(s)
            model.setdefault(pred, set()).add(args)
        ctx.model = model
        ctx.model_text = "\n".join(f"{s}." for s in atoms)
        ctx.dist = Chem.GetDistanceMatrix(mol)
        ctx.xprogram = xclingo_program(programs)
        return ctx

    def holds(self, pred: str, args: tuple) -> bool:
        return tuple(args) in self.model.get(pred, set())

    def atom_preds(self, idx: int) -> set[str]:
        """Unary model predicates true of atom ``idx``."""
        a = f"a{MOL_ID}_{idx + 1}"
        return {p for p, rows in self.model.items() if (a,) in rows}

    def bond_pred(self, i: int, j: int) -> str | None:
        a, b = f"a{MOL_ID}_{i + 1}", f"a{MOL_ID}_{j + 1}"
        for p in ("bSINGLE", "bDOUBLE", "bTRIPLE", "bAROMATIC"):
            if (a, b) in self.model.get(p, set()):
                return p
        return None


def split_atom(s: str) -> tuple[str, tuple]:
    sym = clingo.parse_term(s)
    if sym.type == clingo.SymbolType.Function:
        return sym.name, tuple(str(a) for a in sym.arguments)
    return str(sym), ()


def positive_vars(clause: Clause) -> list[str]:
    return ordered_vars(
        [clause.head] + [b for b in clause.body if isinstance(b, Lit) and not b.neg]
    )


def enumerate_bindings(
    clause: Clause, ctx: MolContext, limit: int = 20000, timeout: float = 20.0
) -> list[dict]:
    """All bindings of the clause's positively bound variables over the model (capped)."""
    vars_ = positive_vars(clause)
    if not vars_:
        vars_ = []
    head = Lit("_wit", tuple(vars_)) if vars_ else Lit("_wit", ())
    asp = to_asp([Clause(head, clause.body)])
    ctl = clingo.Control(["--warn=none"])
    show = f"#show _wit/{len(vars_)}."
    ctl.add("base", [], ctx.model_text + "\n" + asp + "\n" + show)
    ctl.ground([("base", [])])
    out = []
    with ctl.solve(yield_=True, async_=True) as h:
        h.resume()
        if not h.wait(timeout):
            h.cancel()
            return out
        m = h.model()
        if m is None:
            return out
        for s in m.symbols(shown=True):
            out.append(dict(zip(vars_, (str(a) for a in s.arguments))))
            if len(out) >= limit:
                break
    return out


def binding_score(b: dict, ctx: MolContext):
    """Lower is better: prefer injective, then compact (small pairwise graph distance), then low indices."""
    vals = [v for v in b.values() if is_atom_term(v)]
    idx = sorted({atom_index(v) for v in vals})
    collapse = len(vals) - len(idx)
    spread = 0.0
    for i, j in itertools.combinations(idx, 2):
        d = ctx.dist[i, j]
        spread += d if d < 1e6 else 50
    return (collapse, spread, idx)


def best_binding(
    bindings: list[dict], ctx: MolContext, prefer_injective: bool = True
) -> dict | None:
    if not bindings:
        return None
    key = (
        (lambda b: binding_score(b, ctx))
        if prefer_injective
        else (lambda b: binding_score(b, ctx)[1:])
    )
    return min(bindings, key=key)


def minimal_binding(bindings: list[dict], ctx: MolContext) -> dict | None:
    """Binding with the fewest distinct atoms."""
    if not bindings:
        return None
    return min(
        bindings,
        key=lambda b: (
            len({v for v in b.values() if is_atom_term(v)}),
            binding_score(b, ctx)[1:],
        ),
    )


# ── proof traces for derived literals (opaque aux predicates, aggregate conditions) ──


@dataclass
class Trace:
    atoms: set = field(default_factory=set)
    bonds: set = field(default_factory=set)  # frozenset({i, j})
    steps: int = 0


class _NameAnonymous(Transformer):
    """Replaces ``_`` outside negated literals by fresh variables; xclingo cannot ground anonymous ones."""

    def __init__(self):
        self.k = 0

    def visit_Literal(self, lit):
        return (
            lit if lit.sign != Sign.NoSign else lit.update(**self.visit_children(lit))
        )

    def visit_Variable(self, var):
        if var.name != "_":
            return var
        self.k += 1
        return var.update(name=f"Anon_{self.k}")


def xclingo_program(programs: list[str]) -> str:
    """``programs`` without comments and anonymous variables, ready for xclingo."""
    rename, out = _NameAnonymous(), []
    for src in programs:
        parse_string(
            src,
            lambda st: out.append(str(rename(st)))
            if st.ast_type != ASTType.Program
            else None,
        )
    return "\n".join(out)


def trace_literal(pred: str, args: tuple, ctx: MolContext) -> Trace:
    """Atoms and bonds in xclingo's explanation of the derived ground atom ``pred(args)``."""
    from xclingo import XclingoControl

    key = (pred, tuple(args))
    if key in ctx.traces:
        return ctx.traces[key]
    out = Trace()
    atom = clingo.parse_term(f"{pred}({','.join(args)})" if args else pred)
    xctl = XclingoControl(["--warn=none"], n_explanations="1")
    xctl.add("base", [], "\n".join(ctx.facts) + "\n" + ctx.xprogram)
    xctl.extend_explainer(
        "base", [], "_xclingo_label(Atom, Atom) :- _xclingo_f_atom(Atom)."
    )
    xctl.add_show_trace(atom)
    xctl.ground([("base", [])])
    for model in xctl.solve():
        for graph in model.explain_model():
            expl = graph.index.get(atom)
            if expl is None:
                continue
            items = []
            for node, _ in expl.preorder_iterator():
                pred_i, args_i = split_atom(str(node.atom))
                items.append(Lit(pred_i, args_i))
                out.steps += 1
            collect_atoms(items, out)
            break
        break
    ctx.traces[key] = out
    return out


def collect_atoms(items, out: Trace):
    for g in items:
        if not isinstance(g, Lit) or g.neg:
            continue
        idx = [atom_index(a) for a in g.args if is_atom_term(a)]
        out.atoms.update(idx)
        if g.pred in BOND_PREDS and len(idx) == 2 and idx[0] != idx[1]:
            out.bonds.add(frozenset(idx))


# ── aggregates ──


def aggregate_tuples(agg: Agg, binding: dict, ctx: MolContext) -> list[list[tuple]]:
    """Per aggregate element, the tuples it counts (outer variables substituted from ``binding``)."""
    res = []
    for e in agg.elems:
        terms = [substitute(Lit("_x", e.terms), binding)][0].args
        cond = [substitute(c, binding) for c in e.cond]
        vars_ = ordered_vars([Lit("_x", terms)] + list(cond))
        tv = [t for t in terms]
        head = Lit("_agg", tuple(tv))
        asp = to_asp([Clause(head, cond)])
        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], ctx.model_text + "\n" + asp + f"\n#show _agg/{len(tv)}.")
        try:
            ctl.ground([("base", [])])
        except RuntimeError:
            res.append([])
            continue
        rows = []
        with ctl.solve(yield_=True) as h:
            for m in h:
                rows = [
                    tuple(str(a) for a in s.arguments) for s in m.symbols(shown=True)
                ]
                break
        res.append(sorted(set(rows)))
        del vars_
    return res


def aggregate_value(agg: Agg, rows: list[list[tuple]]) -> float | None:
    allrows = sorted(set(itertools.chain.from_iterable(rows)))
    if agg.func == "count":
        return len(allrows)
    nums = []
    for r in allrows:
        try:
            nums.append(float(r[0]))
        except (ValueError, IndexError):
            return None
    if not nums:
        return 0 if agg.func.startswith("sum") else None
    return {"sum": sum, "sum+": sum, "min": min, "max": max}[agg.func](nums)


def neg_group_holds(ng: NegGroup, binding: dict, ctx: MolContext) -> bool:
    """True iff no alternative of the NegGroup holds under ``binding``."""
    for alt in ng.alternatives:
        body = [substitute(x, binding) for x in alt]
        if enumerate_bindings(Clause(Lit("_t", ()), body), ctx, limit=1, timeout=5):
            return False
    return True


def partial_neg_match(ng: NegGroup, binding: dict, ctx: MolContext):
    """For each alternative, the atoms matching its positive part (where the absence check looked)."""
    out = []
    for alt in ng.alternatives:
        body = [substitute(x, binding) for x in alt]
        pos = [x for x in body if isinstance(x, Lit) and not x.neg]
        bs = (
            enumerate_bindings(Clause(Lit("_t", ()), pos), ctx, limit=200, timeout=5)
            if pos
            else []
        )
        b = best_binding(bs, ctx, prefer_injective=True) if bs else None
        atoms = set()
        if b is not None:
            for x in pos:
                g = substitute(x, b)
                atoms.update(atom_index(a) for a in g.args if is_atom_term(a))
        out.append(atoms)
    return out
