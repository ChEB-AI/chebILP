"""Local explanation of one molecule: where the rule matches it, or the closest match when it does not.

The result holds what the picture needs (colours per atom and bond, short notes, legend) and the facts
the explanation sentence is built from.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

from rdkit import Chem

from chebILP.explainability import phrases
from chebILP.explainability.molecule import (
    BOND_PREDS,
    MOL_ID,
    MOL_PREDS,
    MolContext,
    aggregate_tuples,
    aggregate_value,
    atom_index,
    best_binding,
    enumerate_bindings,
    is_atom_term,
    is_var,
    minimal_binding,
    trace_literal,
)
from chebILP.explainability.near_miss import NearMiss, family, near_miss, rank_key
from chebILP.explainability.rule_ir import (
    Agg,
    Clause,
    Cmp,
    Lit,
    NegGroup,
    definitions,
    ordered_vars,
    parse_program,
    substitute,
)
from chebILP.explainability.style import GROUP_COLOURS
from chebILP.explainability.tagged import TClause, unfold_tagged

LETTERS = string.ascii_lowercase
_VISIBLE_NOTE = re.compile(
    r"\d+-ring|ring|no ring|not in \d+-ring|\d+H|≥\d+H|<\d+H|no H"
)


@dataclass
class RuleContext:
    """A rule with its aux definitions, unfolded once for all molecules."""

    clauses: list[Clause]
    defs: dict
    programs: list[str]
    aux_desc: dict
    tclauses: list[TClause]

    @classmethod
    def build(cls, rule: str, aux_programs: list[dict]):
        """``aux_programs``: ``{"name", "description", "source"}`` for every aux predicate the rule needs."""
        clauses = parse_program(rule)
        defs = definitions(aux_programs) if aux_programs else {}
        aux_desc = {
            p["name"]: p["description"] for p in aux_programs if p.get("description")
        }
        tclauses, _ = unfold_tagged(clauses, defs)
        return cls(
            clauses, defs, [p["source"] for p in aux_programs], aux_desc, tclauses
        )


@dataclass
class Group:
    key: str  # "rule" or the aux predicate written in the rule
    colour: str = "match"
    items: list = field(default_factory=list)
    mol_items: list = field(default_factory=list)
    atom_lits: list = field(default_factory=list)
    cmps: list = field(default_factory=list)
    aggs: list = field(default_factory=list)  # [agg, rows, value, holds]
    negs: list = field(default_factory=list)
    derived: list = field(
        default_factory=list
    )  # aux literals kept folded (recursive definitions)
    nodes: list = field(default_factory=list)  # atom variables of the pattern
    edges: dict = field(default_factory=dict)  # frozenset(vars) -> bond predicate
    anchor: str | None = None
    atoms: set = field(default_factory=set)
    bonds: set = field(default_factory=set)
    checked: dict = field(
        default_factory=dict
    )  # atom idx -> note, atoms an absence condition inspected


@dataclass
class Local:
    fires: bool
    mol: Chem.Mol
    ctx: MolContext
    atom_cols: dict = field(default_factory=dict)  # idx -> [colour keys]
    bond_cols: dict = field(default_factory=dict)  # frozenset({i, j}) -> [colour keys]
    notes: dict = field(default_factory=dict)  # idx -> note text
    legend: list = field(default_factory=list)  # (colour key or None, text)
    # facts for the sentence
    matched: list = field(default_factory=list)  # formula strings of the matched groups
    counts: list = field(default_factory=list)  # (value, noun)
    mol_facts: list = field(default_factory=list)  # phrases completing "it ..."
    missing: list = field(default_factory=list)  # why the closest match fails
    elsewhere: list = field(default_factory=list)


# ── groups of a matching clause ────────────────────────────────────────────


def build_groups(tc: TClause, rc: RuleContext) -> list[Group]:
    molvar = tc.mol_var
    groups: dict[str, Group] = {}
    for item, tag in tc.body:
        groups.setdefault(
            tag[0] if tag else "rule", Group(tag[0] if tag else "rule")
        ).items.append(item)
    clause_pos_vars = set(
        ordered_vars(
            [tc.head] + [b for b, _ in tc.body if isinstance(b, Lit) and not b.neg]
        )
    )
    for g in groups.values():
        for it in g.items:
            if isinstance(it, Lit) and it.pred == "has_atom" and not it.neg:
                continue
            if isinstance(it, Agg):
                g.aggs.append([it, None, None, None])
            elif isinstance(it, NegGroup):
                outer = (
                    set(ordered_vars([c for alt in it.alternatives for c in alt]))
                    & clause_pos_vars
                )
                (g.mol_items if outer <= {molvar} else g.negs).append(it)
            elif isinstance(it, Cmp):
                g.cmps.append(it)
            elif isinstance(it, Lit) and (
                it.pred in MOL_PREDS or set(it.args) <= {molvar}
            ):
                g.mol_items.append(it)
            elif isinstance(it, Lit) and it.pred in rc.defs and not it.neg:
                g.derived.append(it)
            elif isinstance(it, Lit):
                g.atom_lits.append(it)
        for lit in g.atom_lits:
            for a in lit.args:
                if is_var(a) and a != molvar and a not in g.nodes:
                    g.nodes.append(a)
        for lit in g.atom_lits:
            if lit.pred in BOND_PREDS and not lit.neg and len(lit.args) == 2:
                k = frozenset(lit.args)
                if len(k) == 2 and (k not in g.edges or g.edges[k] == "has_bond_to"):
                    g.edges[k] = lit.pred
    src = rc.clauses[tc.source]
    for g in groups.values():
        if g.key != "rule":
            lit = next(
                (b for b in src.body if isinstance(b, Lit) and b.pred == g.key), None
            )
            if lit is not None:
                g.anchor = next((a for a in lit.args if a in g.nodes), None)
    return sorted(groups.values(), key=lambda g: g.key != "rule")


def fill_groups(
    groups: list[Group], tc: TClause, b: dict, ctx: MolContext, rc: RuleContext
):
    molvar = tc.mol_var
    for g in groups:
        for lit in g.atom_lits:
            if lit.neg:
                continue
            idx = [atom_index(a) for a in substitute(lit, b).args if is_atom_term(a)]
            g.atoms.update(idx)
            if lit.pred in BOND_PREDS and len(set(idx)) == 2:
                g.bonds.add(frozenset(idx))
        for lit in g.derived:
            gl = substitute(lit, b)
            t = trace_literal(gl.pred, gl.args, ctx)
            g.atoms |= t.atoms
            g.bonds |= t.bonds
            g.atoms.update(atom_index(a) for a in gl.args if is_atom_term(a))
        for rec in g.aggs:
            agg = rec[0]
            rows = aggregate_tuples(agg, b, ctx)
            val = aggregate_value(agg, rows)
            rec[1], rec[2], rec[3] = rows, val, phrases.guard_holds(agg, val)
            for e, rs in zip(agg.elems, rows):
                derived = [
                    c
                    for c in e.cond
                    if isinstance(c, Lit) and not c.neg and c.pred in rc.defs
                ]
                for r in rs:
                    idx = [atom_index(a) for a in r if is_atom_term(a)]
                    g.atoms.update(idx)
                    if len(idx) == 2 and ctx.bond_pred(*idx):
                        g.bonds.add(frozenset(idx))
                    sub = dict(zip(e.terms, r))
                    sub[molvar] = b.get(molvar, MOL_ID)
                    for d in derived:
                        gd = substitute(d, sub)
                        if all(not is_var(a) for a in gd.args):
                            t = trace_literal(gd.pred, gd.args, ctx)
                            g.atoms |= t.atoms
                            g.bonds |= t.bonds
        for ng in g.negs:
            g.checked.update(neg_checked_atoms(ng, b, ctx))
        g.atoms.discard(None)


def neg_checked_atoms(ng: NegGroup, b: dict, ctx: MolContext) -> dict:
    """Bonded neighbours of the bound atoms that an absence condition inspected (its unary conditions hold)."""
    out = {}
    for alt in ng.alternatives:
        alt = [substitute(x, b) for x in alt]
        for lit in alt:
            if not (
                isinstance(lit, Lit)
                and lit.pred in BOND_PREDS
                and not lit.neg
                and len(lit.args) == 2
            ):
                continue
            ground = [a for a in lit.args if is_atom_term(a)]
            free = [a for a in lit.args if is_var(a)]
            if len(ground) != 1 or len(free) != 1:
                continue
            x = free[0]
            unary = [
                u for u in alt if isinstance(u, Lit) and not u.neg and u.args == (x,)
            ]
            if not unary:
                continue
            for bb in enumerate_bindings(
                Clause(Lit("_c", ()), [lit] + unary), ctx, limit=10, timeout=5
            ):
                out[atom_index(bb[x])] = ", ".join(
                    phrases.unary_phrase(u.pred, False)[1] for u in unary
                )
    return out


# ── tokens and strings ─────────────────────────────────────────────────────


def atom_token(ctx: MolContext, idx: int, show_h: bool) -> str:
    a = ctx.mol.GetAtomWithIdx(idx)
    s = a.GetSymbol()
    if show_h:
        n = next(
            (
                phrases.exact_h(p)
                for p in ctx.atom_preds(idx)
                if phrases.exact_h(p) is not None
            ),
            None,
        )
        if n:
            s += phrases.h_suffix(n)
    return s + phrases.charge_suffix(a.GetFormalCharge())


def var_token(g: Group, v: str) -> str:
    el, h, q = None, None, None
    for lit in g.atom_lits:
        if lit.args != (v,) or lit.neg:
            continue
        if phrases.element_of(lit.pred):
            el = lit.pred
        elif phrases.exact_h(lit.pred):
            h = phrases.exact_h(lit.pred)
        elif phrases.charge_of(lit.pred) is not None:
            q = phrases.charge_of(lit.pred)
    return (
        (phrases.sym(el) if el else "*")
        + (phrases.h_suffix(h) if h else "")
        + (phrases.charge_suffix(q) if q else "")
    )


def var_extras(g: Group, v: str, aux_desc) -> list[tuple[str, str]]:
    """(phrase, short note) for the conditions on ``v`` that its formula token does not show."""
    out = []
    for lit in g.atom_lits:
        if lit.args != (v,):
            continue
        if not lit.neg and lit.pred == "h":
            if ("is a hydrogen atom drawn in the structure", "explicit H") not in out:
                out.append(("is a hydrogen atom drawn in the structure", "explicit H"))
            continue
        if not lit.neg and (
            phrases.element_of(lit.pred)
            or (phrases.exact_h(lit.pred) or 0) > 0
            or phrases.charge_of(lit.pred) is not None
        ):
            continue
        ph = phrases.unary_phrase(lit.pred, lit.neg, aux_desc)
        if ph not in out:
            out.append(ph)
    for lit in g.derived:
        ph = phrases.unary_phrase(lit.pred, False, aux_desc)
        if lit.args == (v,) and ph not in out:
            out.append(ph)
    return out


def requires_h(g: Group, v: str) -> bool:
    return any(
        lit.args == (v,)
        and not lit.neg
        and (
            phrases.exact_h(lit.pred) is not None or lit.pred.startswith("has_at_least")
        )
        for lit in g.atom_lits
    )


def instance_graph(g: Group, b: dict, ctx: MolContext):
    nodes = []
    for v in g.nodes:
        i = atom_index(b[v])
        if i is not None and i not in nodes:
            nodes.append(i)
    edges = {}
    for e in g.edges:
        u, v = tuple(e)
        iu, iv = atom_index(b[u]), atom_index(b[v])
        if iu is None or iv is None or iu == iv:
            continue
        edges[frozenset((iu, iv))] = phrases.BOND_SYM[
            ctx.bond_pred(iu, iv) or "has_bond_to"
        ]
    return nodes, edges


def var_starts(g: Group):
    edges = {e: phrases.BOND_SYM[p] for e, p in g.edges.items()}
    return phrases.chain_start(
        g.nodes, edges, prefer=lambda v: not var_token(g, v).startswith("*")
    )


def atoms_formula(atoms: set, ctx: MolContext) -> str:
    """Formula string of ``atoms`` with the molecule's bonds between them."""
    nodes = sorted(atoms)
    edges = {}
    for i in nodes:
        for nb in ctx.mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if j in atoms and i < j:
                edges[frozenset((i, j))] = phrases.BOND_SYM[
                    ctx.bond_pred(i, j) or "has_bond_to"
                ]
    starts = phrases.chain_start(
        nodes, edges, prefer=lambda i: ctx.mol.GetAtomWithIdx(i).GetSymbol() != "C"
    )
    s, _ = phrases.render_graph(
        nodes, edges, lambda i: atom_token(ctx, i, False), starts=starts
    )
    return s.replace("` and `", " and ")


def group_formula(g: Group, b: dict, ctx: MolContext) -> str:
    if not g.nodes:
        return atoms_formula(g.atoms, ctx)
    nodes, edges = instance_graph(g, b, ctx)
    show_h = {atom_index(b[v]) for v in g.nodes if requires_h(g, v)}
    extra = g.atoms - set(nodes)  # atoms of folded aux literals
    if extra:
        return atoms_formula(g.atoms, ctx)
    s, _ = phrases.render_graph(
        nodes,
        edges,
        lambda i: atom_token(ctx, i, i in show_h),
        starts=[atom_index(b[v]) for v in var_starts(g)],
    )
    return s.replace("` and `", " and ")


def merge_phrases(ps: list[str]) -> str:
    """'is in a 5-ring', 'is in a 6-ring' -> 'is in a 5-ring and in a 6-ring'."""
    out = []
    for p in ps:
        if out and p.startswith("is ") and out[-1].startswith("is "):
            out[-1] = out[-1] + " and " + p[3:]
        else:
            out.append(p)
    return phrases.join_and(out)


def neg_alternatives(ng: NegGroup, b: dict, name_of_atom, molvar: str):
    """Formula strings of an absence condition's alternatives, anchored at the bound atoms."""
    shared = [
        v
        for v in ordered_vars([c for alt in ng.alternatives for c in alt])
        if v in b and is_atom_term(b[v])
    ]
    name_of = {v: name_of_atom(atom_index(b[v])) for v in shared}
    alts, wheres = [], []
    letters = iter("XYZUVW")
    for alt in ng.alternatives:
        sub = Group("neg")
        sub.atom_lits = [x for x in alt if isinstance(x, Lit) and x.pred != "has_atom"]
        nodes = []
        for lit in sub.atom_lits:
            for a in lit.args:
                if is_var(a) and a not in nodes and a != molvar:
                    nodes.append(a)
        edges = {
            frozenset(lit.args): phrases.BOND_SYM[lit.pred]
            for lit in sub.atom_lits
            if lit.pred in BOND_PREDS and not lit.neg and len(set(lit.args)) == 2
        }
        local_names = {}
        for v in nodes:
            if v in name_of:
                continue
            tok = var_token(sub, v)
            conds = [
                phrases.unary_phrase(lit.pred, lit.neg)[0]
                for lit in sub.atom_lits
                if lit.args == (v,)
                and not (
                    not lit.neg
                    and (phrases.element_of(lit.pred) or phrases.exact_h(lit.pred))
                )
            ]
            if conds:
                nm = next(letters, "X")
                local_names[v] = nm if tok == "*" else f"{tok}({nm})"
                wheres.append(f"{nm} {merge_phrases(conds)}")
            else:
                local_names[v] = tok
        roots = [v for v in nodes if v in name_of]
        s, _ = phrases.render_graph(
            nodes,
            edges,
            lambda v: name_of.get(v) or local_names.get(v, "*"),
            starts=roots,
        )
        alts.append(s.replace("` and `", " and "))
    return alts, wheres


def _mol_fact(pred: str, neg: bool, net: int) -> str:
    """'has a negative net charge (−1)'-style phrase for a molecule-level condition that holds."""
    q = f"{net:+d}".replace("-", "−")
    if pred.startswith("net_charge"):
        return (
            "has no net charge"
            if net == 0
            else f"has a {'positive' if net > 0 else 'negative'} net charge ({q})"
        )
    aromatic = (pred == "aromatic") != neg
    return "has an aromatic atom" if aromatic else "has no aromatic atom"


def _plural_verb(phrase: str, n: int) -> str:
    if n == 1:
        return phrase
    verb, _, rest = phrase.partition(" ")
    m = re.fullmatch(r"an? (.*) atom", rest)
    if verb == "is" and m:
        return f"are {m.group(1)} atoms"
    return (
        {"is": "are", "has": "have", "carries": "carry", "does": "do"}.get(verb, verb)
        + " "
        + rest
    )


def _visible(pred: str) -> bool:
    """Whether the drawing already shows this atom property (element or charge)."""
    return (
        bool(phrases.element_of(pred))
        or phrases.charge_of(pred) is not None
        or pred in ("charge_p", "charge_n")
    )


def paint(cols: dict, key, colour: str):
    if colour not in cols.setdefault(key, []):
        cols[key].append(colour)


# ── the molecule matches ───────────────────────────────────────────────────


def explain_match(loc: Local, rc: RuleContext, per_flat: list):
    ctx = loc.ctx
    fired = [i for i, b in enumerate(per_flat) if b]

    def size(i):
        return sum(
            1
            for b, _ in rc.tclauses[i].body
            if not (isinstance(b, Lit) and b.pred == "has_atom")
        )

    k = max(fired, key=lambda i: (size(i), -i))
    tc, bindings = rc.tclauses[k], per_flat[k]
    b = best_binding(bindings, ctx)
    vals = [v for v in b.values() if is_atom_term(v)]
    if len(vals) != len(set(vals)):  # no injective match exists: show the smallest one
        b = minimal_binding(bindings, ctx)
    groups = build_groups(tc, rc)
    fill_groups(groups, tc, b, ctx, rc)
    coloured = [g for g in groups if g.atoms]
    for j, g in enumerate(coloured):
        g.colour = GROUP_COLOURS[j % len(GROUP_COLOURS)]
    net = Chem.GetFormalCharge(loc.mol)
    notes: dict[int, list[str]] = {}
    for g in groups:
        for i in g.atoms:
            paint(loc.atom_cols, i, g.colour)
        for bnd in g.bonds:
            paint(loc.bond_cols, bnd, g.colour)
        for v in g.nodes:
            i = atom_index(b[v])
            lst = notes.setdefault(i, [])
            if requires_h(g, v) and loc.mol.GetAtomWithIdx(i).GetSymbol() == "C":
                for lit in g.atom_lits:
                    h = phrases.exact_h(lit.pred)
                    if lit.args == (v,) and h and f"{h}H" not in lst:
                        lst.insert(0, f"{h}H")
            for _, short in var_extras(g, v, rc.aux_desc):
                if short not in lst:
                    lst.append(short)
        for i, short in g.checked.items():
            if not loc.atom_cols.get(i):
                loc.atom_cols[i] = ["context"]
                notes.setdefault(i, []).append(short)
        if g.aggs:
            for agg, _rows, val, _ok in g.aggs:
                noun = " or ".join(
                    phrases.counted_noun(e, rc.aux_desc) for e in agg.elems
                )
                v = 0 if val is None else int(val) if float(val).is_integer() else val
                loc.counts.append((v, noun))
        elif g.atoms:
            loc.matched.append(group_formula(g, b, ctx))
        for it in g.mol_items:
            if isinstance(it, Lit):
                if it.pred in MOL_PREDS:
                    loc.mol_facts.append(_mol_fact(it.pred, it.neg, net))
                elif rc.aux_desc.get(it.pred):
                    loc.legend.append(
                        (None, f"Also true: {phrases.as_clause(rc.aux_desc[it.pred])}")
                    )
            elif isinstance(it, NegGroup) and rc.aux_desc.get(g.key):
                loc.legend.append(
                    (None, f"Also true: {phrases.as_clause(rc.aux_desc[g.key])}")
                )
    if (
        sum(1 for ps in notes.values() if ps) > 3
    ):  # crowded: drop what the drawing already shows
        notes = {
            i: [p for p in ps if not _VISIBLE_NOTE.fullmatch(p)]
            for i, ps in notes.items()
        }
    loc.notes = {i: ", ".join(ps) for i, ps in notes.items() if ps}
    single = len(coloured) == 1
    group_legend = []
    for g in coloured:
        if g.aggs:
            what = "counted atoms and bonds"
        elif single:
            what = "the matched atoms"
        elif g.key == "rule":
            what = "further matched atoms"
        else:
            what = phrases.as_clause(rc.aux_desc.get(g.key, "")) or "matched atoms"
        group_legend.append((g.colour, what[0].upper() + what[1:]))
    loc.legend[:0] = group_legend
    if any(c == ["context"] for c in loc.atom_cols.values()):
        loc.legend.append(
            ("context", "Atoms checked for a group that must be absent; it is absent.")
        )
    if loc.notes:
        loc.legend.append(
            (
                None,
                "Boxes: further conditions on an atom (ring size, H count, steroid position, ...).",
            )
        )


# ── the molecule does not match ────────────────────────────────────────────


class Letters:
    """Letters for atoms referred to in the text, handed out on first use."""

    def __init__(self, ctx: MolContext):
        self.ctx, self.of = ctx, {}

    def __call__(self, i: int | None) -> str:
        if i is None:
            return "an atom"
        if i not in self.of and len(self.of) < len(LETTERS):
            self.of[i] = LETTERS[len(self.of)]
        return self.of.get(i, self.ctx.mol.GetAtomWithIdx(i).GetSymbol())


def wrong_value(ctx: MolContext, idx: int, pred: str, aux_desc: dict) -> str:
    """'is at steroid position 8, not 6'-style phrase for an atom that fails ``pred``."""
    preds = ctx.atom_preds(idx)
    fam = family(pred)
    atom = ctx.mol.GetAtomWithIdx(idx)
    el = phrases.element_of(pred)
    if el:
        return f"is {atom.GetSymbol()}, not {phrases.sym(el)}"
    if fam == "steroid":
        ks = sorted(int(p.split("_")[1]) for p in preds if family(p) == "steroid")
        want = pred.split("_")[1]
        if ks:
            return f"is at steroid position {', '.join(map(str, ks))}, not {want}"
        return f"is not part of the steroid core (position {want} is needed)"
    if fam == "ring" or pred == "in_ring":
        ks = sorted(
            int(p[7:]) for p in preds if p.startswith("in_ring") and p[7:].isdigit()
        )
        want = pred[7:]
        if not ks:
            return "is not in a ring" + (
                f" (a {want}-membered ring is needed)" if want else ""
            )
        return f"is in a ring of size {', '.join(map(str, ks))}, not {want}"
    if fam == "cip":
        cs = [p[-1] for p in preds if family(p) == "cip"]
        return (
            f"is ({cs[0]})-configured, not ({pred[-1]})"
            if cs
            else f"has no (R)/(S) label; ({pred[-1]}) is needed"
        )
    if fam == "hcount":
        n = next(
            (phrases.exact_h(p) for p in preds if phrases.exact_h(p) is not None), 0
        )
        want = phrases.exact_h(pred)
        if not want:
            return f"carries {phrases.num(n)} H but must carry none"
        return f"carries {phrases.num(n) if n else 'no'} H, not {phrases.num(want)}"
    if pred.startswith("has_at_least_"):
        n = next(
            (phrases.exact_h(p) for p in preds if phrases.exact_h(p) is not None), 0
        )
        k = int(pred.split("_")[3])
        return f"carries {phrases.num(n) if n else 'no'} H; at least {phrases.num(k)} {'is' if k == 1 else 'are'} needed"
    if fam == "charge" or pred in ("charge0", "charge_p", "charge_n"):
        q = f"{atom.GetFormalCharge():+d}".replace("-", "−").replace("+0", "0")
        want, _ = phrases.unary_phrase(pred)
        return f"has charge {q}, but {want.replace('has ', 'needs ')}"
    ph, _ = phrases.unary_phrase(pred, True, aux_desc)
    return ph


def missing_phrase(nm: NearMiss, i: int, ctx: MolContext, name, aux_desc: dict) -> str:
    """Why body item ``i`` fails in the closest match, completing a sentence."""
    tc, b = nm.tclause, nm.binding
    item, tag = tc.body[i]
    molvar = tc.mol_var
    outer_desc = aux_desc.get(tag[0]) if tag else None

    def who(v):
        return name(atom_index(b.get(v, "")))

    if isinstance(item, Lit) and (item.pred in MOL_PREDS or set(item.args) <= {molvar}):
        net = Chem.GetFormalCharge(ctx.mol)
        q = f"{net:+d}".replace("-", "−") if net else "0"
        if item.pred.startswith("net_charge"):
            want = {"neutral": "zero", "positive": "positive", "negative": "negative"}[
                item.pred.split("_")[-1]
            ]
            return f"its net charge is {q}, not {want}"
        if item.pred == "aliphatic":
            return "it has aromatic atoms"
        if item.pred == "aromatic":
            return "it has no aromatic atom"
        d = aux_desc.get(item.pred)
        return (
            f"it does not meet the condition “{phrases.as_clause(d)}”"
            if d
            else f"it lacks {phrases.humanise(item.pred)}"
        )
    if isinstance(item, Agg):
        rows = aggregate_tuples(item, {molvar: MOL_ID}, ctx)
        val = aggregate_value(item, rows)
        noun = " or ".join(phrases.counted_noun(e, aux_desc) for e in item.elems)
        v = (
            "an unknown number of"
            if val is None
            else int(val)
            if float(val).is_integer()
            else val
        )
        guard = phrases.guard_phrase(item)
        return f"it has {v} {phrases.plural(noun, v)}, but {guard} {'is' if guard.endswith(' 1') else 'are'} needed"
    if isinstance(item, NegGroup):
        alts, wheres = neg_alternatives(item, b, name, molvar)
        where = f", where {'; '.join(wheres)}" if wheres else ""
        d = (aux_desc.get(item.source_pred) if item.source_pred else None) or outer_desc
        if d:
            return f"it does not meet the condition “{phrases.as_clause(d)}”"
        return f"it contains {' or '.join(alts)}{where}, which must be absent"
    if isinstance(item, Cmp):
        return (
            f"{who(item.left)} and {who(item.right)} would have to be different atoms"
        )
    if isinstance(item, Lit) and len(item.args) == 1:
        v = item.args[0]
        if v not in nm.anchored:
            el = phrases.element_of(item.pred)
            if el and not item.neg:
                return (
                    "there is no hydrogen atom drawn in the structure"
                    if el == "h"
                    else f"there is no {phrases.ELEMENT_NAME.get(el, phrases.sym(el))} atom where it is needed"
                )
            ph, _ = phrases.unary_phrase(item.pred, item.neg, aux_desc)
            return f"no atom {ph}"
        idx = atom_index(b[v])
        if item.neg:
            ph, _ = phrases.unary_phrase(item.pred, False, aux_desc)
            return f"{who(v)} {ph}, but must not be"
        return f"{who(v)} {wrong_value(ctx, idx, item.pred, aux_desc)}"
    if isinstance(item, Lit) and item.pred in BOND_PREDS and len(item.args) == 2:
        u, v = item.args
        word = {
            "bSINGLE": "single",
            "bDOUBLE": "double",
            "bTRIPLE": "triple",
            "bAROMATIC": "aromatic",
        }.get(item.pred)
        if u in nm.anchored and v in nm.anchored:
            iu, iv = atom_index(b[u]), atom_index(b[v])
            actual = ctx.bond_pred(iu, iv)
            if actual and word:
                return f"the {who(u)}–{who(v)} bond is {phrases.bond_word(actual)}, not {word}"
            return f"{who(u)} and {who(v)} are not bonded"
        anchored = u if u in nm.anchored else (v if v in nm.anchored else None)
        if anchored:
            other = v if anchored == u else u
            el = next(
                (
                    phrases.element_of(it.pred)
                    for it, _ in tc.body
                    if isinstance(it, Lit)
                    and not it.neg
                    and it.args == (other,)
                    and phrases.element_of(it.pred)
                ),
                None,
            )
            what = (
                f"{phrases.article(phrases.ELEMENT_NAME.get(el, el))} {phrases.ELEMENT_NAME.get(el, el)} atom"
                if el
                else "an atom"
            )
            kind = f"{word} bond" if word else "bond"
            return f"{who(anchored)} has no {kind} to {what} that fits the rest of the pattern"
        return (
            f"there is no {word} bond where it is needed"
            if word
            else "a bond of the pattern is missing"
        )
    if outer_desc:
        return f"it does not meet the condition “{phrases.as_clause(outer_desc)}”"
    return "one more condition fails"


def explain_miss(loc: Local, rc: RuleContext):
    ctx = loc.ctx
    nms = [near_miss(tc, ctx) for tc in rc.tclauses]
    if not nms:
        return
    nm = min(nms, key=rank_key)
    tc, b = nm.tclause, nm.binding
    name = Letters(ctx)
    loc.missing = [missing_phrase(nm, i, ctx, name, rc.aux_desc) for i in nm.unsat]
    notes: dict[int, list[str]] = {}
    agg_legend: list = []
    struct_cols: set = set()
    for i in nm.sat:
        item = tc.body[i][0]
        if isinstance(item, Lit) and not item.neg and item.pred != "has_atom":
            idx = [atom_index(b[a]) for a in item.args if a in nm.anchored]
            for j in idx:
                paint(loc.atom_cols, j, "match")
                struct_cols.add("match")
            if item.pred in BOND_PREDS and len(set(idx)) == 2:
                loc.bond_cols[frozenset(idx)] = ["match"]
            if len(item.args) == 1 and idx and not _visible(item.pred):
                short = phrases.unary_phrase(item.pred, False, rc.aux_desc)[1]
                if short not in notes.setdefault(idx[0], []):
                    notes[idx[0]].append(short)
    for i, (item, _) in enumerate(tc.body):
        if not isinstance(item, Agg):
            continue
        col = "match" if i in nm.sat else "absent"
        rows = aggregate_tuples(item, {tc.mol_var: MOL_ID}, ctx)
        val = aggregate_value(item, rows)
        v = "?" if val is None else int(val) if float(val).is_integer() else val
        noun = " or ".join(phrases.counted_noun(e, rc.aux_desc) for e in item.elems)
        agg_legend.append(
            (
                col,
                f"Counted: {v} {phrases.plural(noun, v)} ({phrases.guard_phrase(item)} needed).",
            )
        )
        for rs in rows:
            for r in rs:
                idx = [atom_index(a) for a in r if is_atom_term(a)]
                if len(idx) == 2 and ctx.bond_pred(*idx):
                    paint(loc.bond_cols, frozenset(idx), col)
                for j in idx:
                    paint(loc.atom_cols, j, col)
    for i in nm.unsat:
        item = tc.body[i][0]
        if not isinstance(item, Lit):
            continue
        idx = [atom_index(b[a]) for a in item.args if a in nm.anchored]
        for j in idx:
            loc.atom_cols[j] = ["absent"]
            struct_cols.add("absent")
        if len(item.args) == 1 and idx:
            short = phrases.unary_phrase(item.pred, item.neg, rc.aux_desc)[1]
            notes.setdefault(idx[0], []).append(f"needs {short}")
        if item.pred in BOND_PREDS and len(set(idx)) == 2 and ctx.bond_pred(*idx):
            loc.bond_cols[frozenset(idx)] = ["absent"]
    for i in nm.unsat:
        item = tc.body[i][0]
        if not (
            isinstance(item, Lit)
            and not item.neg
            and len(item.args) == 1
            and item.args[0] in nm.anchored
        ):
            continue
        bound = atom_index(b[item.args[0]])
        hits = sorted(
            {atom_index(r[0]) for r in ctx.model.get(item.pred, ())} - {None, bound}
        )
        if not hits or len(hits) > 6:
            continue
        short = phrases.unary_phrase(item.pred, False, rc.aux_desc)[1]
        for j in hits:
            if not loc.atom_cols.get(j):
                loc.atom_cols[j] = ["elsewhere"]
                if not _visible(item.pred):
                    notes.setdefault(j, []).append(short)
        n = len(hits)
        has = _plural_verb(phrases.unary_phrase(item.pred, False, rc.aux_desc)[0], n)
        loc.elsewhere.append(
            f"The yellow atom{'s' if n > 1 else ''} {has}, but the rest of the pattern does not fit "
            f"around {'them' if n > 1 else 'it'}."
        )
    for i in set(notes) | set(name.of):
        parts, lab = notes.get(i, []), name.of.get(i)
        text = (f"{lab}: " if lab and parts else (lab or "")) + ", ".join(parts)
        if text:
            loc.notes[i] = text
    if "match" in struct_cols:
        loc.legend.append(
            ("match", "Closest match: the part of the pattern this molecule has.")
        )
    if "absent" in struct_cols:
        loc.legend.append(("absent", "Where the pattern fails."))
    loc.legend += agg_legend
    if loc.elsewhere:
        loc.legend.append(("elsewhere", "Atoms elsewhere with the missing property."))
    if name.of:
        loc.legend.append((None, "Letters mark the atoms named in the text."))


def analyse(mol: Chem.Mol, rc: RuleContext) -> Local:
    ctx = MolContext.build(mol, rc.defs, rc.programs)
    per_flat = [enumerate_bindings(tc.clause(), ctx) for tc in rc.tclauses]
    loc = Local(any(per_flat), mol, ctx)
    if loc.fires:
        explain_match(loc, rc, per_flat)
    else:
        explain_miss(loc, rc)
    return loc
