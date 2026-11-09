"""What a global explanation figure shows: per variant of the rule the pattern drawing, chips for
molecule-level conditions and counts, sub-panels for "one of" / "none of" groups, and notes."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from chebILP.explainability.display_logic import ELEMENTS, display_unfold, item_vars
from chebILP.explainability.pattern import (
    BOND_PREDS,
    Pattern,
    build_pattern,
    contract_chains,
    core,
    distinct_pairs,
    strip_anchor_restatements,
    tighten_bonds,
)
from chebILP.explainability.pattern_draw import (
    GENERIC,
    DrawStyle,
    Drawn,
    draw_pattern,
    element_symbol,
    gonane_neighbours,
)
from chebILP.explainability.rule_ir import Agg, Clause, Cmp, Lit, NegGroup
from chebILP.explainability.style import C

MAIN = DrawStyle()
SUB_NEG = DrawStyle(
    bond_px=54,
    font_px=15,
    note_px=12,
    bg=C["neg_bg"],
    halo_local="neg",
    scaffold_scale=0.58,
)
SUB_OR = DrawStyle(
    bond_px=54, font_px=15, note_px=12, bg=C["or_bg"], scaffold_scale=0.58
)
MINI = DrawStyle(bond_px=54, font_px=15, note_px=12, bg=C["count_bg"])

MOL_TEXT = {
    "aliphatic": "no aromatic atom anywhere in the molecule",
    "aromatic": "at least one aromatic atom in the molecule",
    "net_charge_positive": "positive net charge",
    "net_charge_negative": "negative net charge",
    "net_charge_neutral": "no net charge",
}
BOND_SYM = {"single": "–", "double": "=", "triple": "≡", "aromatic": ":", "any": "~"}


@dataclass
class Chip:
    text: str
    kind: str = "plain"  # plain | neg | count | aux


@dataclass
class Mini:
    text: str
    drawn: Drawn


@dataclass
class AltModel:
    drawn: Drawn | None = None
    text: str | None = None
    footnotes: list = field(default_factory=list)


@dataclass
class GroupModel:
    kind: str  # "or" | "neg"
    title: str
    alts: list
    description: str = ""


@dataclass
class ClauseModel:
    chips: list = field(default_factory=list)
    minis: list = field(default_factory=list)
    drawn: Drawn | None = None
    groups: list = field(default_factory=list)
    footnotes: list = field(default_factory=list)
    flags: set = field(default_factory=set)
    numbers: dict = field(default_factory=dict)  # variable -> badge number
    pattern: Pattern | None = None  # the drawn pattern before display-only contractions
    failed: bool = False


@dataclass
class FigureModel:
    clauses: list
    legend: list  # (colour key or None, text)


# ── helpers ────────────────────────────────────────────────────────────────


def _quantifier(agg: Agg) -> str | None:
    def word(op, n, side):
        if side == "l":  # n op #agg
            return {
                "=": f"exactly {n}",
                "<=": f"at least {n}",
                "<": f"more than {n}",
                ">=": f"at most {n}",
                ">": f"fewer than {n}",
                "!=": f"not exactly {n}",
            }[op]
        return {
            "=": f"exactly {n}",
            ">=": f"at least {n}",
            ">": f"more than {n}",
            "<=": f"at most {n}",
            "<": f"fewer than {n}",
            "!=": f"not exactly {n}",
        }[op]

    parts = []
    if agg.lguard:
        parts.append(word(agg.lguard[0], agg.lguard[1], "l"))
    if agg.rguard:
        parts.append(word(agg.rguard[0], agg.rguard[1], "r"))
    return " and ".join(parts) if parts else None


def _plural(n: str, noun: str) -> str:
    return noun if n.strip().endswith(" 1") and not n.startswith("more") else noun + "s"


def agg_model(agg: Agg, mol: str, elemsets: dict, desc: dict):
    """A chip for simple counts (``exactly 22 C atoms``), else a Mini with the counted unit drawn."""
    q = _quantifier(agg)
    if agg.func != "count" or len(agg.elems) != 1 or q is None:
        return Chip(
            "a condition on a sum or extreme value over parts of the molecule", "count"
        )
    e = agg.elems[0]
    cond = [
        c
        for c in e.cond
        if not (
            isinstance(c, Lit)
            and c.pred == "has_atom"
            and not c.neg
            and c.args[0] == mol
        )
    ]
    terms = list(e.terms)
    els = {
        c.args[0]: c.pred
        for c in cond
        if isinstance(c, Lit) and not c.neg and c.pred in ELEMENTS and len(c.args) == 1
    }
    if len(terms) == 1 and len(cond) == 1 and terms[0] in els:
        return Chip(
            f"{q} {els[terms[0]].capitalize()} {_plural(q, 'atom')} in the molecule",
            "count",
        )
    if len(terms) == 2 and len(cond) == 4 and set(terms) <= set(els):
        bonds = [
            c
            for c in cond
            if isinstance(c, Lit) and c.pred in BOND_PREDS and set(c.args) == set(terms)
        ]
        order = [
            c
            for c in cond
            if isinstance(c, Cmp)
            and {c.left, c.right} == set(terms)
            and c.op in ("<", ">", "!=")
        ]
        if len(bonds) == 1 and len(order) == 1:
            a, b = terms
            sym = BOND_SYM[BOND_PREDS[bonds[0].pred]]
            return Chip(
                f"{q} {els[a].capitalize()}{sym}{els[b].capitalize()} {_plural(q, 'bond')} in the molecule",
                "count",
            )
    try:
        pat = build_pattern(cond, mol, elemsets, desc, counted=set(terms))
        unit = "atoms" if len(terms) == 1 else "atom pairs"
        d = draw_pattern(pat, MINI)
        return Mini(
            f"{q} different {unit} like the highlighted one{'s' if len(terms) > 1 else ''}:",
            d,
        )
    except Exception:
        return Chip(f"{q} matching parts of the molecule", "count")


def every_atom_chip(ng: NegGroup, mol: str, elemsets: dict) -> Chip | None:
    """An absence of atoms outside an element set -> 'every atom is one of ...'."""
    if len(ng.alternatives) != 1:
        return None
    alt = [
        c
        for c in ng.alternatives[0]
        if not (isinstance(c, Lit) and c.pred == "has_atom" and c.args[0] == mol)
    ]
    if (
        len(alt) == 1
        and isinstance(alt[0], Lit)
        and alt[0].neg
        and len(alt[0].args) == 1
    ):
        p = alt[0].pred
        els = elemsets.get(p) or ({p} if p in ELEMENTS else None)
        if els:
            names = ", ".join(
                e.capitalize()
                for e in sorted(els, key=lambda e: (e not in "cnosph", e))
            )
            return Chip(f"every atom is one of: {names}", "neg")
    return None


def scaffold_refs(pat: Pattern, dist: set) -> dict:
    """Non-core atoms that may themselves be ring atoms of the steroid core: var -> positions."""
    st = {v: s.steroid for v, s in pat.atoms.items() if s.steroid is not None}
    if not st:
        return {}
    out = {}
    for x, s in pat.atoms.items():
        if x in st or s.not_in_ring or "c" in s.not_elements:
            continue
        els = s.eff_elements()
        if els is not None and "c" not in els:
            continue
        cands = set()
        for w in pat.nbrs(x):
            if w in st:
                cands |= set(gonane_neighbours(st[w]))
        cands |= {st[w] for w in st if frozenset((x, w)) not in dist}
        cands -= {st[w] for w in st if frozenset((x, w)) in dist}
        out[x] = sorted(cands) if cands else None
    return out


def _ref(v: str, pat: Pattern, numbers: dict) -> str:
    if v in numbers:
        return "{" + str(numbers[v]) + "}"
    sym, _ = element_symbol(pat.atoms[v])
    return sym


def _folded_text(full: Pattern, core_pat: Pattern) -> str:
    """Why atoms named in the rule are missing from the drawing."""
    n_full, n_core = len(full.atoms), len(core_pat.atoms)
    syms = {element_symbol(full.atoms[v])[0] for v in full.atoms}
    if len(syms) == 1 and n_core == 1:
        sym = syms.pop()
        what = "atoms" if sym == GENERIC else f"{sym} atoms"
        return f"One atom is enough: the rule names {n_full} {what}, but they may all be the same atom."
    extra = n_full - n_core
    return (
        f"Simplified: the rule names {extra} more atom{'s' if extra > 1 else ''}, which may be atoms "
        f"already drawn; they add no condition."
    )


def _pairs_text(pairs, numbers) -> str:
    items = [f"{{{numbers[u]}}}–{{{numbers[v]}}}" for u, v in pairs]
    if len(items) > 8:
        items = items[:8] + [f"and {len(pairs) - 8} more"]
    return ", ".join(items)


class Numberer:
    def __init__(self):
        self.n = 0
        self.of = {}

    def get(self, v):
        if v not in self.of:
            self.n += 1
            self.of[v] = self.n
        return self.of[v]


# ── clause model ───────────────────────────────────────────────────────────


def clause_model(clause: Clause, elemsets, desc) -> ClauseModel:
    cm = ClauseModel()
    mol = clause.head.args[0] if clause.head.args else None
    full = build_pattern(clause.body, mol, elemsets, desc)
    pat, folded = core(full)
    cm.pattern = pat
    if folded:
        cm.footnotes.append(_folded_text(full, pat))
    dist = distinct_pairs(pat)
    num = Numberer()
    comps = pat.components()
    comp_of = {v: i for i, c in enumerate(comps) for v in c}
    st_refs = scaffold_refs(pat, dist)
    on_core = {v for v, s in pat.atoms.items() if s.steroid is not None}
    within, cross = [], []
    for u, v in itertools.combinations(list(pat.atoms), 2):
        if frozenset((u, v)) in dist:
            continue
        if (u in on_core and v in st_refs) or (v in on_core and u in st_refs):
            continue
        (within if comp_of[u] == comp_of[v] else cross).append((u, v))

    for u, v in within:
        num.get(u), num.get(v)
    for x in st_refs:
        sym = element_symbol(pat.atoms[x])[0]
        if sum(1 for y in pat.atoms if element_symbol(pat.atoms[y])[0] == sym) > 1:
            num.get(x)

    # sub-panels first, so anchors get numbers before drawing
    groups = []
    for g in pat.ors + pat.negs:
        chip = every_atom_chip(g, mol, elemsets) if isinstance(g, NegGroup) else None
        if chip:
            cm.chips.append(chip)
            continue
        groups.append(g)
    for g in groups:
        for v in item_vars(g):
            if v in pat.atoms and pat.atoms[v].steroid is None:
                num.get(v)
    numbers = dict(num.of)

    for pred, neg in pat.mol_conds:
        t = MOL_TEXT.get(pred, pred)
        cm.chips.append(
            Chip(
                ("not: " + t) if neg else t,
                "neg" if pred == "aliphatic" or neg else "plain",
            )
        )
    for agg in pat.aggs:
        m = agg_model(agg, mol, elemsets, desc)
        (cm.minis if isinstance(m, Mini) else cm.chips).append(m)
    for marker, pred, args, d, neg in pat.opaque:
        name = pred.replace("aux_", "").replace("_", " ")
        cm.chips.append(Chip(f"†{marker} {'not ' if neg else ''}{d or name}", "aux"))
    for o in pat.other:
        cm.chips.append(Chip(o, "plain"))

    disp = tighten_bonds(pat, dist)
    keep = set(numbers) | {v for g in groups for v in item_vars(g)} | set(pat.neg_notes)
    disp = contract_chains(disp, dist, keep)
    if disp.atoms:
        cm.drawn = draw_pattern(disp, MAIN, numbers)
        for flag, on in (
            ("generic", cm.drawn.generic_atom),
            ("any", cm.drawn.any_bond),
            ("scaffold", cm.drawn.scaffold),
            ("explicit_h", cm.drawn.explicit_h),
            ("aromatic", cm.drawn.aromatic),
        ):
            if on:
                cm.flags.add(flag)

        if len(disp.atoms) > 1:
            cm.flags.add("distinct")

    if within:
        cm.footnotes.append(
            f"May be one and the same atom: {_pairs_text(within, numbers)}."
        )
    if len(comps) > 1:
        if cross:
            cm.footnotes.append(
                "The separate parts can be anywhere in the molecule; they may share atoms."
            )
        else:
            cm.footnotes.append(
                "The separate parts are different atoms; they can be anywhere in the molecule, "
                "also bonded to each other."
            )
    for x, cands in st_refs.items():
        r = _ref(x, pat, numbers)
        if cands:
            cm.footnotes.append(
                f"{r} may itself be ring atom {' or '.join(f'C{q}' for q in cands)} of the steroid core."
            )
        else:
            cm.footnotes.append(f"{r} may itself be a ring atom of the steroid core.")
    if any(s.contradiction for s in pat.atoms.values()) or any(
        b.contradiction for b in pat.bonds.values()
    ):
        cm.footnotes.append(
            "No molecule can match this: its conditions contradict each other."
        )

    if any("{" in fn for fn in cm.footnotes):
        cm.flags.add("numbers")
    for g in groups:
        try:
            cm.groups.append(group_model(g, pat, num, mol, elemsets, desc))
        except Exception:
            kind = "neg" if isinstance(g, NegGroup) else "or"
            cm.groups.append(
                GroupModel(
                    kind,
                    "and none of:" if kind == "neg" else "and one of:",
                    [
                        AltModel(
                            text=desc.get(g.source_pred, "") or "(could not be drawn)"
                        )
                    ],
                )
            )
    cm.numbers = dict(num.of)
    return cm


def group_model(g, pat: Pattern, num: Numberer, mol, elemsets, desc) -> GroupModel:
    kind = "neg" if isinstance(g, NegGroup) else "or"
    anchors = {v: pat.atoms[v] for v in item_vars(g) if v in pat.atoms}
    alts = []
    for alt in g.alternatives:
        items = strip_anchor_restatements(alt, anchors, elemsets)
        sub = build_pattern(items, mol, elemsets, desc, anchors=anchors)
        sub, _ = core(sub)
        sdist = distinct_pairs(sub)
        foot = []
        local_pairs = [
            (u, v)
            for u, v in itertools.combinations(list(sub.atoms), 2)
            if frozenset((u, v)) not in sdist and not (u in anchors and v in anchors)
        ]
        st_refs = scaffold_refs(sub, sdist)
        local_pairs = [
            (u, v)
            for u, v in local_pairs
            if not (
                (u in st_refs and sub.atoms[v].steroid is not None)
                or (v in st_refs and sub.atoms[u].steroid is not None)
            )
        ]
        for u, v in local_pairs:
            num.get(u), num.get(v)
        nums = {v: num.of[v] for v in sub.atoms if v in num.of}
        disp = contract_chains(
            tighten_bonds(sub, sdist), sdist, set(nums) | set(anchors)
        )
        style = SUB_NEG if kind == "neg" else SUB_OR
        chips = [
            ("not: " if neg else "") + MOL_TEXT.get(pred, pred)
            for pred, neg in sub.mol_conds
        ]
        for agg in sub.aggs:
            chips.append(agg_model(agg, mol, elemsets, desc).text)
        chips += list(sub.other)
        chips += [
            d or pred.replace("aux_", "").replace("_", " ")
            for _, pred, _, d, _ in sub.opaque
        ]
        if sub.negs or sub.ors:
            chips.append("further nested conditions")
        local = [v for v in disp.atoms if v not in anchors]
        drawn = (
            draw_pattern(disp, style, nums)
            if local or disp.bonds or any(disp.neg_notes.values())
            else None
        )
        text = (
            "; ".join(chips)
            if chips
            else (None if drawn is not None else "(no further condition)")
        )
        if local_pairs:
            foot.append(
                f"May be one and the same atom: {_pairs_text(local_pairs, num.of)}."
            )
        for x, cands in st_refs.items():
            if cands:
                foot.append(
                    f"{_ref(x, sub, num.of)} may itself be ring atom "
                    f"{' or '.join(f'C{q}' for q in cands)} of the steroid core."
                )
        alts.append(AltModel(drawn, text, foot))
    title = (
        "and one of:"
        if kind == "or"
        else ("and none of:" if len(alts) > 1 else "and not:")
    )
    return GroupModel(kind, title, alts, desc.get(g.source_pred, ""))


# ── figure model ───────────────────────────────────────────────────────────


def figure_model(rule_clauses: list[Clause], defs: dict, desc: dict) -> FigureModel:
    """Panels for the rule ``rule_clauses`` with aux definitions ``defs`` and descriptions ``desc``."""
    try:
        res = display_unfold(rule_clauses, defs)
        clauses, elemsets = res.clauses, res.elemsets
    except Exception:
        clauses, elemsets = rule_clauses, {}
    cms = []
    for c in clauses:
        try:
            cms.append(clause_model(c, elemsets, desc))
        except Exception:
            cms.append(
                ClauseModel(
                    chips=[Chip("This variant could not be drawn.", "plain")],
                    failed=True,
                )
            )
    flags = set().union(*(c.flags for c in cms)) if cms else set()
    legend = []
    if "distinct" in flags:
        legend.append(
            (
                None,
                "Atoms drawn separately are different atoms, unless a note says they may be the same.",
            )
        )
    if "generic" in flags:
        legend.append((None, f"{GENERIC}: an atom of any element."))
    if "any" in flags:
        legend.append((None, "Dashed line: a bond of any type."))
    if "aromatic" in flags:
        legend.append((None, "Line with a dotted line beside it: aromatic bond."))
    if "scaffold" in flags:
        legend.append(
            (
                None,
                "Grey: the steroid core (positions 1–17); its bond types are not fixed.",
            )
        )
    if "explicit_h" in flags:
        legend.append(
            (
                None,
                "H: a hydrogen atom drawn explicitly in the structure; implicit hydrogens do not count.",
            )
        )
    if "numbers" in flags:
        legend.append((None, "Numbered atoms are referred to in the notes."))
    if any(g.kind == "neg" for c in cms for g in c.groups):
        legend.append(
            (
                "absent",
                "Red panel: must not be present; numbered atoms are the same as in the main drawing.",
            )
        )
    if any(g.kind == "or" for c in cms for g in c.groups):
        legend.append(("alt", "Blue panel: at least one of these must be present."))
    return FigureModel(cms, legend)
