"""Pattern graphs built from display clauses: atom specs, bonds, distinctness and core."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field, replace

import networkx as nx

from chebILP.explainability.rule_ir import Agg, Cmp, Lit, NegGroup
from chebILP.explainability.display_logic import (
    ELEMENTS,
    MOL_PREDS,
    OrGroup,
    is_var,
    item_vars,
)
from chebILP.preprocessing.chebi.fg_matching import is_fg_seed_name

BOND_PREDS = {
    "bSINGLE": "single",
    "bDOUBLE": "double",
    "bTRIPLE": "triple",
    "bAROMATIC": "aromatic",
    "has_bond_to": "any",
}
STEREO_PREDS = {
    "bSTEREOE": "E",
    "bSTEREOZ": "Z",
    "bSTEREOANY": "stereo",
    "bSTEREOCIS": "cis",
    "bSTEREOTRANS": "trans",
}
ORDER = {"single": 1.0, "double": 2.0, "triple": 3.0, "aromatic": 1.5, "any": 1.0}
# lowest valence a bond can contribute (an aromatic bond counts 1 at a pyrrole-type N)
VALENCE_LB = {"single": 1.0, "double": 2.0, "triple": 3.0, "aromatic": 1.0, "any": 1.0}
INF = float("inf")


@dataclass
class AtomSpec:
    var: str
    elements: frozenset | None = None
    not_elements: frozenset = frozenset()
    h_exact: int | None = None
    h_min: int = 0
    h_max: int | None = None
    charge: int | None = None
    charge_sign: str | None = None  # "+", "-", "0"
    rings: frozenset = frozenset()
    in_ring: bool = False
    not_in_ring: bool = False
    not_rings: frozenset = frozenset()
    cip: str | None = None
    steroid: int | None = None
    notes_extra: tuple = ()  # (text,) notes for literals without a dedicated slot
    role: str = "main"  # main | anchor | counted
    contradiction: bool = False
    mult: int = 1  # >1: a contracted run of identical chain atoms, drawn as (CH2)n

    def eff_elements(self) -> frozenset | None:
        els = self.elements
        if self.steroid is not None:
            els = frozenset({"c"}) if els is None else els & {"c"}
        if els is not None and self.not_elements:
            els = els - self.not_elements
        return els

    def facts(self) -> set:
        """Requirements this spec places on an atom (for homomorphism checks)."""
        f = set()
        if self.elements is not None:
            f.add(("el", self.elements))
        f |= {("notel", e) for e in self.not_elements}
        if self.h_exact is not None:
            f.add(("hx", self.h_exact))
        if self.h_min:
            f.add(("hmin", self.h_min))
        if self.h_max is not None:
            f.add(("hmax", self.h_max))
        if self.charge is not None:
            f.add(("chg", self.charge))
        if self.charge_sign:
            f.add(("sgn", self.charge_sign))
        f |= {("ring", r) for r in self.rings}
        if self.in_ring:
            f.add(("inring",))
        if self.not_in_ring:
            f.add(("noring",))
        f |= {("noringn", r) for r in self.not_rings}
        if self.cip:
            f.add(("cip", self.cip))
        if self.steroid is not None:
            f.add(("st", self.steroid))
        f |= {("x", t) for t in self.notes_extra}
        return f

    def satisfies(self, req: tuple) -> bool:
        """Whether an atom with this spec is guaranteed to meet requirement ``req``."""
        k = req[0]
        if k == "el":
            e = self.eff_elements()
            return e is not None and e <= req[1]
        if k == "hmin":
            return (
                self.h_exact is not None and self.h_exact >= req[1]
            ) or self.h_min >= req[1]
        if k == "hmax":
            return (self.h_exact is not None and self.h_exact <= req[1]) or (
                self.h_max is not None and self.h_max <= req[1]
            )
        if k == "sgn":
            if self.charge is not None:
                return {
                    "+": self.charge > 0,
                    "-": self.charge < 0,
                    "0": self.charge == 0,
                }[req[1]]
            return self.charge_sign == req[1]
        if k == "inring":
            return self.in_ring or bool(self.rings) or self.steroid is not None
        if k == "notel":
            e = self.eff_elements()
            return req[1] in self.not_elements or (e is not None and req[1] not in e)
        return req in self.facts()


@dataclass
class Bond:
    type: str = "any"
    stereo: frozenset = frozenset()
    contradiction: bool = False


@dataclass
class Pattern:
    mol: str | None
    atoms: dict = field(default_factory=dict)  # var -> AtomSpec (insertion ordered)
    bonds: dict = field(default_factory=dict)  # frozenset({u, v}) -> Bond
    distinct: set = field(default_factory=set)  # explicit frozenset({u, v})
    mol_conds: list = field(default_factory=list)  # (pred, neg)
    aggs: list = field(default_factory=list)
    negs: list = field(default_factory=list)
    ors: list = field(default_factory=list)
    opaque: list = field(default_factory=list)  # (marker, pred, args, description, neg)
    other: list = field(default_factory=list)  # text conditions not drawn
    neg_notes: dict = field(default_factory=dict)  # var -> [text] from simple NegGroups

    def nbrs(self, v) -> list:
        return [next(iter(k - {v})) for k in self.bonds if v in k and len(k) == 2]

    def bond(self, u, v) -> Bond | None:
        return self.bonds.get(frozenset((u, v)))

    def graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(self.atoms)
        g.add_edges_from(tuple(k) for k in self.bonds)
        return g

    def components(self) -> list[list[str]]:
        order = list(self.atoms)
        comps = [
            sorted(c, key=order.index) for c in nx.connected_components(self.graph())
        ]
        return sorted(comps, key=lambda c: order.index(c[0]))


# ── building ───────────────────────────────────────────────────────────────


def apply_unary(
    spec: AtomSpec, pred: str, neg: bool, elemsets: dict, arg2: str | None = None
) -> AtomSpec:
    """Spec with the unary literal ``[not] pred(X)`` added; None if the literal has no slot."""
    m = None
    if pred in ELEMENTS or pred in elemsets or pred == "elem_in":
        els = (
            frozenset({pred})
            if pred in ELEMENTS
            else elemsets.get(pred) or frozenset(arg2.split("|"))
        )
        if neg:
            return replace(spec, not_elements=spec.not_elements | els)
        new = els if spec.elements is None else spec.elements & els
        return replace(spec, elements=new, contradiction=spec.contradiction or not new)
    if m := re.fullmatch(r"has_(\d+)_hs", pred):
        n = int(m.group(1))
        if neg:
            return None
        return replace(
            spec,
            h_exact=n,
            contradiction=spec.contradiction or (spec.h_exact not in (None, n)),
        )
    if m := re.fullmatch(r"has_at_least_(\d+)_hs", pred):
        n = int(m.group(1))
        if neg:
            hm = n - 1 if spec.h_max is None else min(spec.h_max, n - 1)
            return replace(spec, h_max=hm)
        return replace(spec, h_min=max(spec.h_min, n))
    if pred == "charge0" and not neg:
        return replace(spec, charge=0)
    if pred in ("charge_p", "charge_n") and not neg:
        return replace(spec, charge_sign="+" if pred == "charge_p" else "-")
    if (m := re.fullmatch(r"charge(\d+)", pred)) and not neg:
        return replace(spec, charge=int(m.group(1)))
    if (m := re.fullmatch(r"charge_m(\d+)", pred)) and not neg:
        return replace(spec, charge=-int(m.group(1)))
    if pred == "in_ring":
        return replace(spec, not_in_ring=True) if neg else replace(spec, in_ring=True)
    if m := re.fullmatch(r"in_ring(\d+)", pred):
        n = int(m.group(1))
        return (
            replace(spec, not_rings=spec.not_rings | {n})
            if neg
            else replace(spec, rings=spec.rings | {n})
        )
    if (m := re.fullmatch(r"cip_code_([RS])", pred)) and not neg:
        return replace(spec, cip=m.group(1))
    if (m := re.fullmatch(r"steroid_(\d+)", pred)) and not neg:
        return replace(spec, steroid=int(m.group(1)))
    return None


def unary_text(pred: str, neg: bool) -> str:
    """Short note for a unary literal that has no slot in AtomSpec."""
    m = re.fullmatch(r"has_(\d+)_hs", pred)
    if m:
        t = f"{m.group(1)} H"
    elif pred.startswith("charge"):
        t = (
            pred.replace("charge_m", "charge -")
            .replace("charge_p", "positive")
            .replace("charge_n", "negative")
        )
    else:
        t = pred.replace("aux_", "").replace("_", " ")
    return f"not {t}" if neg else t


def _other_text(it) -> str:
    """Plain wording for a condition the drawing has no place for."""
    if isinstance(it, Lit):
        return ("not: " if it.neg else "") + re.sub(r"^aux_", "", it.pred).replace(
            "_", " "
        )
    return "a further numeric condition"


def build_pattern(
    items: list,
    mol: str | None,
    elemsets: dict,
    aux_desc: dict,
    anchors: dict | None = None,
    counted: set | None = None,
    marker_start: int = 1,
) -> Pattern:
    """Pattern for a conjunction of body items; ``anchors`` are atoms of an enclosing pattern."""
    anchors = anchors or {}
    counted = counted or set()
    pat = Pattern(mol)
    for v, spec in anchors.items():
        pat.atoms[v] = replace(spec, role="anchor", notes_extra=())

    def atom(v):
        if v not in pat.atoms:
            pat.atoms[v] = AtomSpec(v, role="counted" if v in counted else "main")
        return pat.atoms[v]

    marker = marker_start
    for it in items:
        if isinstance(it, Lit):
            args = it.args
            if it.pred == "has_atom" and not it.neg and len(args) == 2:
                if args[0] == mol and is_var(args[1]) and args[1] != "_":
                    atom(args[1])
                continue
            if len(args) == 1 and args[0] == mol and (it.pred in MOL_PREDS):
                pat.mol_conds.append((it.pred, it.neg))
                continue
            if (
                (len(args) == 1 or (it.pred == "elem_in" and len(args) == 2))
                and is_var(args[0])
                and args[0] != mol
            ):
                spec = atom(args[0])
                new = apply_unary(
                    spec, it.pred, it.neg, elemsets, args[1] if len(args) == 2 else None
                )
                if new is None:
                    if it.pred.startswith("aux_"):
                        pat.opaque.append(
                            (marker, it.pred, args, aux_desc.get(it.pred, ""), it.neg)
                        )
                        new = replace(
                            spec, notes_extra=spec.notes_extra + (f"†{marker}",)
                        )
                        marker += 1
                    else:
                        new = replace(
                            spec,
                            notes_extra=spec.notes_extra
                            + (unary_text(it.pred, it.neg),),
                        )
                pat.atoms[args[0]] = new
                continue
            if (
                len(args) == 2
                and (it.pred in BOND_PREDS or it.pred in STEREO_PREDS)
                and all(is_var(a) for a in args)
                and not it.neg
            ):
                u, v = args
                atom(u), atom(v)
                key = frozenset((u, v))
                b = pat.bonds.get(key, Bond())
                if it.pred in STEREO_PREDS:
                    b = replace(b, stereo=b.stereo | {STEREO_PREDS[it.pred]})
                else:
                    t = BOND_PREDS[it.pred]
                    if b.type == "any":
                        b = replace(b, type=t)
                    elif t not in ("any", b.type):
                        b = replace(b, contradiction=True)
                pat.bonds[key] = b
                if u == v:
                    pat.atoms[u] = replace(pat.atoms[u], contradiction=True)
                continue
            if (
                it.pred.startswith("aux_")
                or is_fg_seed_name(it.pred)
                or len(args) == 1
                and args[0] == mol
            ):
                atom_args = [a for a in args if is_var(a) and a != mol and a != "_"]
                if atom_args:
                    first = atom(atom_args[0])
                    pat.atoms[atom_args[0]] = replace(
                        first, notes_extra=first.notes_extra + (f"†{marker}",)
                    )
                pat.opaque.append(
                    (marker, it.pred, args, aux_desc.get(it.pred, ""), it.neg)
                )
                marker += 1
                continue
            pat.other.append(_other_text(it))
        elif isinstance(it, Cmp):
            lhs, rhs = it.left, it.right
            if (
                it.op in ("!=", "<", ">")
                and is_var(lhs)
                and is_var(rhs)
                and mol not in (lhs, rhs)
            ):
                atom(lhs), atom(rhs)
                pat.distinct.add(frozenset((lhs, rhs)))
            else:
                pat.other.append(_other_text(it))
        elif isinstance(it, Agg):
            pat.aggs.append(it)
        elif isinstance(it, NegGroup):
            pat.negs.append(it)
        elif isinstance(it, OrGroup):
            pat.ors.append(it)
        else:
            pat.other.append(_other_text(it))
    _absorb_simple_negs(pat, elemsets)
    _drop_lonely_atoms(pat)
    return pat


def _drop_lonely_atoms(pat: Pattern):
    """Atoms with no property, bond or reference are only has_atom guards (always true)."""
    used = (
        {v for k in pat.bonds for v in k}
        | {v for k in pat.distinct for v in k}
        | set(pat.neg_notes)
    )
    for group in pat.aggs + pat.negs + pat.ors:
        used |= set(item_vars(group))
    for _, _, args, _, _ in pat.opaque:
        used |= set(args)
    for v in list(pat.atoms):
        s = pat.atoms[v]
        if v not in used and s.role == "main" and not s.facts() and not s.contradiction:
            del pat.atoms[v]


def strip_anchor_restatements(alt: tuple, anchors: dict, elemsets: dict) -> list:
    """Alternative without unary literals on anchors that the anchor spec already guarantees."""
    out = []
    for it in alt:
        if (
            isinstance(it, Lit)
            and not it.neg
            and len(it.args) == 1
            and it.args[0] in anchors
        ):
            spec = anchors[it.args[0]]
            probe = apply_unary(AtomSpec("_"), it.pred, False, elemsets)
            if probe is not None and all(spec.satisfies(f) for f in probe.facts()):
                continue
        if isinstance(it, Lit) and it.pred == "has_atom":
            continue
        out.append(it)
    return out


def _absorb_simple_negs(pat: Pattern, elemsets: dict):
    """``not (X has a double bond to some atom)`` becomes the note "no double bond" on X."""
    keep = []
    for ng in pat.negs:
        note = None
        if len(ng.alternatives) == 1:
            alt = strip_anchor_restatements(ng.alternatives[0], pat.atoms, elemsets)
            bonds = [
                b
                for b in alt
                if isinstance(b, Lit) and b.pred in BOND_PREDS and not b.neg
            ]
            if len(bonds) == 1:
                u, v = bonds[0].args
                anchor, local = (
                    (u, v)
                    if u in pat.atoms and v not in pat.atoms
                    else (v, u)
                    if v in pat.atoms and u not in pat.atoms
                    else (None, None)
                )
                rest = [b for b in alt if b is not bonds[0]]
                el = None
                if anchor and len(rest) <= 1:
                    if (
                        rest
                        and isinstance(rest[0], Lit)
                        and not rest[0].neg
                        and rest[0].args == (local,)
                        and rest[0].pred in ELEMENTS
                    ):
                        el = rest[0].pred.capitalize()
                        rest = []
                    if not rest:
                        t = BOND_PREDS[bonds[0].pred]
                        what = "bond" if t == "any" else f"{t} bond"
                        note = (anchor, f"no {what}" + (f" to {el}" if el else ""))
        if note:
            pat.neg_notes.setdefault(note[0], []).append(note[1])
        else:
            keep.append(ng)
    pat.negs = keep


# ── distinctness ───────────────────────────────────────────────────────────

_MAXVAL = {"h": 1, "f": 1, "c": 4, "b": 4, "n": 4, "o": 3}


def max_valence(els: frozenset | None, charge: int | None) -> float:
    if els is None:
        return INF
    best = 0.0
    for e in els:
        v = _MAXVAL.get(e, INF)
        if charge is not None:
            if e == "n":
                v = {0: 3, 1: 4}.get(charge, 2 if charge < 0 else 4)
            elif e == "o":
                v = {0: 2, 1: 3}.get(charge, 1 if charge < 0 else 3)
            elif e == "c":
                v = 4 if charge == 0 else 3
        best = max(best, v)
    return best


def specs_incompatible(a: AtomSpec, b: AtomSpec) -> bool:
    ea, eb = a.eff_elements(), b.eff_elements()
    if ea is not None and eb is not None and not (ea & eb):
        return True
    if (
        ea is not None
        and ea <= b.not_elements
        or eb is not None
        and eb <= a.not_elements
    ):
        return True
    if a.h_exact is not None and b.h_exact is not None and a.h_exact != b.h_exact:
        return True
    for x, y in ((a, b), (b, a)):
        if x.h_exact is not None and (
            x.h_exact < y.h_min or (y.h_max is not None and x.h_exact > y.h_max)
        ):
            return True
        if x.h_max is not None and y.h_min > x.h_max:
            return True
        if x.not_in_ring and (y.in_ring or y.rings or y.steroid is not None):
            return True
        if x.not_rings & y.rings:
            return True
    if a.charge is not None and b.charge is not None and a.charge != b.charge:
        return True
    sa = a.charge_sign or (
        None
        if a.charge is None
        else "+"
        if a.charge > 0
        else "-"
        if a.charge < 0
        else "0"
    )
    sb = b.charge_sign or (
        None
        if b.charge is None
        else "+"
        if b.charge > 0
        else "-"
        if b.charge < 0
        else "0"
    )
    if sa and sb and sa != sb:
        return True
    if a.cip and b.cip and a.cip != b.cip:
        return True
    if a.steroid is not None and b.steroid is not None and a.steroid != b.steroid:
        return True
    return False


def merged_spec(a: AtomSpec, b: AtomSpec) -> AtomSpec:
    ea, eb = a.eff_elements(), b.eff_elements()
    els = ea if eb is None else eb if ea is None else ea & eb
    h = max(x for x in (a.h_exact, b.h_exact, a.h_min, b.h_min) if x is not None)
    return AtomSpec(
        "_",
        elements=els,
        h_exact=h,
        charge=a.charge if a.charge is not None else b.charge,
    )


def _h_capable(pat: Pattern, w: str, bond: Bond) -> bool:
    s = pat.atoms[w]
    e = s.eff_elements()
    return (
        (e is None or "h" in e)
        and bond.type in ("single", "any")
        and len(pat.nbrs(w)) == 1
        and not (s.in_ring or s.rings or s.steroid is not None)
        and s.h_min == 0
        and s.h_exact in (None, 0)
    )


def distinct_pairs(pat: Pattern, scaffold: bool = False) -> set:
    """Pairs of atoms that the clause forces to be different atoms."""
    atoms = list(pat.atoms)
    dist = set()
    for k in pat.distinct:
        dist.add(k)
    for k in pat.bonds:
        if len(k) == 2:
            dist.add(k)
    for u, v in itertools.combinations(atoms, 2):
        if specs_incompatible(pat.atoms[u], pat.atoms[v]):
            dist.add(frozenset((u, v)))
    # common neighbour reached by bonds of different types
    for w in atoms:
        nb = pat.nbrs(w)
        for u, v in itertools.combinations(nb, 2):
            tu, tv = pat.bond(u, w).type, pat.bond(v, w).type
            if "any" not in (tu, tv) and tu != tv:
                dist.add(frozenset((u, v)))
    g = pat.graph()
    changed = True
    while changed:
        changed = False
        for u, v in itertools.combinations(atoms, 2):
            key = frozenset((u, v))
            if key in dist:
                continue
            if _valence_excludes(pat, u, v, dist) or _path_excludes(pat, g, u, v, dist):
                dist.add(key)
                changed = True
    return dist


def _valence_excludes(pat, u, v, dist) -> bool:
    m = merged_spec(pat.atoms[u], pat.atoms[v])
    cap = max_valence(m.eff_elements(), m.charge)
    if cap == INF:
        return False
    nb = {}
    for x in (u, v):
        for w in pat.nbrs(x):
            if w in (u, v):
                continue
            b = pat.bond(x, w)
            prev = nb.get(w)
            nb[w] = (
                b
                if prev is None
                or ORDER[b.type] > ORDER[prev.type]
                or prev.type == "any"
                else prev
            )
    ws = list(nb)
    hx = m.h_exact or 0
    if hx > cap:
        return True
    for r in range(len(ws), 0, -1):
        if len(ws) > 10 and r < len(ws) - 2:
            break
        for sub in itertools.combinations(ws, r):
            if any(
                frozenset((a, b)) not in dist for a, b in itertools.combinations(sub, 2)
            ):
                continue
            heavy = sum(
                VALENCE_LB[nb[w].type] for w in sub if not _h_capable(pat, w, nb[w])
            )
            hcap = sum(1 for w in sub if _h_capable(pat, w, nb[w]))
            if heavy + max(hx, hcap) > cap:
                return True
    return False


def _path_excludes(pat, g, u, v, dist) -> bool:
    """A non-backtracking walk whose inner atoms are all outside rings visits distinct atoms."""
    try:
        paths = nx.all_simple_paths(g, u, v, cutoff=16)
        for path in paths:
            if len(path) < 3:
                continue
            inner = path[1:-1]
            if not all(pat.atoms[x].not_in_ring for x in inner):
                continue
            if all(
                frozenset((path[i - 1], path[i + 1])) in dist
                for i in range(1, len(path) - 1)
            ):
                return True
    except nx.NetworkXNoPath:
        pass
    return False


def may_coincide(pat: Pattern) -> list[tuple[str, str]]:
    dist = distinct_pairs(pat)
    out = []
    for u, v in itertools.combinations(list(pat.atoms), 2):
        if frozenset((u, v)) not in dist:
            if pat.atoms[u].role == "anchor" and pat.atoms[v].role == "anchor":
                continue  # handled in the enclosing pattern
            out.append((u, v))
    return out


# ── core (theta-subsumption) ───────────────────────────────────────────────


def _locally_distinct(pat, a, b) -> bool:
    return (
        frozenset((a, b)) in pat.distinct
        or pat.bond(a, b) is not None
        or specs_incompatible(pat.atoms[a], pat.atoms[b])
    )


def _find_folding(pat: Pattern, x: str, fixed: set):
    """A homomorphism of the pattern into itself that avoids ``x``, or None."""
    atoms = list(pat.atoms)
    reqs = {u: pat.atoms[u].facts() for u in atoms}
    dom = {}
    for u in atoms:
        if u in fixed:
            dom[u] = [u] if u != x else []
        else:
            cands = [
                w
                for w in atoms
                if w != x and all(pat.atoms[w].satisfies(r) for r in reqs[u])
            ]
            dom[u] = sorted(cands, key=lambda w: w != u)
        if not dom[u]:
            return None
    g = pat.graph()
    order = list(nx.bfs_tree(g, x)) + [
        a for a in atoms if a not in nx.node_connected_component(g, x)
    ]
    h = {}

    def ok(u, w):
        for nb in pat.nbrs(u):
            if nb in h:
                b = pat.bond(u, nb)
                tb = pat.bond(w, h[nb])
                if (
                    w == h[nb]
                    or tb is None
                    or (b.type != "any" and tb.type != b.type)
                    or not b.stereo <= tb.stereo
                ):
                    return False
        for k in pat.distinct:
            if u in k:
                o = next(iter(k - {u}))
                if o in h and (w == h[o] or not _locally_distinct(pat, w, h[o])):
                    return False
        return True

    def bt(i):
        if i == len(order):
            return True
        u = order[i]
        for w in dom[u]:
            if ok(u, w):
                h[u] = w
                if bt(i + 1):
                    return True
                del h[u]
        return False

    return dict(h) if bt(0) else None


def core(pat: Pattern) -> tuple[Pattern, dict]:
    """Drop atoms whose requirements other atoms already meet (the clause's theta-core)."""
    fixed = {v for v, s in pat.atoms.items() if s.role != "main"}
    for group in pat.aggs + pat.negs + pat.ors:
        fixed |= set(item_vars(group)) & set(pat.atoms)
    for _, _, args, _, _ in pat.opaque:
        fixed |= set(args) & set(pat.atoms)
    fixed |= set(pat.neg_notes)
    folded = {}
    changed = True
    while changed:
        changed = False
        for x in list(pat.atoms):
            if x in fixed:
                continue
            h = _find_folding(pat, x, fixed)
            if h is None:
                continue
            image = set(h.values())
            folded.update({u: h[u] for u in pat.atoms if u not in image})
            pat = Pattern(
                pat.mol,
                {v: s for v, s in pat.atoms.items() if v in image},
                {k: b for k, b in pat.bonds.items() if k <= image},
                {k for k in pat.distinct if k <= image},
                pat.mol_conds,
                pat.aggs,
                pat.negs,
                pat.ors,
                pat.opaque,
                pat.other,
                pat.neg_notes,
            )
            changed = True
            break
    return pat, folded


# ── display transforms ─────────────────────────────────────────────────────


def tighten_bonds(pat: Pattern, dist: set) -> Pattern:
    """Any-type bonds that valence forces to be single (e.g. inside a CH2 chain)."""
    bonds = dict(pat.bonds)
    for v, s in pat.atoms.items():
        els = s.eff_elements()
        if s.h_exact is None or els is None or len(els) != 1:
            continue
        cap = max_valence(els, s.charge)
        nb = pat.nbrs(v)
        if cap == INF or any(
            frozenset((a, b)) not in dist for a, b in itertools.combinations(nb, 2)
        ):
            continue
        if sum(VALENCE_LB[pat.bond(v, w).type] for w in nb) + s.h_exact != cap:
            continue
        for w in nb:
            b = bonds[frozenset((v, w))]
            # aromatic bonds need a ring, so a non-ring end rules them out
            if b.type == "any" and (s.not_in_ring or pat.atoms[w].not_in_ring):
                bonds[frozenset((v, w))] = replace(b, type="single")
    return replace(pat, bonds=bonds)


def _signature(s: AtomSpec):
    return (
        s.elements,
        s.not_elements,
        s.h_exact,
        s.h_min,
        s.h_max,
        s.charge,
        s.charge_sign,
        s.rings,
        s.in_ring,
        s.not_in_ring,
        s.not_rings,
        s.cip,
        s.steroid,
        s.notes_extra,
        s.role,
    )


def contract_chains(pat: Pattern, dist: set, keep: set, min_len: int = 3) -> Pattern:
    """Runs of identical, pairwise distinct chain atoms joined by single bonds become one (CH2)n atom."""
    cand = [
        v
        for v, s in pat.atoms.items()
        if s.role == "main"
        and v not in keep
        and len(pat.nbrs(v)) == 2
        and s.charge in (None, 0)
        and s.steroid is None
    ]
    g = nx.Graph()
    g.add_nodes_from(cand)
    for u, v in itertools.combinations(cand, 2):
        b = pat.bond(u, v)
        if (
            b is not None
            and b.type == "single"
            and not b.stereo
            and _signature(pat.atoms[u]) == _signature(pat.atoms[v])
        ):
            g.add_edge(u, v)
    atoms, bonds = dict(pat.atoms), dict(pat.bonds)
    for comp in list(nx.connected_components(g)):
        sub = g.subgraph(comp)
        if (
            len(comp) < min_len
            or not nx.is_tree(sub)
            or max(d for _, d in sub.degree()) > 2
        ):
            continue
        if any(
            frozenset((a, b)) not in dist for a, b in itertools.combinations(comp, 2)
        ):
            continue
        ends = [v for v, d in sub.degree() if d == 1]
        outer = []
        for e in ends:
            for w in pat.nbrs(e):
                if w not in comp:
                    outer.append((w, bonds[frozenset((e, w))]))
        if len(outer) != 2 or outer[0][0] == outer[1][0]:
            continue
        rep = ends[0]
        for v in comp:
            for w in pat.nbrs(v):
                bonds.pop(frozenset((v, w)), None)
            if v != rep:
                atoms.pop(v)
        atoms[rep] = replace(atoms[rep], mult=len(comp))
        for w, b in outer:
            bonds[frozenset((rep, w))] = b
    return replace(pat, atoms=atoms, bonds=bonds)
