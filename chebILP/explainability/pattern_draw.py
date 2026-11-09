"""Drawing a Pattern: RDKit draws atoms and typed bonds, PIL adds the rest.

PIL draws any-type bonds (dashed), the steroid scaffold (grey), atom notes, number badges
and stereo bond notes, so all annotation text can use Unicode and its placement is ours.
Everything is drawn at SS times the size and scaled down, for antialiased PIL lines.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageDraw
from rdkit import Chem, Geometry
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from chebILP.explainability.pattern import AtomSpec, Pattern
from chebILP.explainability.style import C, FONT, HIGHLIGHT, font, text_size

SS = 2
GENERIC = "*"  # label of an atom whose element is not fixed

HALO = {
    "neg": HIGHLIGHT["absent"],
    "counted": HIGHLIGHT["match"],
    "steroid": HIGHLIGHT["match"],
}
# RDKit's default atom colours, for text drawn next to an atom label
_ATOM_RGB = {
    0: tuple(c / 255 for c in (97, 110, 124)),
    1: (0.35, 0.35, 0.35),
    7: (0.0, 0.0, 1.0),
    8: (1.0, 0.0, 0.0),
    9: (0.2, 0.8, 0.8),
    15: (1.0, 0.5, 0.0),
    16: (0.8, 0.8, 0.0),
    17: (0.0, 0.8, 0.0),
    35: (0.5, 0.08, 0.12),
    53: (0.63, 0.12, 0.94),
}

# ── steroid scaffold ───────────────────────────────────────────────────────

GONANE_XY = {
    1: (0.0, 1.0),
    2: (-0.866, 0.5),
    3: (-0.866, -0.5),
    4: (0.0, -1.0),
    5: (0.866, -0.5),
    10: (0.866, 0.5),
    6: (1.732, -1.0),
    7: (2.598, -0.5),
    8: (2.598, 0.5),
    9: (1.732, 1.0),
    11: (1.732, 2.0),
    12: (2.598, 2.5),
    13: (3.464, 2.0),
    14: (3.464, 1.0),
    15: (4.415, 0.691),
    16: (5.003, 1.5),
    17: (4.415, 2.309),
}
GONANE_BONDS = [
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 10),
    (10, 1),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (9, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 8),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 13),
]
_RING_CENTRES = {
    "A": (0.0, 0.0),
    "B": (1.732, 0.0),
    "C": (2.598, 1.5),
    "D": (4.152, 1.5),
}
_RINGS = {
    "A": {1, 2, 3, 4, 5, 10},
    "B": {5, 6, 7, 8, 9, 10},
    "C": {8, 9, 11, 12, 13, 14},
    "D": {13, 14, 15, 16, 17},
}


def gonane_neighbours(k: int) -> list[int]:
    return sorted(
        {b for a, b in GONANE_BONDS if a == k} | {a for a, b in GONANE_BONDS if b == k}
    )


# ── labels and notes ───────────────────────────────────────────────────────


def _charge_sup(q: int) -> str:
    if q == 0:
        return ""
    n = "" if abs(q) == 1 else str(abs(q))
    return f"{n}{'+' if q > 0 else '-'}"


def element_symbol(spec: AtomSpec) -> tuple[str, int]:
    els = spec.eff_elements()
    if els is None or len(els) != 1:
        if els is None:
            return GENERIC, 0
        return "/".join(
            e.capitalize() for e in sorted(els, key=lambda e: (e != "c", e))
        ), 0
    sym = next(iter(els)).capitalize()
    return sym, Chem.GetPeriodicTable().GetAtomicNumber(sym)


def atom_label(spec: AtomSpec, h_left: bool = False) -> tuple[str, str, int]:
    """(RDKit label markup, plain text for sizing, atomic number for colour)."""
    sym, z = element_symbol(spec)
    hpart, hplain = "", ""
    if spec.h_exact and z not in (0, 1):
        hpart = "H" if spec.h_exact == 1 else f"H<sub>{spec.h_exact}</sub>"
        hplain = "H" if spec.h_exact == 1 else f"H{spec.h_exact}"
    if spec.mult > 1:
        return f"({sym}{hpart})<sub>{spec.mult}</sub>", f"({sym}{hplain}){spec.mult}", z
    # the charge is drawn by PIL (RDKit's superscript hyphen is too small to read)
    if h_left and hpart:
        return f"{hpart}{sym}", f"{hplain}{sym}", z
    return f"{sym}{hpart}", f"{sym}{hplain}", z


def charge_text(spec: AtomSpec) -> str:
    if spec.charge in (None, 0) or spec.mult > 1:
        return ""
    n = "" if abs(spec.charge) == 1 else str(abs(spec.charge))
    return f"{n}{'+' if spec.charge > 0 else '−'}"


def atom_notes(
    spec: AtomSpec, label_is_generic: bool, on_scaffold: bool, extra: list | None = None
) -> list[str]:
    notes = []
    if spec.cip:
        notes.append(f"({spec.cip})")
    if spec.h_exact == 0:
        notes.append("no H")
    elif spec.h_exact and label_is_generic:
        notes.append(f"{spec.h_exact} H")
    if spec.h_exact is None and spec.h_min:
        notes.append(f"≥{spec.h_min} H")
    if spec.h_max is not None and spec.h_exact is None:
        notes.append("no H" if spec.h_max == 0 else f"≤{spec.h_max} H")
    if spec.charge is None and spec.charge_sign:
        notes.append(
            {"+": "positively charged", "-": "negatively charged", "0": "uncharged"}[
                spec.charge_sign
            ]
        )
    if spec.charge == 0:
        notes.append("uncharged")
    if spec.rings:
        rs = sorted(spec.rings)
        notes.append(
            f"in {rs[0]}-ring"
            if len(rs) == 1
            else "in " + ", ".join(f"{r}-" for r in rs[:-1]) + f" and {rs[-1]}-ring"
        )
    elif spec.in_ring and not (on_scaffold and spec.steroid is not None):
        notes.append("in a ring")
    if spec.not_in_ring:
        notes.append("not in a ring")
    for r in sorted(spec.not_rings):
        notes.append(f"not in {r}-ring")
    if spec.steroid is not None and not on_scaffold:
        notes.append(f"steroid C{spec.steroid}")
    if spec.not_elements and (
        spec.elements is None or len(spec.eff_elements() or ()) != 1
    ):
        els = sorted(e.capitalize() for e in spec.not_elements)
        notes.append(
            "not " + (", ".join(els) if len(els) <= 4 else ", ".join(els[:4]) + ", …")
        )
    notes.extend(spec.notes_extra)
    notes.extend(extra or [])
    return notes


# ── geometry helpers ───────────────────────────────────────────────────────


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _angle_between(u, v) -> float:
    d = abs(math.atan2(u[1], u[0]) - math.atan2(v[1], v[0])) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def dashed_line(draw, p, q, color, width=2, dash=6, gap=4):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    if L < 1:
        return
    ux, uy = dx / L, dy / L
    t = 0.0
    while t < L:
        e = min(t + dash, L)
        draw.line(
            [(p[0] + ux * t, p[1] + uy * t), (p[0] + ux * e, p[1] + uy * e)],
            fill=color,
            width=width,
        )
        t = e + gap


def _shorten(p, q, rp, rq):
    L = math.dist(p, q) or 1.0
    ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
    return (p[0] + ux * rp, p[1] + uy * rp), (q[0] - ux * rq, q[1] - uy * rq)


def _box_radius(half, u) -> float:
    """Distance from a box centre to its edge along unit vector u."""
    hw, hh = half
    if hw <= 0 and hh <= 0:
        return 0.0
    tx = hw / abs(u[0]) if abs(u[0]) > 1e-6 else math.inf
    ty = hh / abs(u[1]) if abs(u[1]) > 1e-6 else math.inf
    return min(tx, ty)


# ── note boxes ─────────────────────────────────────────────────────────────


@dataclass
class NoteBox:
    lines: list  # list of token lists: ("text", s) | ("badge", n)
    size: int
    color: tuple

    def measure(self):
        f = font(self.size)
        w, h = 0, 0
        for line in self.lines:
            lw = sum(
                self.size * 1.4 if kind == "badge" else f.getlength(val)
                for kind, val in line
            )
            w = max(w, lw)
            h += int(self.size * 1.3)
        return w, h

    def draw(self, draw, x, y):
        f = font(self.size)
        fb = font(max(8, int(self.size * 0.78)), bold=True)
        for line in self.lines:
            cx = x
            for kind, val in line:
                if kind == "badge":
                    r = self.size * 0.55
                    cy = y + self.size * 0.65
                    draw.ellipse([cx, cy - r, cx + 2 * r, cy + r], fill=C["badge"])
                    draw.text(
                        (cx + r, cy), str(val), font=fb, fill=C["white"], anchor="mm"
                    )
                    cx += 2 * r + self.size * 0.3
                else:
                    draw.text((cx, y), val, font=f, fill=self.color)
                    cx += f.getlength(val)
            y += int(self.size * 1.3)


def wrap_notes(
    notes: list[str], badge, size: int, color, max_chars: int = 18
) -> NoteBox | None:
    badges = (
        []
        if badge is None
        else list(badge)
        if isinstance(badge, (list, tuple))
        else [badge]
    )
    if not notes and not badges:
        return None
    lines, cur, cur_len = [], [], 0
    for b in badges:
        cur.append(("badge", b))
        cur_len += 2
    for n in notes:
        add = (", " if cur and cur[-1][0] == "text" else "") + n
        if (
            cur_len
            and cur_len + len(add) > max_chars
            and any(k == "text" for k, _ in cur)
        ):
            lines.append(cur)
            cur, cur_len, add = [], 0, n
        cur.append(("text", add))
        cur_len += len(add)
    if cur:
        lines.append(cur)
    return NoteBox(lines, size, color)


# ── main drawing ───────────────────────────────────────────────────────────


@dataclass
class DrawStyle:
    bond_px: float = 64
    font_px: int = 17
    note_px: int = 13
    bg: tuple = (255, 255, 255)
    halo_local: str | None = (
        None  # halo kind for non-anchor atoms ("neg" in must-not panels)
    )
    scaffold_scale: float = 0.7


@dataclass
class Drawn:
    image: Image.Image
    numbers_used: set
    scaffold: bool
    any_bond: bool
    generic_atom: bool
    aromatic: bool
    explicit_h: bool


def _normalise(mol, ref_pair=None):
    conf = mol.GetConformer()
    if ref_pair:
        p1, p2 = conf.GetAtomPosition(ref_pair[0]), conf.GetAtomPosition(ref_pair[1])
        ref = math.hypot(p1.x - p2.x, p1.y - p2.y)
    else:
        bl = []
        for b in mol.GetBonds():
            p, q = (
                conf.GetAtomPosition(b.GetBeginAtomIdx()),
                conf.GetAtomPosition(b.GetEndAtomIdx()),
            )
            bl.append(math.hypot(p.x - q.x, p.y - q.y))
        ref = sorted(bl)[len(bl) // 2] if bl else 1.5
    s = 1.0 / max(ref, 1e-6)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Geometry.Point3D(p.x * s, p.y * s, 0))


def _layout(pat: Pattern, use_scaffold: bool):
    """RDKit mol with 2D coords (unit bonds); returns (mol, var->idx, scaffold pos->idx)."""
    rw = Chem.RWMol()
    idx, spos = {}, {}
    st_atoms = (
        {s.steroid: v for v, s in pat.atoms.items() if s.steroid is not None}
        if use_scaffold
        else {}
    )
    if use_scaffold:
        for k in GONANE_XY:
            a = Chem.Atom(6)
            a.SetNoImplicit(True)
            spos[k] = rw.AddAtom(a)
            if k in st_atoms:
                idx[st_atoms[k]] = spos[k]
    for v in pat.atoms:
        if v not in idx:
            a = Chem.Atom(6)
            a.SetNoImplicit(True)
            idx[v] = rw.AddAtom(a)
    if use_scaffold:
        for a, b in GONANE_BONDS:
            rw.AddBond(spos[a], spos[b], Chem.BondType.SINGLE)
    for k in pat.bonds:
        u, v = tuple(k)
        if idx[u] != idx[v] and rw.GetBondBetweenAtoms(idx[u], idx[v]) is None:
            rw.AddBond(idx[u], idx[v], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    if use_scaffold:
        rdDepictor.SetPreferCoordGen(False)
        rdDepictor.Compute2DCoords(
            mol,
            coordMap={spos[k]: Geometry.Point2D(*xy) for k, xy in GONANE_XY.items()},
        )
        _normalise(mol, (spos[1], spos[2]))
    else:
        _layout_components(mol)
    return mol, idx, spos


def _layout_components(mol):
    """CoordGen per connected component, components placed left to right."""
    frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    conf = Chem.Conformer(mol.GetNumAtoms())
    x0 = 0.0
    for frag in frags:
        sub = Chem.RWMol()
        m = {}
        for i in frag:
            a = Chem.Atom(6)
            a.SetNoImplicit(True)
            m[i] = sub.AddAtom(a)
        for b in mol.GetBonds():
            if b.GetBeginAtomIdx() in m and b.GetEndAtomIdx() in m:
                sub.AddBond(
                    m[b.GetBeginAtomIdx()], m[b.GetEndAtomIdx()], Chem.BondType.SINGLE
                )
        sm = sub.GetMol()
        sm.UpdatePropertyCache(strict=False)
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(sm)
        _normalise(sm)
        sc = sm.GetConformer()
        pts = {i: sc.GetAtomPosition(m[i]) for i in frag}
        minx = min(p.x for p in pts.values())
        maxx = max(p.x for p in pts.values())
        cy = (min(p.y for p in pts.values()) + max(p.y for p in pts.values())) / 2
        for i, p in pts.items():
            conf.SetAtomPosition(i, Geometry.Point3D(p.x - minx + x0, p.y - cy, 0))
        x0 += (maxx - minx) + 1.8
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)


def draw_pattern(
    pat: Pattern,
    style: DrawStyle | None = None,
    numbers: dict | None = None,
    neg_notes: dict | None = None,
    use_scaffold: bool | None = None,
) -> Drawn:
    style = style or DrawStyle()
    numbers = numbers or {}
    neg_notes = neg_notes if neg_notes is not None else pat.neg_notes
    if use_scaffold is None:
        use_scaffold = any(s.steroid is not None for s in pat.atoms.values())
    if not pat.atoms:
        return Drawn(
            Image.new("RGB", (1, 1), style.bg), set(), False, False, False, False, False
        )
    k = SS
    bond_px, font_px, note_px = style.bond_px * k, style.font_px * k, style.note_px * k
    if use_scaffold:
        bond_px *= (
            style.scaffold_scale
        )  # the 17-atom core needs less room per bond than labelled atoms
    mol, idx, spos = _layout(pat, use_scaffold)
    conf = mol.GetConformer()
    xy = {
        i: (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
        for i in range(mol.GetNumAtoms())
    }
    var_of = {i: v for v, i in idx.items()}

    rw = Chem.RWMol(mol)
    flags = dict(any_bond=False, generic=False, aromatic=False, explicit_h=False)
    plain = {}
    for v, i in idx.items():
        spec = pat.atoms[v]
        nbx = [
            xy[n.GetIdx()][0] - xy[i][0] for n in mol.GetAtomWithIdx(i).GetNeighbors()
        ]
        h_left = bool(nbx) and sum(nbx) > 0.3 and all(dx > -0.3 for dx in nbx)
        lab, pl, z = atom_label(spec, h_left)
        a = rw.GetAtomWithIdx(i)
        a.SetAtomicNum(z)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetProp("atomLabel", lab)
        plain[i] = pl
        flags["generic"] |= z == 0 and spec.role != "anchor"
        flags["explicit_h"] |= z == 1
    for i in spos.values():
        if i not in var_of:
            rw.GetAtomWithIdx(i).SetAtomicNum(0)
            rw.GetAtomWithIdx(i).SetProp("atomLabel", " ")
    pil_bonds = []  # (i, j, kind) drawn by PIL
    btypes = {
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
        "aromatic": Chem.BondType.AROMATIC,
    }
    for b in list(mol.GetBonds()):
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        vi, vj = var_of.get(i), var_of.get(j)
        pb = pat.bond(vi, vj) if vi is not None and vj is not None else None
        if pb is None:
            rw.RemoveBond(i, j)
            pil_bonds.append((i, j, "scaffold"))
        elif pb.type == "any":
            rw.RemoveBond(i, j)
            pil_bonds.append((i, j, "any"))
            flags["any_bond"] = True
        else:
            rb = rw.GetBondBetweenAtoms(i, j)
            rb.SetBondType(btypes[pb.type])
            if pb.type == "aromatic":
                rb.SetIsAromatic(True)
                flags["aromatic"] = True
    dm = rw.GetMol()
    dm.UpdatePropertyCache(strict=False)

    xs = [p[0] for p in xy.values()]
    ys = [p[1] for p in xy.values()]
    pad = 2.8 * bond_px
    W = int((max(xs) - min(xs)) * bond_px + 2 * pad)
    H = int((max(ys) - min(ys)) * bond_px + 2 * pad)
    d2 = rdMolDraw2D.MolDraw2DCairo(W, H)
    o = d2.drawOptions()
    o.fontFile = FONT
    o.fixedBondLength = bond_px
    o.fixedFontSize = font_px
    o.bondLineWidth = 2 * k
    o.scaleBondWidth = False
    o.padding = 0.0
    o.multipleBondOffset = 0.16
    o.setBackgroundColour(tuple(c / 255 for c in style.bg) + (1.0,))
    o.updateAtomPalette({0: tuple(c / 255 for c in C["muted"]), 1: (0.35, 0.35, 0.35)})
    o.highlightRadius = 0.4
    hl_atoms, hl_cols = [], {}
    for v, i in idx.items():
        s = pat.atoms[v]
        kind = None
        if s.role == "counted":
            kind = "counted"
        elif s.role != "anchor" and style.halo_local:
            kind = style.halo_local
        elif use_scaffold and s.steroid is not None and s.role != "anchor":
            kind = "steroid"
        if kind:
            hl_atoms.append(i)
            hl_cols[i] = HALO[kind]
    d2.DrawMolecule(
        dm,
        highlightAtoms=hl_atoms,
        highlightAtomColors=hl_cols,
        highlightBonds=[],
        highlightBondColors={},
    )
    d2.FinishDrawing()
    img = Image.open(io.BytesIO(d2.GetDrawingText())).convert("RGB")
    P = {i: tuple(d2.GetDrawCoords(i)) for i in range(dm.GetNumAtoms())}
    draw = ImageDraw.Draw(img)

    lf = font(font_px)
    half = {}
    for i in range(dm.GetNumAtoms()):
        if i in plain:
            tw, _ = text_size(plain[i], lf)
            half[i] = (tw / 2 + 2 * k, font_px * 0.6)
        else:
            half[i] = (0, 0)

    for i, j, kind in pil_bonds:
        p, q = P[i], P[j]
        L = math.dist(p, q) or 1
        u = ((q[0] - p[0]) / L, (q[1] - p[1]) / L)
        rp = _box_radius(half[i], u) + (3 * k if i in plain else 0)
        rq = _box_radius(half[j], (-u[0], -u[1])) + (3 * k if j in plain else 0)
        a, b = _shorten(p, q, rp, rq)
        if kind == "scaffold":
            draw.line([a, b], fill=C["scaffold"], width=int(2.5 * k))
        else:
            dashed_line(draw, a, b, C["any"], width=2 * k, dash=5 * k, gap=4 * k)
    segs = [(P[b.GetBeginAtomIdx()], P[b.GetEndAtomIdx()]) for b in dm.GetBonds()]
    segs += [(P[i], P[j]) for i, j, _ in pil_bonds]
    avoid = []
    for p, q in segs:
        n = max(2, int(math.dist(p, q) / (5 * k)))
        avoid += [
            (p[0] + (q[0] - p[0]) * t / n, p[1] + (q[1] - p[1]) * t / n)
            for t in range(1, n)
        ]
    obstacles = []
    for i in plain:
        hw, hh = half[i]
        obstacles.append((P[i][0] - hw, P[i][1] - hh, P[i][0] + hw, P[i][1] + hh))

    if use_scaffold:
        for pos, i in spos.items():
            rings = [r for r, members in _RINGS.items() if pos in members]
            cx = sum(_RING_CENTRES[r][0] for r in rings) / len(rings)
            cy = sum(_RING_CENTRES[r][1] for r in rings) / len(rings)
            dx, dy = GONANE_XY[pos][0] - cx, GONANE_XY[pos][1] - cy
            L = math.hypot(dx, dy) or 1
            u = (dx / L, -dy / L)
            req = i in var_of
            f = font(note_px - k, bold=req)
            s = str(pos)
            tw, th = text_size(s, f)
            if req:
                rr = 0.47 * bond_px
                cxp, cyp = P[i][0] - u[0] * rr, P[i][1] - u[1] * rr
            else:
                r = 4 * k + _box_radius((tw / 2, th / 3), u)
                cxp, cyp = P[i][0] + u[0] * r, P[i][1] + u[1] * r
            draw.text(
                (cxp, cyp),
                s,
                font=f,
                fill=C["accent"] if req else C["scaffold_num"],
                anchor="mm",
            )
            obstacles.append((cxp - tw / 2, cyp - th / 2, cxp + tw / 2, cyp + th / 2))

    used_numbers = set()

    def place(i, box: NoteBox, prefer):
        w, h = box.measure()
        dirs = [
            (P[n.GetIdx()][0] - P[i][0], P[n.GetIdx()][1] - P[i][1])
            for n in mol.GetAtomWithIdx(i).GetNeighbors()
        ]
        best = None
        for rank, ang in enumerate(prefer):
            u = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
            clear = min((_angle_between(u, d) for d in dirs), default=math.pi)
            r = _box_radius(half[i], u) + 3 * k + _box_radius((w / 2, h / 2), u)
            cx, cy = P[i][0] + u[0] * r, P[i][1] + u[1] * r
            rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            ov = sum(_overlap(rect, ob) for ob in obstacles) / (k * k)
            hits = sum(
                1
                for p in avoid
                if rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]
            )
            score = ov * 12 + hits * 40 - clear * 30 + rank * 4
            if best is None or score < best[0]:
                best = (score, rect)
        rect = best[1]
        obstacles.append(rect)
        box.draw(draw, rect[0], rect[1])

    # charges as superscripts at the freer upper corner of the label
    for v, i in idx.items():
        q = charge_text(pat.atoms[v])
        if not q:
            continue
        f = font(int(font_px * 0.78))
        tw, th = f.getlength(q), font_px * 0.78
        hw, hh = half[i]
        best = None
        for sx in (1, -1):
            cx = P[i][0] + sx * (hw + tw / 2)
            cy = P[i][1] - hh * 0.75
            rect = (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2)
            hits = sum(
                1
                for p in avoid
                if rect[0] - 2 * k <= p[0] <= rect[2] + 2 * k
                and rect[1] <= p[1] <= rect[3]
            )
            if best is None or hits < best[0]:
                best = (hits, cx, cy, rect)
        _, cx, cy, rect = best
        z = rw.GetAtomWithIdx(i).GetAtomicNum()
        col = tuple(int(c * 255) for c in _ATOM_RGB.get(z, (0.0, 0.0, 0.0)))
        draw.text((cx, cy), q, font=f, fill=col, anchor="mm")
        obstacles.append(rect)

    badge_boxes, boxes = [], []
    for v, i in idx.items():
        spec = pat.atoms[v]
        generic = spec.eff_elements() is None or len(spec.eff_elements()) != 1
        if spec.role == "anchor":
            notes = list(neg_notes.get(v, []))
        else:
            notes = atom_notes(spec, generic, use_scaffold, neg_notes.get(v))
        badge = numbers.get(v)
        if badge is not None:
            used_numbers.add(v)
            badge_boxes.append((i, wrap_notes([], badge, note_px, C["note"])))
        box = wrap_notes(notes, None, note_px, C["note"])
        if box:
            boxes.append((-box.measure()[0], i, box))
    for i, box in badge_boxes:
        place(i, box, [315, 225, 45, 135, 270, 0, 180, 90])
    for _, i, box in sorted(boxes, key=lambda t: t[0]):
        place(i, box, [90, 270, 0, 180, 45, 135, 315, 225, 60, 120, 300, 240])

    for key, b in pat.bonds.items():
        if b.stereo:
            u, v = tuple(key)
            p, q = P[idx[u]], P[idx[v]]
            s = "/".join(sorted(b.stereo))
            f = font(note_px, italic=True)
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            L = math.dist(p, q) or 1
            draw.text(
                (mx - (q[1] - p[1]) / L * 12 * k, my + (q[0] - p[0]) / L * 12 * k),
                s,
                font=f,
                fill=C["note"],
                anchor="mm",
            )

    img = crop_to_content(img, style.bg, margin=8 * k)
    img = img.resize((max(1, img.width // k), max(1, img.height // k)), Image.LANCZOS)
    return Drawn(
        img,
        used_numbers,
        use_scaffold,
        flags["any_bond"],
        flags["generic"],
        flags["aromatic"],
        flags["explicit_h"],
    )


def crop_to_content(img: Image.Image, bg, margin: int = 8) -> Image.Image:
    diff = (
        ImageChops.difference(img, Image.new("RGB", img.size, bg))
        .convert("L")
        .point(lambda x: 255 if x > 12 else 0)
    )
    box = diff.getbbox()
    if not box:
        return img
    left, t, r, b = box
    return img.crop(
        (
            max(0, left - margin),
            max(0, t - margin),
            min(img.width, r + margin),
            min(img.height, b + margin),
        )
    )
