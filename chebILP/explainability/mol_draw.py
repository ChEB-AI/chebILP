"""Molecule picture with coloured highlights and short boxed atom notes."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from chebILP.explainability.style import HIGHLIGHT
from chebILP.explainability.style import font as style_font


def prepare(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        m.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(m)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(m)
    return m


def draw_molecule(
    mol, atom_cols: dict, bond_cols: dict, notes: dict, max_w=1500, max_h=1050
) -> Image.Image:
    """``atom_cols``/``bond_cols``: index -> list of colour names; bonds keyed by frozenset({i, j})."""
    m = prepare(mol)
    conf = m.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(m.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(m.GetNumAtoms())]
    unit = min(
        80.0, max(42.0, 42.0 * (25 / max(1, m.GetNumAtoms())) ** 0.5)
    )  # px per bond length
    w = int(min(max_w, max(520, (max(xs) - min(xs)) * unit + 160)))
    h = int(min(max_h, max(360, (max(ys) - min(ys)) * unit + 140)))
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.annotationFontScale = 0.75
    o.fixedBondLength = unit
    o.padding = 0.06
    o.setBackgroundColour((1, 1, 1, 1))
    o.clearBackground = True
    ah = {i: [HIGHLIGHT[c] for c in cs] for i, cs in atom_cols.items() if cs}
    bh = {}
    for key, cs in bond_cols.items():
        i, j = tuple(key)
        b = m.GetBondBetweenAtoms(i, j)
        if b is not None and cs:
            bh[b.GetIdx()] = [HIGHLIGHT[c] for c in cs]
    radii = {i: 0.42 for i in ah}
    try:
        d.DrawMoleculeWithHighlights(m, "", ah, bh, radii, {})
    except Exception:
        d = rdMolDraw2D.MolDraw2DCairo(w, h)
        d.drawOptions().fixedBondLength = unit
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, m, highlightAtoms=list(ah), highlightBonds=list(bh)
        )
    d.FinishDrawing()
    img = Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB")
    pos = {
        i: (d.GetDrawCoords(i).x, d.GetDrawCoords(i).y) for i in range(m.GetNumAtoms())
    }
    place_labels(img, m, pos, notes, atom_cols, unit)
    return img


def place_labels(
    img: Image.Image, m, pos: dict, notes: dict, atom_cols: dict, unit: float = 42.0
):
    """Draw each note in a boxed label next to its atom, on the side away from its bonds."""
    import math

    dr = ImageDraw.Draw(img)
    font = style_font(15)
    seg_pts = []
    for b in m.GetBonds():
        (x1, y1), (x2, y2) = pos[b.GetBeginAtomIdx()], pos[b.GetEndAtomIdx()]
        n = max(2, int(math.hypot(x2 - x1, y2 - y1) / 4))
        seg_pts += [
            (x1 + (x2 - x1) * t / n, y1 + (y2 - y1) * t / n) for t in range(n + 1)
        ]
    atom_pts = list(pos.values())
    # drawn element symbols (with their H) occupy a box around the atom position
    sym_boxes = []
    for a in m.GetAtoms():
        if a.GetSymbol() != "C" or a.GetFormalCharge():
            x, y = pos[a.GetIdx()]
            wx = unit * (
                0.25
                + (0.45 if a.GetTotalNumHs() else 0)
                + (0.15 if a.GetFormalCharge() else 0)
            )
            wy = unit * 0.28 + (unit * 0.15 if a.GetTotalNumHs() else 0)
            sym_boxes.append((x - wx, y - wy, x + wx, y + wy))
    boxes = []
    for i in sorted(notes, key=lambda k: -len(notes[k])):
        text = notes[i]
        if not text:
            continue
        x, y = pos[i]
        tw, th = dr.textbbox((0, 0), text, font=font)[2:]
        vx = vy = 0.0
        for nb in m.GetAtomWithIdx(i).GetNeighbors():
            nx_, ny_ = pos[nb.GetIdx()]
            dlen = math.hypot(nx_ - x, ny_ - y) or 1
            vx += (nx_ - x) / dlen
            vy += (ny_ - y) / dlen
        base = math.atan2(-vy, -vx) if (vx or vy) else -math.pi / 4
        best = None
        angles = [base] + [
            base + s * j * math.pi / 12 for j in range(1, 13) for s in (1, -1)
        ]
        for k, ang in enumerate(angles):
            for r in (0.62 * unit, 0.9 * unit, 1.25 * unit):
                cx, cy = x + r * math.cos(ang), y + r * math.sin(ang)
                bx0 = (
                    cx - tw / 2 - 3
                    if abs(math.cos(ang)) < 0.4
                    else (cx - 3 if math.cos(ang) > 0 else cx - tw - 3)
                )
                by0 = cy - th / 2 - 2
                box = (bx0, by0, bx0 + tw + 6, by0 + th + 5)
                if (
                    box[0] < 2
                    or box[1] < 2
                    or box[2] > img.width - 2
                    or box[3] > img.height - 2
                ):
                    continue
                cost = sum(
                    1
                    for (px, py) in seg_pts
                    if box[0] - 2 <= px <= box[2] + 2 and box[1] - 2 <= py <= box[3] + 2
                )
                cost += 8 * sum(
                    1
                    for (px, py) in atom_pts
                    if box[0] - 8 <= px <= box[2] + 8 and box[1] - 8 <= py <= box[3] + 8
                )
                cost += 100 * sum(
                    1
                    for o in boxes
                    if not (
                        box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3]
                    )
                )
                cost += 12 * sum(
                    1
                    for o in sym_boxes
                    if not (
                        box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3]
                    )
                )
                bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                own = math.hypot(bcx - x, bcy - y)
                if any(
                    math.hypot(bcx - px, bcy - py) < own - 4
                    for j, (px, py) in pos.items()
                    if j != i
                ):
                    cost += 6
                cost += r / (0.62 * unit) + k * 0.05
                if best is None or cost < best[0]:
                    best = (cost, box)
        if best is None:
            continue
        box = best[1]
        boxes.append(box)
        cols = atom_cols.get(i) or ["context"]
        edge = tuple(int(255 * c * 0.75) for c in HIGHLIGHT[cols[0]])
        bx = min(max(x, box[0]), box[2])
        by = min(max(y, box[1]), box[3])
        dist = math.hypot(bx - x, by - y)
        if dist > 18:
            sx, sy = x + (bx - x) * 12 / dist, y + (by - y) * 12 / dist
            dr.line([(sx, sy), (bx, by)], fill=edge, width=1)
        dr.rectangle(box, fill=(255, 255, 255), outline=edge, width=2)
        dr.text((box[0] + 3, box[1] + 1), text, fill=(20, 20, 20), font=font)
