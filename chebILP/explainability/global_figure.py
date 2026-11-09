"""Composing a global FigureModel into one picture."""

from __future__ import annotations

from PIL import Image

from chebILP.explainability.global_model import Chip, ClauseModel, FigureModel
from chebILP.explainability.style import (
    C,
    MAX_W,
    box,
    compose,
    fit,
    flow,
    header,
    hstack,
    legend,
    pill,
    rich_text,
    vstack,
)

CHIP_STYLE = {
    "plain": (C["chip_bg"], C["chip_line"], C["text"]),
    "neg": (C["neg_bg"], C["neg_line"], C["neg"]),
    "count": (C["count_bg"], C["count_line"], C["count_text"]),
    "aux": ((243, 240, 255), (200, 190, 240), (70, 50, 140)),
}


def chip_image(chip: Chip, max_w: int) -> Image.Image:
    bg, line, fg = CHIP_STYLE.get(chip.kind, CHIP_STYLE["plain"])
    return box(
        rich_text(chip.text, max_w - 20, 13, fg, bg=bg), bg, line, pad=(9, 5), radius=10
    )


def group_image(g, max_w: int) -> Image.Image:
    neg = g.kind == "neg"
    bg = C["neg_bg"] if neg else C["or_bg"]
    line = C["neg_line"] if neg else C["or_line"]
    fg = C["neg"] if neg else C["or"]
    title = rich_text(g.title, max_w - 30, 13, fg, bold=True, bg=bg)
    cells = []
    for a in g.alts:
        parts = []
        if a.drawn is not None:
            parts.append(a.drawn.image)
        if a.text:
            parts.append(rich_text(a.text, max(160, max_w // 2), 12, C["text"], bg=bg))
        for fn in a.footnotes:
            parts.append(
                rich_text(
                    fn,
                    max(160, (a.drawn.image.width if a.drawn else 200)),
                    11,
                    C["muted"],
                    bg=bg,
                )
            )
        cells.append(vstack(parts, 4, "center", bg=bg))
    inner_w = max_w - 30
    sep = rich_text("or", 40, 12, fg, bold=True, bg=bg)
    row = []
    for i, c in enumerate(cells):
        if i:
            row.append(sep)
        row.append(c)
    if sum(c.width for c in row) + 10 * (len(row) - 1) <= inner_w:
        body = hstack(row, 10, bg=bg)
    else:
        body = vstack(
            [fit(c, inner_w) if c is not sep else sep for c in row], 6, "center", bg=bg
        )
    content = vstack([title, body], 6, bg=bg)
    if g.description:
        content = vstack(
            [
                content,
                rich_text(
                    f"({g.description})", inner_w, 11, C["muted"], italic=True, bg=bg
                ),
            ],
            4,
            bg=bg,
        )
    return box(content, bg, line, pad=(10, 8), radius=8)


def clause_image(cm: ClauseModel, max_w: int) -> Image.Image:
    inner = max_w - 24
    parts = []
    chips = [chip_image(c, inner) for c in cm.chips]
    if chips:
        parts.append(flow(chips, inner))
    for m in cm.minis:
        t = rich_text(m.text, inner - 30, 13, C["count_text"], bg=C["count_bg"])
        parts.append(
            box(
                vstack(
                    [t, fit(m.drawn.image, inner - 30)], 4, "center", bg=C["count_bg"]
                ),
                C["count_bg"],
                C["count_line"],
                pad=(10, 8),
            )
        )
    if cm.drawn is not None:
        parts.append(fit(cm.drawn.image, inner))
    for g in cm.groups:
        parts.append(group_image(g, inner))
    content_w = max([p.width for p in parts] + [200])
    for fn in cm.footnotes:
        parts.append(rich_text(fn, min(inner, max(content_w, 260)), 12, C["muted"]))
    if not parts:
        parts.append(rich_text("(no condition)", inner, 13, C["muted"]))
    return box(
        vstack(parts, 10, "center"),
        (255, 255, 255),
        C["panel_line"],
        pad=(12, 10),
        radius=10,
    )


def arrange_clauses(panels: list[Image.Image], max_w: int) -> Image.Image:
    if len(panels) == 1:
        return panels[0]
    p = pill("OR")
    if len(panels) == 2 and panels[0].width + panels[1].width + p.width + 24 <= max_w:
        return hstack([panels[0], p, panels[1]], 12)
    cols = 1 if len(panels) == 2 else 2 if len(panels) <= 4 else 3
    cell_w = (max_w - (cols - 1) * (p.width + 16)) // cols
    rows = [panels[i : i + cols] for i in range(0, len(panels), cols)]
    out = []
    for k, r in enumerate(rows):
        cells = []
        for j, c in enumerate(r):
            if j:
                cells.append(p)
            cells.append(fit(c, cell_w))
        if k:
            out.append(p)
        out.append(hstack(cells, 8, valign="center"))
    return vstack(out, 6, "center")


def figure_image(fm: FigureModel, title: str, tag: str, text: str) -> Image.Image:
    max_w = MAX_W - 32
    n = len(fm.clauses)
    if n == 1:
        panels = [clause_image(fm.clauses[0], max_w)]
    elif n == 2:
        side = [clause_image(c, (max_w - 70) // 2) for c in fm.clauses]
        panels = (
            side
            if side[0].width + side[1].width + 70 <= max_w
            else [clause_image(c, max_w) for c in fm.clauses]
        )
    else:
        cols = 2 if n <= 4 else 3
        panels = [
            clause_image(c, (max_w - (cols - 1) * 60) // cols) for c in fm.clauses
        ]
    body = arrange_clauses(panels, max_w)
    width = max(min(max_w, body.width), 560)
    return compose(
        header(title, tag, text, width), body, legend(fm.legend, width), width
    )
