"""Shared look of the explanation figures: fonts, colours, text layout, header and legend."""

from __future__ import annotations

import math
import os
import re

from PIL import Image, ImageDraw, ImageFont


def _font_dir() -> str:
    try:
        import matplotlib

        return os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    except ImportError:
        return "/usr/share/fonts/truetype/dejavu"


FONT_DIR = _font_dir()
FONT = os.path.join(FONT_DIR, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_ITALIC = os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")

WHITE = (255, 255, 255)
C = {
    "text": (31, 41, 51),
    "muted": (97, 110, 124),
    "note": (52, 72, 94),
    "any": (82, 96, 109),
    "scaffold": (160, 168, 178),
    "scaffold_num": (125, 135, 148),
    "neg": (197, 48, 48),
    "neg_bg": (255, 245, 245),
    "neg_line": (230, 160, 160),
    "or": (43, 108, 176),
    "or_bg": (240, 246, 255),
    "or_line": (150, 185, 225),
    "badge": (31, 78, 121),
    "white": WHITE,
    "chip_bg": (237, 242, 247),
    "chip_line": (203, 210, 217),
    "accent": (192, 86, 33),
    "header_bg": (245, 247, 250),
    "count_bg": (255, 248, 230),
    "count_line": (236, 201, 120),
    "count_text": (120, 72, 10),
    "panel_line": (222, 226, 231),
    "ok": (30, 130, 60),
}

# Highlight colours (RGB 0..1), shared by pattern drawings and molecule pictures.
HIGHLIGHT = {
    "match": (1.0, 0.68, 0.38),  # atoms the pattern needs / that match it
    "absent": (1.0, 0.55, 0.55),  # must be absent, or fails in this molecule
    "elsewhere": (
        0.96,
        0.84,
        0.32,
    ),  # fulfils a failed condition elsewhere in the molecule
    "context": (
        0.78,
        0.78,
        0.78,
    ),  # atoms that were inspected but are not part of the match
    "alt": (0.55, 0.72, 1.0),  # one of several alternatives
}
# Further groups in one molecule picture (one per named part of the pattern).
GROUP_COLOURS = ["match", "teal", "purple", "pink", "green"]
HIGHLIGHT.update(
    {
        "teal": (0.35, 0.80, 0.78),
        "purple": (0.78, 0.58, 0.95),
        "pink": (1.0, 0.60, 0.76),
        "green": (0.50, 0.82, 0.50),
    }
)

MAX_W = 900

_fonts: dict = {}


def font(size: int, bold: bool = False, italic: bool = False):
    key = (int(size), bold, italic)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(
            FONT_BOLD if bold else FONT_ITALIC if italic else FONT, int(size)
        )
    return _fonts[key]


def text_size(s: str, f) -> tuple[int, int]:
    box_ = f.getbbox(s or " ")
    asc, desc = f.getmetrics()
    return box_[2] - box_[0], asc + desc


def rgb255(colour) -> tuple[int, int, int]:
    """A HIGHLIGHT key or an RGB 0..1 tuple as 0..255 ints."""
    c = HIGHLIGHT[colour] if isinstance(colour, str) else colour
    return tuple(int(round(255 * x)) for x in c)


# ── text and boxes ─────────────────────────────────────────────────────────


def rich_text(
    text: str,
    max_w: int,
    size: int = 13,
    color=C["text"],
    bold=False,
    italic=False,
    bg=WHITE,
    line_gap: float = 1.35,
) -> Image.Image:
    """Word-wrapped text; ``{n}`` becomes a numbered badge as in the pattern drawings."""
    f = font(size, bold, italic)
    chunks = []  # wrapping happens only at whitespace; a chunk may mix text and badges
    for w in re.split(r"(\s+)", text):
        if not w:
            continue
        toks = []
        for t in re.split(r"(\{\d+\})", w):
            if not t:
                continue
            m = re.fullmatch(r"\{(\d+)\}", t)
            toks.append(
                ("badge", m.group(1), size * 1.25) if m else ("text", t, f.getlength(t))
            )
        chunks.append(toks)
    lines, cur = [[]], 0.0
    for toks in chunks:
        w = sum(t[2] for t in toks)
        blank = len(toks) == 1 and toks[0][0] == "text" and not toks[0][1].strip()
        if cur + w > max_w and lines[-1] and not blank:
            lines.append([])
            cur = 0.0
        if blank and not lines[-1]:
            continue
        lines[-1].extend(toks)
        cur += w
    lh = int(size * line_gap)
    width = max(1, int(math.ceil(max(sum(w for *_, w in ln) for ln in lines))))
    img = Image.new("RGB", (width + 2, lh * len(lines) + 2), bg)
    d = ImageDraw.Draw(img)
    fb = font(max(8, int(size * 0.75)), bold=True)
    y = 0
    for ln in lines:
        x = 0.0
        for kind, val, w in ln:
            if kind == "badge":
                r = size * 0.55
                cy = y + lh / 2
                d.ellipse([x + 1, cy - r, x + 1 + 2 * r, cy + r], fill=C["badge"])
                d.text((x + 1 + r, cy), val, font=fb, fill=WHITE, anchor="mm")
            else:
                d.text((x, y + (lh - size * 1.2) / 2), val, font=f, fill=color)
            x += w
        y += lh
    return img


def box(img: Image.Image, bg, line=None, pad=(8, 6), radius=8, width=1) -> Image.Image:
    W, H = img.width + 2 * pad[0], img.height + 2 * pad[1]
    out = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(out)
    d.rounded_rectangle(
        [0, 0, W - 1, H - 1], radius=radius, fill=bg, outline=line, width=width
    )
    out.paste(img, pad)
    return out


def on_bg(img: Image.Image, bg) -> Image.Image:
    """Recolour a white-background image's white pixels to ``bg`` (for text on tinted panels)."""
    if bg == WHITE:
        return img
    out = Image.new("RGB", img.size, bg)
    mask = img.convert("L").point(lambda x: 0 if x > 250 else 255)
    out.paste(img, (0, 0), mask)
    return out


def vstack(imgs, gap=8, align="left", bg=WHITE) -> Image.Image:
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return Image.new("RGB", (1, 1), bg)
    W = max(i.width for i in imgs)
    H = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (W, H), bg)
    y = 0
    for i in imgs:
        x = (
            0
            if align == "left"
            else (W - i.width) // 2
            if align == "center"
            else W - i.width
        )
        out.paste(i, (x, y))
        y += i.height + gap
    return out


def hstack(imgs, gap=8, valign="center", bg=WHITE) -> Image.Image:
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return Image.new("RGB", (1, 1), bg)
    W = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    H = max(i.height for i in imgs)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for i in imgs:
        y = 0 if valign == "top" else (H - i.height) // 2
        out.paste(i, (x, y))
        x += i.width + gap
    return out


def fit(img: Image.Image, max_w: int) -> Image.Image:
    if img.width <= max_w:
        return img
    s = max_w / img.width
    return img.resize((max_w, max(1, int(img.height * s))), Image.LANCZOS)


def flow(imgs, max_w, gap=6, bg=WHITE) -> Image.Image | None:
    """Left-to-right wrapping row layout."""
    if not imgs:
        return None
    rows, cur, w = [], [], 0
    for i in imgs:
        if cur and w + gap + i.width > max_w:
            rows.append(cur)
            cur, w = [], 0
        cur.append(i)
        w += i.width + (gap if len(cur) > 1 else 0)
    rows.append(cur)
    return vstack([hstack(r, gap, bg=bg) for r in rows], gap, bg=bg)


def pill(text: str, fg=C["or"], bg=(225, 236, 250)) -> Image.Image:
    t = rich_text(text, 200, 12, fg, bold=True, bg=bg)
    return box(t, bg, None, pad=(8, 3), radius=9)


def pad_to(img: Image.Image, W: int, x: int | None = None, bg=WHITE) -> Image.Image:
    """``img`` on a canvas of width ``W``, centred unless ``x`` is given."""
    out = Image.new("RGB", (max(W, img.width), img.height), bg)
    out.paste(img, ((W - img.width) // 2 if x is None else x, 0))
    return out


# ── figure parts ───────────────────────────────────────────────────────────


def header(title: str, tag: str, text: str, width: int) -> Image.Image:
    """Grey band: bold title, a muted tag (e.g. the ChEBI id) and the explanation text."""
    bg = C["header_bg"]
    t = rich_text(title, width - 140, 20, C["text"], bold=True, bg=bg)
    parts = [t]
    if tag:
        g = rich_text(tag, 220, 13, C["muted"], bg=bg)
        row = hstack([t, g], 10, bg=bg)
        parts = [row] if row.width <= width else [t, g]
    if text:
        parts.append(rich_text(text, width, 15, C["text"], bg=bg))
    head = pad_to(vstack(parts, 6, bg=bg), width, 0, bg)
    return box(head, bg, None, pad=(16, 12), radius=0)


def swatch(colour, size: int = 14) -> Image.Image:
    img = Image.new("RGB", (size, size), WHITE)
    ImageDraw.Draw(img).rounded_rectangle(
        [0, 0, size - 1, size - 1],
        radius=3,
        fill=rgb255(colour),
        outline=(150, 150, 150),
    )
    return img


def legend(items: list[tuple], width: int) -> Image.Image | None:
    """Legend rows ``(colour key or None, text)``: a swatch when a colour is given."""
    rows = []
    for colour, text in items:
        t = rich_text(text, width - 24, 12, C["muted"])
        rows.append(hstack([swatch(colour), t], 8, valign="top") if colour else t)
    return vstack(rows, 4) if rows else None


def compose(
    head: Image.Image, body: Image.Image, foot: Image.Image | None, width: int
) -> Image.Image:
    """Header band, centred body and left-aligned legend, with a white margin."""
    W = width + 32
    parts = [pad_to(head, W, 0, C["header_bg"]), pad_to(body, W)]
    if foot is not None:
        parts.append(pad_to(foot, W, 16))
    canvas = vstack(parts, 14)
    out = Image.new("RGB", (canvas.width, canvas.height + 14), WHITE)
    out.paste(canvas, (0, 0))
    return out
