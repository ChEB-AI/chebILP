"""One- or two-sentence explanations in plain chemical language."""

from __future__ import annotations

import re

from chebILP.explainability import phrases
from chebILP.explainability.global_model import MOL_TEXT, ClauseModel, FigureModel
from chebILP.explainability.local_model import Local
from chebILP.explainability.pattern import AtomSpec, distinct_pairs
from chebILP.explainability.pattern_draw import element_symbol

MAX_WORDS = 45
_SINGLE_ATOM = re.compile(r"[A-Z][a-z]?[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]*")


def is_plural(name: str) -> bool:
    n = name.lower()
    return n.endswith("s") and not n.endswith(("ss", "us", "is"))


def plural(name: str) -> str:
    """ChEBI class names are singular or plural; this gives the plural."""
    if is_plural(name):
        return name
    if name.endswith("y") and name[-2:-1] not in ("a", "e", "o", "u"):
        return name[:-1] + "ies"
    if name.endswith(("ch", "sh", "x", "ss", "us")):
        return name + "es"
    return name + "s"


def scope(parents: list[str]) -> str:
    """'Among piperidines that are also organooxygen compounds, ' or ''."""
    if not parents:
        return ""
    ps = [plural(p) for p in parents]
    head = ps[0] + (" that are also " + phrases.join_and(ps[1:]) if len(ps) > 1 else "")
    return f"Among {head}, "


# ── global ─────────────────────────────────────────────────────────────────


def atom_noun(spec: AtomSpec) -> str:
    """'an oxygen atom with at least one hydrogen'-style phrase for one pattern atom."""
    els = spec.eff_elements()
    pre, post = [], []
    if spec.cip:
        pre.append(f"({spec.cip})-configured")
    if spec.charge is None and spec.charge_sign in ("+", "-"):
        pre.append(
            "positively charged" if spec.charge_sign == "+" else "negatively charged"
        )
    if els is not None and len(els) == 1 and next(iter(els)) == "h":
        base = "hydrogen atom drawn in the structure"
    elif els is not None and els:
        names = sorted(
            (phrases.ELEMENT_NAME.get(e, phrases.sym(e)) for e in els),
            key=lambda n: (n != "carbon", n),
        )
        base = phrases.join_and(names, "or") + " atom"
    else:
        base = "atom"
    if spec.not_elements and els is None:
        post.append(
            "other than "
            + phrases.join_and(
                [phrases.sym(e) for e in sorted(spec.not_elements)], "or"
            )
        )
    if spec.charge not in (None, 0):
        post.append(f"with charge {spec.charge:+d}".replace("-", "−"))
    elif spec.charge == 0 or spec.charge_sign == "0":
        pre.append("uncharged")
    if spec.h_exact == 0:
        post.append("without hydrogens")
    elif spec.h_exact:
        post.append(
            f"with {phrases.num(spec.h_exact)} hydrogen{'s' if spec.h_exact > 1 else ''}"
        )
    elif spec.h_min:
        post.append(
            f"with at least {phrases.num(spec.h_min)} hydrogen{'s' if spec.h_min > 1 else ''}"
        )
    if spec.steroid is not None:
        post.append(f"at position {spec.steroid} of a steroid core")
    elif spec.rings:
        sizes = sorted(spec.rings)
        post.append(
            "in a "
            + phrases.join_and([f"{phrases._RING.get(r, r)}-membered" for r in sizes])
            + " ring"
        )
    elif spec.in_ring:
        post.append("in a ring")
    elif spec.not_in_ring:
        post.append("outside any ring")
    words = " ".join(pre + [base])
    return f"{phrases.article(words)} {words}" + (
        " " + phrases.join_and(post) if post else ""
    )


def _formula(pat) -> str:
    """Formula string of a small connected pattern, e.g. ``O=*–*``."""
    order = list(pat.atoms)
    edges = {k: _BOND[b.type] for k, b in pat.bonds.items() if len(k) == 2}

    def tok(v):
        return _token(pat.atoms[v])

    starts = phrases.chain_start(order, edges, prefer=lambda v: tok(v) != "*")
    s, _ = phrases.render_graph(order, edges, tok, starts=starts)
    return s


def _token(spec: AtomSpec) -> str:
    sym, _ = element_symbol(spec)
    if "/" in sym:
        sym = "*"
    return sym + (
        phrases.h_suffix(spec.h_exact) if spec.h_exact and sym not in ("*", "H") else ""
    )


def _steroid_phrase(pat) -> str:
    """'a steroid core with =O at position 17'."""
    core_atoms = sorted(
        (v for v, s in pat.atoms.items() if s.steroid is not None),
        key=lambda v: pat.atoms[v].steroid,
    )
    subs = []
    for v in core_atoms:
        s = pat.atoms[v]
        for w in pat.nbrs(v):
            if w in core_atoms:
                continue
            comp, todo = [w], [w]
            while todo:
                x = todo.pop()
                for y in pat.nbrs(x):
                    if y not in comp and y not in core_atoms:
                        comp.append(y)
                        todo.append(y)
            edges = {
                k: _BOND[b.type]
                for k, b in pat.bonds.items()
                if k <= set(comp) and len(k) == 2
            }
            f, _ = phrases.render_graph(
                comp, edges, lambda x: _token(pat.atoms[x]), starts=[w]
            )
            btype = pat.bond(v, w).type
            cip = f" ({s.cip}-configured)" if s.cip else ""
            if f == "*":
                kind = {"double": "double-bonded ", "triple": "triple-bonded "}.get(
                    btype, ""
                )
                subs.append(f"a {kind}substituent at position {s.steroid}{cip}")
            else:
                prefix = {"double": "=", "triple": "≡"}.get(btype, "")
                subs.append(f"{prefix}{f} at position {s.steroid}{cip}")
    return (
        "a steroid core with " + phrases.join_and(subs)
        if subs
        else "a steroid core with the atoms shown"
    )


_BOND = {"single": "–", "double": "=", "triple": "≡", "aromatic": ":", "any": "~"}


def pattern_phrase(cm: ClauseModel) -> str | None:
    """What the drawn pattern of one variant asks for, as an object of 'contain'."""
    pat = cm.pattern
    if pat is None or not pat.atoms:
        return None
    specs = list(pat.atoms.values())
    if any(s.steroid is not None for s in specs):
        return _steroid_phrase(pat)
    if not pat.bonds:
        if len(specs) == 1:
            return atom_noun(specs[0])
        nouns = [atom_noun(s) for s in specs]
        dist = distinct_pairs(pat)
        all_distinct = all(
            frozenset((u, v)) in dist
            for i, u in enumerate(pat.atoms)
            for v in list(pat.atoms)[i + 1 :]
        )
        if len(set(nouns)) == 1 and all_distinct:
            n, noun = len(specs), nouns[0].split(" ", 1)[1]
            return f"at least {phrases.num(n)} {noun.replace(' atom', ' atoms', 1)}"
        if len(specs) <= 3:
            return phrases.join_and(nouns)
        return "the atoms shown"
    if len(pat.components()) > 1:
        return "the groups shown"
    f = _formula(pat)
    return (
        f"the group {f}" if len(pat.atoms) <= 6 and len(f) <= 16 else "the group shown"
    )


def _chip_phrase(chip) -> str | None:
    t = chip.text
    inv = {v: k for k, v in MOL_TEXT.items()}
    if t in inv or t.startswith("not: "):
        neg = t.startswith("not: ")
        pred = inv.get(t[5:] if neg else t)
        base = {
            "aliphatic": "have no aromatic atom",
            "aromatic": "have an aromatic atom",
            "net_charge_positive": "have a positive net charge",
            "net_charge_negative": "have a negative net charge",
            "net_charge_neutral": "have no net charge",
        }.get(pred)
        if base is None:
            return None
        if neg:
            base = {
                "have no aromatic atom": "have an aromatic atom",
                "have no net charge": "have a net charge",
            }.get(base, base.replace("have a", "do not have a"))
        return base
    if t.startswith("every atom is one of: "):
        els = t[len("every atom is one of: ") :].split(", ")
        if len(els) > 6:
            return f"contain no atoms other than the {len(els)} elements listed"
        return "consist only of " + phrases.join_and(els, "and") + " atoms"
    if chip.kind == "count":
        return "have " + t.replace(" in the molecule", "")
    if t.startswith("†"):
        marker = t.split(" ", 1)[0]
        return f"meet the condition marked {marker}"
    return None


def clause_gist(cm: ClauseModel) -> str:
    """A verb phrase for one variant: 'have no net charge and contain an oxygen atom with ...'."""
    if cm.failed:
        return "match the pattern shown"
    parts = [p for p in (_chip_phrase(c) for c in cm.chips) if p]
    for m in cm.minis:
        parts.append(
            "have "
            + m.text.rstrip(":")
            .replace(" like the highlighted ones", " like the ones shown")
            .replace(" like the highlighted one", " like the one shown")
        )
    obj = pattern_phrase(cm)
    if obj:
        parts.append(("have " if obj.startswith("a steroid") else "contain ") + obj)
    gist = (
        phrases.join_and(_drop_repeated_verbs(parts))
        if parts
        else "match the pattern shown"
    )
    kinds = {g.kind for g in cm.groups}
    if "or" in kinds:
        gist += " and one of the parts in the blue panel"
    if "neg" in kinds:
        gist += ", but none of the parts in the red panel"
    return gist


def _merge_single_atoms(gists: list[str]) -> list[str]:
    """'contain a chlorine atom', 'contain a bromine atom' -> 'contain a chlorine or bromine atom'."""
    simple = [
        g
        for g in gists
        if g.startswith("contain a") and g.endswith(" atom") and len(g.split()) == 4
    ]
    if len(simple) < 2:
        return gists
    names = [g.split()[2] for g in simple]
    merged = f"contain {phrases.article(names[0])} {phrases.join_and(names, 'or')} atom"
    out, done = [], False
    for g in gists:
        if g in simple:
            if not done:
                out.append(merged)
                done = True
        else:
            out.append(g)
    return out


def _drop_repeated_verbs(parts: list[str]) -> list[str]:
    """'have X', 'have Y' -> 'have X', 'Y' (read as 'have X and Y')."""
    out = []
    for k, p in enumerate(parts):
        verb = p.split(" ", 1)[0]
        prev = parts[k - 1].split(" ", 1)[0] if k else None
        out.append(
            p.split(" ", 1)[1]
            if verb == prev and verb in ("have", "contain") and " " in p
            else p
        )
    return out


def _join_variants(gists: list[str]) -> str:
    if len(gists) == 1:
        return gists[0]
    nested = any(" and " in g or "," in g for g in gists)
    if len(gists) == 2:
        return (
            f"either {gists[0]}, or {gists[1]}"
            if nested
            else f"{gists[0]} or {gists[1]}"
        )
    return (
        ("; " if nested else ", ").join(gists[:-1])
        + ("; or " if nested else ", or ")
        + gists[-1]
    )


def global_text(fm: FigureModel, class_name: str | None, parents: list[str]) -> str:
    gists = _merge_single_atoms([clause_gist(c) for c in fm.clauses])
    who = (
        f"{plural(class_name)} are those that"
        if class_name
        else "the rule accepts molecules that"
    )
    lead = scope(parents) + who
    text = f"{lead} {_join_variants(gists)}."
    n = len(fm.clauses)
    if len(text.split()) > MAX_WORDS:
        if n == 1:
            text = f"{lead} match the pattern shown."
        else:
            short = min(gists, key=lambda g: len(g.split()))
            text = f"{lead} match one of the {n} patterns shown"
            text += f", for example they {short}." if len(short.split()) <= 12 else "."
    return text[0].upper() + text[1:]


# ── local ──────────────────────────────────────────────────────────────────


def _match_facts(loc: Local) -> list[str]:
    parts = []
    pieces = [p for f in loc.matched if f for p in f.split(" and ")]
    atoms = [p for p in pieces if _SINGLE_ATOM.fullmatch(p)]
    groups = [p for p in pieces if p not in atoms]
    if pieces and len(pieces) <= 3 and all(len(p) <= 24 for p in pieces):
        things = []
        if groups:
            things.append(
                f"{'group' if len(groups) == 1 else 'groups'} {phrases.join_and(groups)}"
            )
        if atoms and len(set(atoms)) == 1:
            things.append(f"{atoms[0]} atom{'s' if len(atoms) > 1 else ''}")
        elif atoms:
            things.append(f"atoms {phrases.join_and(atoms)}")
        parts.append("contains the highlighted " + " and ".join(things))
    elif pieces:
        parts.append(
            "contains the highlighted " + ("group" if len(pieces) == 1 else "groups")
        )
    for v, noun in loc.counts:
        parts.append(f"has {v} {phrases.plural(noun, v)}")
    parts += [p for p in loc.mol_facts if p]
    return parts


def local_text(loc: Local, class_name: str | None, parents: list[str]) -> str:
    belongs = (
        f"belongs to the {plural(class_name)}" if class_name else "satisfies the rule"
    )
    if loc.fires:
        facts = _match_facts(loc) or ["meets all conditions"]
        text = (
            scope(parents)
            + f"this molecule {belongs} because it {phrases.join_and(facts)}."
        )
        return text[0].upper() + text[1:]
    neg = (
        f"does not belong to the {plural(class_name)}"
        if class_name
        else "does not satisfy the rule"
    )
    if not loc.missing:
        return f"This molecule {neg}."
    matched = any("match" in c for c in loc.atom_cols.values())
    where = (
        "in the closest match, "
        if matched and not loc.missing[0].startswith(("it ", "there "))
        else ""
    )
    text = f"This molecule {neg}: {where}{phrases.join_and(loc.missing[:2])}."
    if loc.elsewhere:
        text += " " + loc.elsewhere[0]
    return text
