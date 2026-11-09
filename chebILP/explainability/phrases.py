"""Short phrases for BK literals and SMILES-like strings for matched patterns."""

from __future__ import annotations

import re

from rdkit import Chem

from chebILP.explainability.molecule import BOND_PREDS, ELEMENTS

_NUM = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}
_RING = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
BOND_SYM = {
    "bSINGLE": "–",
    "bDOUBLE": "=",
    "bTRIPLE": "≡",
    "bAROMATIC": ":",
    "has_bond_to": "~",
    "bSTEREOE": "=",
    "bSTEREOZ": "=",
    "bSTEREOANY": "=",
}
RD_BOND = {
    Chem.BondType.SINGLE: "–",
    Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: "≡",
    Chem.BondType.AROMATIC: ":",
}
ELEMENT_NAME = {
    "c": "carbon",
    "n": "nitrogen",
    "o": "oxygen",
    "s": "sulfur",
    "p": "phosphorus",
    "h": "hydrogen",
    "f": "fluorine",
    "cl": "chlorine",
    "br": "bromine",
    "i": "iodine",
    "na": "sodium",
    "k": "potassium",
    "se": "selenium",
    "b": "boron",
    "si": "silicon",
    "fe": "iron",
    "mg": "magnesium",
    "ca": "calcium",
}


def num(n: int) -> str:
    return _NUM.get(n, str(n))


def element_of(pred: str) -> str | None:
    return pred if pred in ELEMENTS else None


def sym(el: str) -> str:
    return el[0].upper() + el[1:]


def h_suffix(n: int) -> str:
    return "" if n == 0 else ("H" if n == 1 else "H" + str(n).translate(_SUB))


def charge_suffix(q: int) -> str:
    if q == 0:
        return ""
    s = ("" if abs(q) == 1 else str(abs(q))) + ("+" if q > 0 else "-")
    return s.translate(_SUP)


def charge_of(pred: str) -> int | None:
    m = re.fullmatch(r"charge(\d+)", pred)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"charge_m(\d+)", pred)
    if m:
        return -int(m.group(1))
    return None


def exact_h(pred: str) -> int | None:
    m = re.fullmatch(r"has_(\d+)_hs", pred)
    return int(m.group(1)) if m else None


def unary_phrase(
    pred: str, neg: bool = False, aux_desc: dict | None = None
) -> tuple[str, str]:
    """(phrase, short note) for an atom-level unary literal; phrase completes "<atom> ..."."""
    aux_desc = aux_desc or {}
    m = re.fullmatch(r"in_ring(\d*)", pred)
    if m:
        n = m.group(1)
        if not n:
            return ("is not in a ring", "no ring") if neg else ("is in a ring", "ring")
        size = _RING.get(int(n), n)
        return (
            (f"is not in a {size}-membered ring", f"not in {n}-ring")
            if neg
            else (f"is in a {size}-membered ring", f"{n}-ring")
        )
    if pred.startswith("cip_code_"):
        c = pred[len("cip_code_") :]
        return (
            (f"is not ({c})-configured", f"not ({c})")
            if neg
            else (f"is ({c})-configured", f"({c})")
        )
    m = re.fullmatch(r"steroid_(\d+)", pred)
    if m:
        return (
            (f"is not at steroid position {m.group(1)}", f"not C{m.group(1)}")
            if neg
            else (f"is at steroid position {m.group(1)}", f"C{m.group(1)}")
        )
    n = exact_h(pred)
    if n is not None:
        txt = "no H" if n == 0 else f"exactly {num(n)} H"
        return (
            (f"does not carry {txt}", f"not {n}H")
            if neg
            else (f"carries {txt}", "no H" if n == 0 else f"{n}H")
        )
    m = re.fullmatch(r"has_at_least_(\d+)_hs", pred)
    if m:
        k = int(m.group(1))
        return (
            (f"carries fewer than {num(k)} H", f"<{k}H")
            if neg
            else (f"carries at least {num(k)} H", f"≥{k}H")
        )
    q = charge_of(pred)
    if q is not None:
        s = f"{q:+d}".replace("-", "−")
        return (
            (f"does not have charge {s}", f"not {s}")
            if neg
            else (f"has formal charge {s}", s)
        )
    if pred == "charge0":
        return ("is charged", "charged") if neg else ("has no formal charge", "q 0")
    if pred == "charge_p":
        return (
            ("has no positive charge", "not +")
            if neg
            else ("has a positive charge", "+")
        )
    if pred == "charge_n":
        return (
            ("has no negative charge", "not −")
            if neg
            else ("has a negative charge", "−")
        )
    el = element_of(pred)
    if el:
        name = ELEMENT_NAME.get(el, sym(el))
        return (
            (f"is not {_a(name)} {name} atom", f"not {sym(el)}")
            if neg
            else (f"is {_a(name)} {name} atom", sym(el))
        )
    desc = aux_desc.get(pred)
    name = humanise(pred)
    if desc:
        return (f"is not: {desc}", f"not {name}") if neg else (f"is: {desc}", name)
    return (
        (f"does not satisfy {name}", f"not {name}")
        if neg
        else (f"satisfies {name}", name)
    )


def _a(word: str) -> str:
    return "an" if word[:1].lower() in "aeio" else "a"


def article(word: str) -> str:
    m = re.match(r"\(([A-Za-z])\)", word)
    if m:  # "(R)-configured" is read "R-configured"
        return "an" if m.group(1).upper() in "AEFHILMNORSX" else "a"
    w = word.lower()
    return "an" if w[:1] in "aeiou" and not w.startswith(("one", "uni", "eu")) else "a"


def humanise(pred: str) -> str:
    s = re.sub(r"^(aux|efga|efg|chembl_fg)_", "", pred)
    s = re.sub(r"_\d{3,}$", "", s)
    return s.replace("_", " ")


def join_and(ps: list[str], conj: str = "and") -> str:
    ps = [p for p in ps if p]
    if len(ps) <= 1:
        return ps[0] if ps else ""
    return ", ".join(ps[:-1]) + f" {conj} " + ps[-1]


def plural(noun_phrase: str, n) -> str:
    """``"carbon atoms"`` for any count but one, where it becomes ``"carbon atom"``."""
    if n == 1:
        return noun_phrase.replace(" atoms", " atom", 1).replace(" bonds", " bond", 1)
    return noun_phrase


def bond_word(pred: str) -> str:
    return {
        "bSINGLE": "single",
        "bDOUBLE": "double",
        "bTRIPLE": "triple",
        "bAROMATIC": "aromatic",
    }.get(pred, "a bond")


def as_clause(desc: str) -> str:
    """A predicate description as a lower-case clause without the final full stop."""
    d = (desc or "").strip().rstrip(".")
    if len(d) > 1 and d[0].isupper() and not d[1].isupper():
        d = d[0].lower() + d[1:]
    return d


def mol_phrase(pred: str, neg: bool, net_charge: int) -> str:
    q = f"{net_charge:+d}".replace("-", "−") if net_charge else "0"
    base = {
        "aliphatic": "has no aromatic atom (aliphatic)",
        "aromatic": "has at least one aromatic atom",
        "net_charge_neutral": f"has net charge 0 (here {q})",
        "net_charge_positive": f"has a positive net charge (here {q})",
        "net_charge_negative": f"has a negative net charge (here {q})",
    }.get(pred, f"satisfies {humanise(pred)}")
    return f"not: {base}" if neg else base


def guard_phrase(agg) -> str:
    parts = []
    inv = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}
    words = {
        "=": "exactly",
        "!=": "not exactly",
        "<": "fewer than",
        "<=": "at most",
        ">": "more than",
        ">=": "at least",
    }
    if agg.lguard:  # "t op #agg" == "#agg inv(op) t"
        op, t = agg.lguard
        parts.append(f"{words[inv[op]]} {t}")
    if agg.rguard:
        op, t = agg.rguard
        parts.append(f"{words[op]} {t}")
    return " and ".join(parts)


def guard_holds(agg, value) -> bool | None:
    if value is None:
        return None
    ops = {
        "=": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
    }
    ok = True
    try:
        if agg.lguard:
            op, t = agg.lguard
            ok &= ops[op](float(t), value)
        if agg.rguard:
            op, t = agg.rguard
            ok &= ops[op](value, float(t))
    except ValueError:
        return None
    return ok


def counted_noun(elem, aux_desc: dict) -> str:
    """What one aggregate element counts, e.g. "carbon atoms" or "C=C bonds"."""
    from chebILP.explainability.rule_ir import Lit

    terms = list(elem.terms)
    lits = [c for c in elem.cond if isinstance(c, Lit)]
    by_var: dict[str, list] = {}
    for lit in lits:
        for a in lit.args:
            by_var.setdefault(a, []).append(lit)

    def el_of(v):
        for lit in by_var.get(v, []):
            if len(lit.args) == 1 and not lit.neg and element_of(lit.pred):
                return lit.pred
        return None

    described = [lit for lit in lits if lit.pred in aux_desc and not lit.neg]
    if len(terms) == 2:
        bond = next(
            (
                lit
                for lit in lits
                if lit.pred in BOND_PREDS and set(lit.args) == set(terms)
            ),
            None,
        )
        if bond:
            a, b = el_of(terms[0]), el_of(terms[1])
            return f"{sym(a) if a else '*'}{BOND_SYM[bond.pred]}{sym(b) if b else '*'} bonds"
    if described:
        return f"atoms that are: {aux_desc[described[0].pred]}"
    if len(terms) == 1:
        e = el_of(terms[0])
        extras = [
            unary_phrase(lit.pred, lit.neg)[0]
            for lit in by_var.get(terms[0], [])
            if len(lit.args) == 1 and not element_of(lit.pred)
        ]
        noun = f"{ELEMENT_NAME.get(e, sym(e))} atoms" if e else "atoms"
        return noun + (" that " + " and ".join(extras) if extras else "")
    return "matching parts"


# ── SMILES-like strings ──


def chain_start(nodes: list, edges: dict, prefer=None) -> list:
    """One start node per component: an end of a longest path, preferring ``prefer(u)``."""
    adj = {u: set() for u in nodes}
    for e in edges:
        u, v = tuple(e)
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)

    def bfs(s):
        dist, todo = {s: 0}, [s]
        while todo:
            u = todo.pop(0)
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    todo.append(v)
        return dist

    starts, seen = [], set()
    for u in nodes:
        if u in seen:
            continue
        comp = bfs(u)
        seen |= set(comp)
        ecc = {x: max(bfs(x).values()) for x in comp}
        best = max(
            comp,
            key=lambda x: (
                ecc[x],
                bool(prefer and prefer(x)),
                -len(adj[x]),
                -nodes.index(x),
            ),
        )
        starts.append(best)
    return starts


def render_graph(
    nodes: list, edges: dict, token, root=None, starts=None
) -> tuple[str, list]:
    """DFS string with branches in parentheses and ring-closure digits.

    ``edges``: frozenset({u, v}) -> bond symbol. ``token(u)`` gives the atom text.
    Returns the components joined by " and ", and the node order of appearance.
    """
    adj = {u: [] for u in nodes}
    for e in edges:
        u, v = tuple(e)
        if u in adj and v in adj:
            adj[u].append(v)
            adj[v].append(u)
    order: list = []
    visited: set = set()
    parts = []
    starts = list(starts or []) + ([root] if root in adj else []) + list(nodes)

    def tree_from(r):
        parent, seq, stack = {r: None}, [], [r]
        while stack:
            u = stack.pop()
            if u in seq:
                continue
            seq.append(u)
            for v in sorted(adj[u], key=lambda x: nodes.index(x), reverse=True):
                if v not in parent:
                    parent[v] = u
                    stack.append(v)
        return parent, seq

    for r in starts:
        if r in visited:
            continue
        parent, seq = tree_from(r)
        visited.update(seq)
        children = {u: [v for v in seq if parent.get(v) == u] for u in seq}
        size = {}

        def sz(u):
            if u not in size:
                size[u] = 1 + sum(sz(c) for c in children[u])
            return size[u]

        tree_edges = {frozenset((u, p)) for u, p in parent.items() if p is not None}
        closures = [e for e in edges if e not in tree_edges and set(e) <= set(seq)]
        ring_digit: dict = {}
        for k, e in enumerate(closures, start=1):
            for u in e:
                ring_digit.setdefault(u, []).append((k, e))

        def emit(u):
            order.append(u)
            s = token(u)
            for k, e in ring_digit.get(u, []):
                other = next(iter(set(e) - {u}))
                bs = edges[e] if other in order else ""
                s += (bs if bs not in ("–",) else "") + str(k)
            kids = sorted(children[u], key=sz)
            out = s
            for i, c in enumerate(kids):
                sub = edges[frozenset((u, c))] + emit(c)
                out += f"({sub})" if i < len(kids) - 1 else sub
            return out

        parts.append(emit(r))
    return "` and `".join(parts), order
