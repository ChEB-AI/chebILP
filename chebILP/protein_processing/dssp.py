"""
Secondary structure of a protein domain, after DSSP (Kabsch & Sander 1983), in numpy.

Only the parts the fold encoding needs: backbone H-bonds, alpha helices (two consecutive
4-turns), and beta bridges / ladders with their direction (residues in a ladder of at least two
bridges, bulges included, are strand). 3-10 and pi helices, turns and bends are not assigned.
Residues are then merged into secondary-structure elements (SSEs), strands paired into sheets,
and SSE pairs with packed Cα atoms marked as contacts.
"""

import numpy as np

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
}
BACKBONE = ("N", "CA", "C", "O")
HBOND_ENERGY_CUTOFF = -0.5  # kcal/mol
MIN_HELIX_LENGTH = 4
MIN_STRAND_LENGTH = 2
CONTACT_DISTANCE = 8.0  # Å between Cα atoms
MIN_CONTACT_PAIRS = 3


def read_backbone(path: str) -> tuple[list[str], np.ndarray]:
    """Residue names and backbone coordinates (residues × N/CA/C/O × xyz) of the first model.

    Residues missing a backbone atom are skipped; the first alternate location is kept.
    """
    residues: dict[tuple, dict] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in " A":
                continue
            name = line[12:16].strip()
            if name not in BACKBONE:
                continue
            key = (line[21], line[22:26], line[26])
            res = residues.setdefault(key, {"name": line[17:20].strip()})
            res.setdefault(
                name, [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    complete = [r for r in residues.values() if all(a in r for a in BACKBONE)]
    names = [r["name"] for r in complete]
    coords = np.array(
        [[r[a] for a in BACKBONE] for r in complete], dtype=float
    ).reshape(-1, 4, 3)
    return names, coords


def hbond_matrix(coords: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``hb[i, j]``: the C=O of residue i accepts an H-bond from the N-H of residue j; and the chain-break mask.

    ``cont[i]`` is True when residue i+1 follows residue i without a break (C–N under 2.5 Å).
    """
    n = len(coords)
    N, CA, C, Ox = (coords[:, k] for k in range(4))
    cont = np.linalg.norm(N[1:] - C[:-1], axis=1) < 2.5
    # Amide H: on N, opposite the previous residue's carbonyl, 1 Å out.
    H = N.copy()
    co = C[:-1] - Ox[:-1]
    H[1:] = N[1:] + co / np.linalg.norm(co, axis=1, keepdims=True)
    has_h = np.zeros(n, dtype=bool)
    has_h[1:] = cont & np.array([nm != "PRO" for nm in names[1:]])

    def dist(a, b):
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

    with np.errstate(divide="ignore"):
        energy = 27.888 * (
            1 / dist(Ox, N) + 1 / dist(C, H) - 1 / dist(Ox, H) - 1 / dist(C, N)
        )
    hb = (energy < HBOND_ENERGY_CUTOFF) & has_h[None, :] & (dist(CA, CA) < 9.0)
    idx = np.arange(n)
    hb &= np.abs(idx[:, None] - idx[None, :]) >= 2
    return hb, cont


def _unbroken(cont: np.ndarray, i: int, j: int) -> bool:
    """No chain break between residues ``i`` and ``j`` (inclusive, ``i <= j``)."""
    return bool(cont[i:j].all())


def assign_secondary_structure(coords: np.ndarray, names: list[str]):
    """Per-residue ``H`` / ``E`` / ``-``, the beta ladders, and the chain-break mask.

    A ladder is ``(type, [(i, j), ...])`` with type ``"P"`` (parallel) or ``"A"`` (antiparallel)
    and its bridge pairs, ``i < j``.
    """
    n = len(coords)
    hb, cont = hbond_matrix(coords, names)
    ss = np.array(["-"] * n)

    turn4 = np.array(
        [i + 4 < n and hb[i, i + 4] and _unbroken(cont, i, i + 4) for i in range(n)]
    )
    helix = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if turn4[i - 1] and turn4[i]:
            helix[i : i + 4] = True

    bridges = []
    for i in range(1, n - 1):
        if not _unbroken(cont, i - 1, i + 1):
            continue
        for j in range(i + 3, n - 1):
            if not _unbroken(cont, j - 1, j + 1):
                continue
            if (hb[i - 1, j] and hb[j, i + 1]) or (hb[j - 1, i] and hb[i, j + 1]):
                bridges.append(("P", i, j))
            elif (hb[i, j] and hb[j, i]) or (hb[i - 1, j + 1] and hb[j - 1, i + 1]):
                bridges.append(("A", i, j))

    ladders: list[tuple[str, list[tuple[int, int]]]] = []
    for kind, i, j in sorted(bridges, key=lambda b: (b[1], b[2])):
        prev = (i - 1, j - 1) if kind == "P" else (i - 1, j + 1)
        for lk, pairs in ladders:
            if lk == kind and pairs[-1] == prev:
                pairs.append((i, j))
                break
        else:
            ladders.append((kind, [(i, j)]))

    # Join ladders across beta bulges: same type, gap of at most 1 on one strand and 4 on the other.
    merged = True
    while merged:
        merged = False
        for a in range(len(ladders)):
            for b in range(len(ladders)):
                if a == b or ladders[a][0] != ladders[b][0]:
                    continue
                kind, la, lb = ladders[a][0], ladders[a][1], ladders[b][1]
                (i1, j1), (i2, j2) = la[-1], lb[0]
                di = i2 - i1 - 1
                dj = (j2 - j1 - 1) if kind == "P" else (j1 - j2 - 1)
                if di >= 0 and dj >= 0 and min(di, dj) <= 1 and max(di, dj) <= 4:
                    ladders[a] = (kind, la + lb)
                    del ladders[b]
                    merged = True
                    break
            if merged:
                break

    ladders = [(k, pairs) for k, pairs in ladders if len(pairs) >= 2]
    for _, pairs in ladders:
        ii = [i for i, _ in pairs]
        jj = [j for _, j in pairs]
        ss[min(ii) : max(ii) + 1] = "E"
        ss[min(jj) : max(jj) + 1] = "E"
    ss[helix] = "H"
    return ss, ladders, cont


def segment(ss: np.ndarray, cont: np.ndarray) -> list[tuple[str, int, int]]:
    """Runs of one SS type as ``(type, start, end)`` (inclusive), split at chain breaks, short runs dropped."""
    runs, start = [], 0
    for i in range(1, len(ss) + 1):
        if i == len(ss) or ss[i] != ss[start] or not cont[i - 1]:
            kind, length = ss[start], i - start
            if (kind == "H" and length >= MIN_HELIX_LENGTH) or (
                kind == "E" and length >= MIN_STRAND_LENGTH
            ):
                runs.append((kind, start, i - 1))
            start = i
    return runs


def domain_structure(path: str) -> dict | None:
    """SSEs, strand pairs, sheets and contacts of the domain in ``path``; ``None`` without any SSE.

    SSEs are ``{"type": "H"|"E", "start", "end", "length", "gly", "pro"}`` in sequence order;
    pairs, sheets and contacts refer to SSE indices.
    """
    names, coords = read_backbone(path)
    if len(names) < 5:
        return None
    ss, ladders, cont = assign_secondary_structure(coords, names)
    runs = segment(ss, cont)
    if not runs:
        return None
    sse_of = np.full(len(ss), -1)
    sses = []
    for k, (kind, s, e) in enumerate(runs):
        sse_of[s : e + 1] = k
        sses.append(
            {
                "type": kind,
                "start": s,
                "end": e,
                "length": e - s + 1,
                "gly": "GLY" in names[s : e + 1],
                "pro": "PRO" in names[s : e + 1],
            }
        )

    votes: dict[tuple[int, int], dict[str, int]] = {}
    for kind, pairs in ladders:
        for i, j in pairs:
            a, b = sse_of[i], sse_of[j]
            if (
                a < 0
                or b < 0
                or a == b
                or sses[a]["type"] != "E"
                or sses[b]["type"] != "E"
            ):
                continue
            key = (min(a, b), max(a, b))
            votes.setdefault(key, {"P": 0, "A": 0})[kind] += 1
    pairs = [[int(a), int(b), max(v, key=v.get)] for (a, b), v in sorted(votes.items())]

    parent = list(range(len(sses)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for k, sse in enumerate(sses):
        if sse["type"] == "E":
            groups.setdefault(find(k), []).append(k)
    sheets = sorted(groups.values())

    ca = coords[:, 1]
    contacts = []
    for a in range(len(sses)):
        for b in range(a + 2, len(sses)):
            xa = ca[sses[a]["start"] : sses[a]["end"] + 1]
            xb = ca[sses[b]["start"] : sses[b]["end"] + 1]
            close = np.linalg.norm(xa[:, None] - xb[None], axis=2) < CONTACT_DISTANCE
            if close.sum() >= MIN_CONTACT_PAIRS:
                contacts.append([a, b])

    return {
        "n_residues": len(names),
        "sequence": "".join(THREE_TO_ONE.get(nm, "X") for nm in names),
        "ss": "".join(ss),
        "sses": sses,
        "pairs": pairs,
        "sheets": sheets,
        "contacts": contacts,
    }
