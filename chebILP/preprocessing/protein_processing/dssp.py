"""
Secondary structure of a protein domain from ``mkdssp`` (Kabsch & Sander 1983, DSSP 4).

``mkdssp`` must be on ``PATH`` (``MKDSSP`` overrides the name). Only alpha helices (DSSP ``H``)
and beta strands (``E``) become secondary-structure elements (SSEs); 3-10 and pi helices,
isolated bridges, turns and bends are coil. Strands are paired through the bridge partners
DSSP reports, with the direction taken from the case of the ladder label (lower case parallel,
upper case antiparallel). SSE pairs with packed Cα atoms are marked as contacts.
"""

import os
import subprocess
import tempfile
from collections.abc import Iterable

import numpy as np

MKDSSP = os.environ.get("MKDSSP", "mkdssp")
MIN_HELIX_LENGTH = 4
MIN_STRAND_LENGTH = 2
CONTACT_DISTANCE = 8.0  # Å between Cα atoms
MIN_CONTACT_PAIRS = 3
MKDSSP_TIMEOUT = 120  # seconds per domain

# Fixed columns of the classic DSSP format.
_COL_AA = 13
_COL_SS = 16
_COL_LABEL = (23, 24)
_COL_BP = ((25, 29), (29, 33))
_COL_CA = ((115, 122), (122, 129), (129, 136))


def run_mkdssp(pdb_text: str) -> str:
    """The classic-format DSSP output for PDB-format text; raises ``RuntimeError`` if ``mkdssp`` fails."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "in.pdb"), os.path.join(tmp, "out.dssp")
        with open(src, "w") as f:
            f.write(pdb_text)
        result = subprocess.run(
            [MKDSSP, "--output-format", "dssp", src, dst],
            capture_output=True,
            text=True,
            timeout=MKDSSP_TIMEOUT,
        )
        if result.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError(result.stderr.strip() or "mkdssp failed")
        with open(dst) as f:
            return f.read()


def parse_dssp(text: str) -> tuple[list[dict], list[bool]]:
    """Per-residue records of a DSSP file, and ``cont`` (``cont[i]``: residue i+1 follows i without a chain break).

    A record holds ``aa`` (one letter), ``ss``, ``ca`` and ``bridges``, a list of
    ``(partner residue index, "P" | "A")`` with indices into the returned list.
    """
    lines = text.splitlines()
    start = (
        next(i for i, line in enumerate(lines) if line.startswith("  #  RESIDUE")) + 1
    )
    residues, cont, index_of, raw_bridges = [], [], {}, []
    broken = False
    for line in lines[start:]:
        if len(line) < _COL_CA[-1][1] or line[_COL_AA] == "!":
            broken = True
            continue
        if residues:
            cont.append(not broken)
        broken = False
        index_of[int(line[:5])] = len(residues)
        aa = line[_COL_AA]
        residues.append(
            {
                "aa": "C" if aa.islower() else aa,
                "ss": line[_COL_SS] if line[_COL_SS] in "HE" else "-",
                "ca": [float(line[a:b]) for a, b in _COL_CA],
            }
        )
        raw_bridges.append(
            [
                (int(line[a:b]), line[label])
                for (a, b), label in zip(_COL_BP, _COL_LABEL)
                if int(line[a:b]) > 0
            ]
        )
    for res, raw in zip(residues, raw_bridges):
        res["bridges"] = [
            (index_of[p], "P" if label.islower() else "A")
            for p, label in raw
            if p in index_of
        ]
    return residues, cont


def segment(ss: list[str], cont: list[bool]) -> list[tuple[str, int, int]]:
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


def domain_structure(pdb_lines: Iterable[str]) -> dict | None:
    """SSEs, strand pairs, sheets and contacts of the domain in the PDB-format ``pdb_lines``; ``None`` without any SSE.

    SSEs are ``{"type": "H"|"E", "start", "end", "length", "gly", "pro"}`` in sequence order;
    pairs, sheets and contacts refer to SSE indices.
    """
    residues, cont = parse_dssp(run_mkdssp("\n".join(pdb_lines) + "\n"))
    if len(residues) < 5:
        return None
    ss = [r["ss"] for r in residues]
    runs = segment(ss, cont)
    if not runs:
        return None
    seq = "".join(r["aa"] for r in residues)
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
                "gly": "G" in seq[s : e + 1],
                "pro": "P" in seq[s : e + 1],
            }
        )

    votes: dict[tuple[int, int], dict[str, int]] = {}
    for i, res in enumerate(residues):
        for j, kind in res["bridges"]:
            if i > j:
                continue
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

    ca = np.array([r["ca"] for r in residues])
    contacts = []
    for a in range(len(sses)):
        for b in range(a + 2, len(sses)):
            xa = ca[sses[a]["start"] : sses[a]["end"] + 1]
            xb = ca[sses[b]["start"] : sses[b]["end"] + 1]
            close = np.linalg.norm(xa[:, None] - xb[None], axis=2) < CONTACT_DISTANCE
            if close.sum() >= MIN_CONTACT_PAIRS:
                contacts.append([a, b])

    return {
        "n_residues": len(residues),
        "sequence": seq,
        "ss": "".join(ss),
        "sses": sses,
        "pairs": pairs,
        "sheets": sheets,
        "contacts": contacts,
    }
