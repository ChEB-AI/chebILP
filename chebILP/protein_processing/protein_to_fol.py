"""
First-order encoding of protein domains at the level of secondary-structure elements (SSEs).

A domain ``D`` links to its SSEs ``s<D>_<k>`` (numbered in sequence order from 1) through
``has_sse(D,S)``, the counterpart of ``has_atom``. Ids are domain-scoped, so every fact has
exactly one owning example. Symmetric relations are written in both directions.

Popper sees numbers only as bins (``short_sse``, ``short_loop``); the exact lengths and
positions are LLM-only computed facts, like ``ring_size`` for molecules.
"""

import pandas as pd

# Inclusive upper bounds of the short and medium length bins, per SSE type.
LENGTH_BINS = {"H": (7, 15), "E": (3, 6)}
SHORT_LOOP = 4


def sse_id(domain: str, k: int) -> str:
    return f"s{domain}_{k + 1}"


def _length_bin(sse: dict) -> str:
    short, medium = LENGTH_BINS[sse["type"]]
    return "short_sse" if sse["length"] <= short else "medium_sse" if sse["length"] <= medium else "long_sse"


def loop_length(a: dict, b: dict) -> int:
    return b["start"] - a["end"] - 1


def domain_facts(domain: str, structure: dict) -> list[str]:
    """BK facts of one domain."""
    sses = structure["sses"]
    ids = [sse_id(domain, k) for k in range(len(sses))]
    facts = [f"has_sse({domain},{s})." for s in ids]
    for s, sse in zip(ids, sses):
        facts.append(f"{'helix' if sse['type'] == 'H' else 'strand'}({s}).")
        facts.append(f"{_length_bin(sse)}({s}).")
        if sse["gly"]:
            facts.append(f"has_gly({s}).")
        if sse["pro"]:
            facts.append(f"has_pro({s}).")
    facts += [f"first_sse({ids[0]}).", f"last_sse({ids[-1]})."]
    for k in range(len(sses) - 1):
        facts.append(f"next_sse({ids[k]},{ids[k + 1]}).")
        if loop_length(sses[k], sses[k + 1]) <= SHORT_LOOP:
            facts.append(f"short_loop({ids[k]},{ids[k + 1]}).")

    partners: dict[int, int] = {}
    for a, b, kind in structure["pairs"]:
        pred = "parallel" if kind == "P" else "antiparallel"
        facts += [f"{pred}({ids[a]},{ids[b]}).", f"{pred}({ids[b]},{ids[a]})."]
        partners[a] = partners.get(a, 0) + 1
        partners[b] = partners.get(b, 0) + 1
    facts += [f"edge_strand({ids[k]})." for k, n in sorted(partners.items()) if n == 1]
    for sheet in structure["sheets"]:
        facts += [f"same_sheet({ids[a]},{ids[b]})." for a in sheet for b in sheet if a != b]
    for a, b in structure["contacts"]:
        facts += [f"contact({ids[a]},{ids[b]}).", f"contact({ids[b]},{ids[a]})."]
    return facts


def build_background_protein(domains: pd.DataFrame, **_) -> tuple[list[str], list[tuple[str, int]]]:
    """BK facts and their body predicates for ``domains`` (index = domain id, ``mol`` = structure)."""
    lines = [fact for did, structure in domains["mol"].items() for fact in domain_facts(did, structure)]
    arities = {}
    for line in lines:
        name, args = line.split("(", 1)
        arities.setdefault(name, args.count(",") + 1)
    return lines, sorted(arities.items())


def topology_string(structure: dict) -> str:
    """Compact text form of a domain for prompts: ``α1(16) β2(5) ... | β2-β4 par, β4-β6 anti``."""
    sses = structure["sses"]
    label = lambda k: f"{'α' if sses[k]['type'] == 'H' else 'β'}{k + 1}"
    elements = " ".join(f"{label(k)}({sse['length']})" for k, sse in enumerate(sses))
    pairs = ", ".join(f"{label(a)}-{label(b)} {'par' if kind == 'P' else 'anti'}" for a, b, kind in structure["pairs"])
    return f"{structure['n_residues']} res: {elements}" + (f" | pairs: {pairs}" if pairs else "")


def build_computed_facts_protein(domains: pd.DataFrame) -> list[str]:
    """``sse_length(S,N)``, ``sse_index(S,I)``, ``loop_length(S1,S2,N)`` and ``domain_length(D,N)``; LLM-only."""
    lines = []
    for did, structure in domains["mol"].items():
        sses = structure["sses"]
        lines.append(f"domain_length({did},{structure['n_residues']}).")
        for k, sse in enumerate(sses):
            s = sse_id(did, k)
            lines += [f"sse_length({s},{sse['length']}).", f"sse_index({s},{k + 1})."]
            if k + 1 < len(sses):
                lines.append(f"loop_length({s},{sse_id(did, k + 1)},{loop_length(sse, sses[k + 1])}).")
    return lines
