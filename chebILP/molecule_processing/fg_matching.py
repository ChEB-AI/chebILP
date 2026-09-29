"""Functional-group matching helpers.

Provides ``get_chembl_fgs`` and ``get_chebi_fgs`` which replicate the
identically-named methods formerly on ``chemlog.preprocessing.chebi_data.ChEBIData``.
Results are cached to pickle files so expensive SMARTS matching is only
performed once per dataset.
"""

from __future__ import annotations

import os
import pickle
import re
from typing import NamedTuple, TYPE_CHECKING

import tqdm
from rdkit import Chem

if TYPE_CHECKING:
    import pandas as pd


# Greek letters carry meaning in functional-group names (e.g. α,β-unsaturated); spell
# them out so the predicate stays interpretable and ASCII. A raw Greek letter survives
# isalnum() and yields a non-ASCII predicate that the clingo/Popper toolchain rejects.
_GREEK_TRANSLITERATION = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron",
    "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
}


def _sanitize_fg_name(raw_name: str) -> str:
    """Convert a functional-group description to an ASCII, Prolog-safe predicate name."""
    name = raw_name.lower()
    for greek, ascii_name in _GREEK_TRANSLITERATION.items():
        name = name.replace(greek, f"_{ascii_name}_")
    name = name.replace(" ", "_").replace("-", "_")
    name = name.replace("/", "_or_").replace(">", "_more_than_").replace("<", "_less_than_")
    # ASCII-only: a non-ASCII byte in a predicate name breaks clingo/Popper.
    name = "".join(c for c in name if c.isascii() and (c.isalnum() or c == "_"))
    name = name.strip("_")
    if name and not name[0].isalpha():
        name = "fg_" + name
    return name


def get_chembl_fgs(
    processed_df: "pd.DataFrame",
    cache_path: str | None = None,
) -> dict[int, list[str]]:
    """Return a dict mapping molecule (ChEBI) IDs to lists of ChEMBL functional-group names.

    Parameters
    ----------
    processed_df : pd.DataFrame
        Must have a ``mol`` column (RDKit ``Mol`` objects) and be indexed by
        integer ChEBI IDs.
    cache_path : str or None
        If given, results are pickled to / loaded from this path.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.CHEMBL)
    catalog = FilterCatalog.FilterCatalog(params)

    fg_matches_by_mol: dict[int, list[str]] = {}
    for row in tqdm.tqdm(
        processed_df.itertuples(),
        total=len(processed_df),
        desc="Matching ChEMBL FGs",
    ):
        fg_matches_by_mol[row.Index] = []
        matches = catalog.GetMatches(row.mol)
        for match in matches:
            fg_name = _sanitize_fg_name(match.GetDescription())
            fg_matches_by_mol[row.Index].append(fg_name)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(fg_matches_by_mol, fh)

    return fg_matches_by_mol


# Prefixes for seeded functional-group predicates. A seeded ``chembl_fg_<name>`` / ``efg_<name>``
# library predicate is NOT an ASP rule: at grounding time its extension is the RDKit match,
# injected directly as facts (see ``fg_fact_lines`` / the seed-source registry below), never
# derived from a clause body.
CHEMBL_FG_FACT_PREFIX = "chembl_fg_"
EFG_FACT_PREFIX = "efg_"
# Atom-anchored EFGs: one fact per group match, carrying the group's *attachment atoms* instead of
# the molecule -- ``efga_<name>(A1,..,Ak)``, with k fixed per group (see ``_attachment_indices``).
# Each argument joins to the molecule through the existing ``has_atom(M,Ai)``, so a rule can place
# a substituent on a specific attachment atom (``efga_furans(A,B,C,D), bSINGLE(C,O), o(O)``) --
# which the molecule-level ``efg_<name>(M)`` presence fact cannot express.
# ``efga_`` is disjoint from ``efg_`` under ``str.startswith`` (the 4th char is ``a`` vs ``_``).
EFG_ATOM_FACT_PREFIX = "efga_"

DEFAULT_EFG_PATH = os.path.join("data", "efg_functional_groups.tsv")
# One-line natural-language description per EFG group (columns ``Name``, ``Description``,
# ``Source``). EFG itself ships names and SMARTS only; the descriptions are ChEBI / IUPAC Gold Book
# definitions where an exact name match describes the same structure, else written from the SMARTS.
DEFAULT_EFG_DESCRIPTIONS_PATH = os.path.join("data", "efg_descriptions.tsv")
# Cache of :func:`get_efg_atom_fgs`. Renamed when the facts became attachment-atom tuples, so a
# stale single-anchor cache is not read back in the new format.
EFG_ATOM_CACHE = "efg_atom_attach_fgs.pkl"


def _efg_pred_name(raw_name: str, idx: int, seen: set[str]) -> str:
    """The ``efg_<sanitized>`` predicate name for one EFG group (dedup with ``_<idx>``)."""
    import re

    pred = "efg_" + _sanitize_fg_name(raw_name)
    pred = re.sub(r"_+", "_", pred).strip("_")
    if pred in seen:  # keep names unique and stable while staying human-readable
        pred = f"{pred}_{idx}"
    seen.add(pred)
    return pred


def _load_efg_patterns(efg_path: str) -> list[tuple[str, list, list]]:
    """Parse the Extended Functional Groups TSV into (predicate_name, reject, accept).

    The file (vendored from ScoPy, originally Salmina, Haider & Tetko 2016) has
    columns ``Name``, ``SMARTS``, ``Reject``, ``Accept``. The ``SMARTS`` column is
    a human-readable ``... AND NOT ...`` expression that RDKit cannot parse; the
    machine-readable form is the ``Reject`` / ``Accept`` columns, each a Python list
    literal of SMARTS strings. A group matches a molecule iff none of its ``Reject``
    patterns match and all of its ``Accept`` patterns match.
    """
    import ast
    import csv

    patterns: list[tuple[str, list, list]] = []
    seen: set[str] = set()
    with open(efg_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header
        for idx, row in enumerate(reader):
            name, _smarts, reject, accept = row[0], row[1], row[-2], row[-1]
            reject_pats = [Chem.MolFromSmarts(s) for s in ast.literal_eval(reject)] if reject else []
            accept_pats = [Chem.MolFromSmarts(s) for s in ast.literal_eval(accept)] if accept else []
            if any(p is None for p in reject_pats + accept_pats):
                raise ValueError(f"EFG group {idx} ({name!r}) has an unparseable SMARTS pattern")
            patterns.append((_efg_pred_name(name, idx, seen), reject_pats, accept_pats))
    return patterns


def get_efg_fgs(
    processed_df: "pd.DataFrame",
    efg_path: str = DEFAULT_EFG_PATH,
    cache_path: str | None = None,
) -> dict[int, list[str]]:
    """Return a dict mapping molecule (ChEBI) IDs to lists of matched EFG predicate names.

    Extended Functional Groups (EFG): 583 curated functional-group definitions,
    each matched molecule-wide (presence/absence). A group matches iff none of its
    ``Reject`` patterns match and all of its ``Accept`` patterns match, reproducing
    ScoPy's ``CheckPattl``.

    Parameters
    ----------
    processed_df : pd.DataFrame
        Must have a ``mol`` column (RDKit ``Mol`` objects) and be indexed by
        integer ChEBI IDs.
    efg_path : str
        Path to the EFG TSV (columns ``Name``, ``SMARTS``, ``Reject``, ``Accept``).
    cache_path : str or None
        If given, results are pickled to / loaded from this path.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    patterns = _load_efg_patterns(efg_path)

    fg_matches_by_mol: dict[int, list[str]] = {}
    for row in tqdm.tqdm(
        processed_df.itertuples(),
        total=len(processed_df),
        desc="Matching EFG FGs",
    ):
        matched = []
        for pred, reject_pats, accept_pats in patterns:
            if any(row.mol.HasSubstructMatch(p) for p in reject_pats):
                continue
            if all(row.mol.HasSubstructMatch(p) for p in accept_pats):
                matched.append(pred)
        fg_matches_by_mol[row.Index] = matched

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(fg_matches_by_mol, fh)

    return fg_matches_by_mol


def _efg_atom_name(efg_name: str) -> str:
    """Map a molecule-level ``efg_<x>`` predicate name to its atom-anchored ``efga_<x>`` name."""
    return EFG_ATOM_FACT_PREFIX + efg_name[len(EFG_FACT_PREFIX):]


# ---------------------------------------------------------------------------
# Attachment points of an atom-anchored EFG
#
# An ``efga_<name>`` fact names the atoms of one group occurrence at which the rest of the molecule
# can attach -- not the whole match, and not one arbitrary anchor. A five-membered heteroaromatic
# ring is open at each ring atom, so its predicate carries all of them and a rule can hang a
# substituent on a particular one; a carboxylic acid is closed except at its carbonyl carbon, so
# its predicate carries only that carbon.
#
# Which atoms those are is decided statically from the SMARTS, so the arity is a property of the
# group and not of whichever molecules happen to be matched (callers match subsets of the data):
#
# * **Substituent placeholders** are dropped. EFG writes the "R" of a group as a terminal,
#   singly-bonded generic atom -- ``[#1,#6]``, ``[#6&!$(C#N)]``, ``c``, ``[#1,*]``. They are not
#   part of the group, and the core atom they hang off is an attachment point.
# * A core atom is an attachment point when its query leaves room for a neighbour the pattern
#   does not name (``X``/``D``/``H`` counts against the pattern bonds, or the element's usual
#   valence). When the query does not settle it, the atom counts as open.
#
# Single-atom patterns that are ORs of recursive SMARTS (``[$(ring1),$(ring2)]``, 148 groups) are
# expanded into their alternatives so the ring atoms become arguments too; the original query is
# kept as a filter on the alternative's root atom (it may carry extra ``&!$(...)`` conditions).
# Alternatives whose attachment counts differ carry all their core atoms instead, and ones whose
# core sizes differ too fall back to the single root atom.
# ---------------------------------------------------------------------------

_DEFAULT_VALENCE = {5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 3, 16: 2, 17: 1, 34: 2, 35: 1, 53: 1}
# Maximum connections of a neutral-or-charged aromatic atom (a pyridinium n has three).
_AROMATIC_MAX_CONN = {6: 3, 7: 3, 8: 2, 15: 3, 16: 2, 34: 2}
_ANY = "any"
_MAX_MATCHES = 100000
_MAX_ALTERNATIVES = 64
_QUERY_LEAF_RE = re.compile(r"^(\w+)\s+(-?\d+)?\s*(=|!=)\s*val$")


def _parse_query_tree(description: str):
    """``Atom.DescribeQuery()`` text -> nested ``(label, [children])`` tree (indent = depth)."""
    root = ("root", [])
    stack = [(-1, root)]
    for line in description.splitlines():
        if not line.strip():
            continue
        depth = (len(line) - len(line.lstrip(" "))) // 2
        node = (line.strip(), [])
        while stack[-1][0] >= depth:
            stack.pop()
        stack[-1][1][1].append(node)
        stack.append((depth, node))
    return root[1][0] if root[1] else ("AtomNull", [])


def _query_alternatives(node) -> list[dict]:
    """Disjunction of the constraints an atom query places on element / X / D / H / charge.

    Each alternative is a dict over ``elem`` (atomic number, ``"any"`` for a genuine wildcard),
    ``aromatic``, ``X``, ``D``, ``H``, ``charge``. A negated or recursive primitive contributes no
    information, so the result may be looser than the query, never stricter.
    """
    label, children = node
    if label == "AtomOr":
        out = [alt for c in children for alt in _query_alternatives(c)]
        return out[:_MAX_ALTERNATIVES]
    if label == "AtomAnd":
        out = [{}]
        for c in children:
            out = [{**b, **{k: v for k, v in a.items() if k not in b}}
                   for a in out for b in _query_alternatives(c)][:_MAX_ALTERNATIVES]
        return out
    if label == "AtomNull":
        return [{"elem": _ANY}]
    m = _QUERY_LEAF_RE.match(label)
    if not m or m.group(3) != "=" or m.group(2) is None:
        return [{}]
    kind, val = m.group(1), int(m.group(2))
    if kind == "AtomAtomicNum":
        return [{"elem": val}]
    if kind == "AtomType":  # aromatic atoms are encoded as 1000 + atomic number
        return [{"elem": val % 1000, "aromatic": val >= 1000}]
    if kind in ("AtomIsAromatic", "AtomIsAliphatic"):  # ``a`` / ``A``: any element
        return [{"elem": _ANY, "aromatic": kind == "AtomIsAromatic"}]
    key = {"AtomTotalDegree": "X", "AtomExplicitDegree": "D", "AtomHCount": "H",
           "AtomFormalCharge": "charge"}.get(kind)
    return [{key: val}] if key else [{}]


def _alternative_is_open(alt: dict, n_bonds: int, bond_order_sum: float) -> bool:
    """Whether a query alternative leaves room for a neighbour the pattern does not name."""
    if "D" in alt:
        return alt["D"] > n_bonds
    h = alt.get("H", 0)
    if "X" in alt:
        return alt["X"] - n_bonds - h > 0
    elem = alt.get("elem")
    if not isinstance(elem, int):
        return True
    if alt.get("aromatic"):
        max_conn = _AROMATIC_MAX_CONN.get(elem)
        return max_conn is None or max_conn - n_bonds - h > 0
    # With no H count given the atom may carry none, so this only closes an atom whose pattern
    # bonds already use its whole valence (a ring ``O``, a ``=[O]``).
    if alt.get("charge", 0) == 0 and elem in _DEFAULT_VALENCE:
        return _DEFAULT_VALENCE[elem] - bond_order_sum - h > 0
    return True


def _bond_order(bond) -> float:
    return {Chem.BondType.DOUBLE: 2.0, Chem.BondType.TRIPLE: 3.0}.get(bond.GetBondType(), 1.0)


def _attachment_indices(pattern, include_closed: bool = False) -> list[int]:
    """Pattern-atom indices that are the group's attachment points, in pattern order.

    See the section comment above for the rules. A group with no open atom at all (fully
    closed) is anchored on its first core atom so the predicate still has an argument.
    ``include_closed`` returns every core (non-placeholder) atom instead.
    """
    n = pattern.GetNumAtoms()
    alts = [_query_alternatives(_parse_query_tree(a.DescribeQuery())) for a in pattern.GetAtoms()]

    def is_placeholder(atom) -> bool:
        if atom.GetDegree() != 1:
            return False
        if atom.GetBonds()[0].GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            return False
        return all(alt.get("elem") in (1, 6, _ANY) for alt in alts[atom.GetIdx()])

    placeholders = {a.GetIdx() for a in pattern.GetAtoms() if is_placeholder(a)}
    # Two placeholders bonded to each other (``[#6][#6]``) are the whole group, not two R's.
    placeholders = {i for i in placeholders
                    if pattern.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() not in placeholders}
    if len(placeholders) == n:
        placeholders = set()

    attach = []
    for atom in pattern.GetAtoms():
        i = atom.GetIdx()
        if i in placeholders:
            continue
        if include_closed or any(nb.GetIdx() in placeholders for nb in atom.GetNeighbors()):
            attach.append(i)
            continue
        bonds = atom.GetBonds()
        order_sum = sum(_bond_order(b) for b in bonds)
        if any(_alternative_is_open(alt, len(bonds), order_sum) for alt in alts[i]):
            attach.append(i)
    if not attach:
        attach = [min(set(range(n)) - placeholders)]
    return attach


def _split_top_level(text: str, sep: str) -> list[str] | None:
    """Split ``text`` on ``sep`` outside ``()``/``[]``; ``None`` if a low-precedence ``;`` is hit."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0 and ch == ";":
            return None
        if depth == 0 and ch == sep:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _recursive_alternatives(smarts: str) -> list[str] | None:
    """The ``X`` of each ``$(X)`` in a single-atom ``[$(X),$(Y)&..]`` pattern, else ``None``."""
    smarts = smarts.strip()
    if not (smarts.startswith("[") and smarts.endswith("]")):
        return None
    pieces = _split_top_level(smarts[1:-1], ",")
    if not pieces:
        return None
    out = []
    for piece in pieces:
        if not piece.startswith("$("):
            return None
        depth = 0
        for j, ch in enumerate(piece[1:], start=1):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0:
                break
        out.append(piece[2:j])
    return out


class EFGAtomSpec(NamedTuple):
    """How to turn one EFG group's matches into attachment-atom tuples.

    ``patterns`` pairs each SMARTS alternative with the pattern-atom indices emitted as the
    predicate's arguments; ``root_filter`` (for expanded recursive patterns) is the original
    single-atom query an alternative's atom 0 must also match.
    """
    patterns: list[tuple]
    root_filter: object | None
    arity: int


def _efg_atom_spec(accept_pattern) -> EFGAtomSpec:
    """The :class:`EFGAtomSpec` for a group whose first accept pattern is ``accept_pattern``."""
    smarts = Chem.MolToSmarts(accept_pattern)
    if accept_pattern.GetNumAtoms() == 1:
        alternatives = _recursive_alternatives(smarts)
        if alternatives:
            compiled = [Chem.MolFromSmarts(a) for a in alternatives]
            if all(c is not None and c.GetNumAtoms() > 1 for c in compiled):
                specs = [(c, _attachment_indices(c)) for c in compiled]
                if len({len(idx) for _, idx in specs}) != 1:
                    # Alternatives differing only in bond placement (``N1-C=C-1`` vs ``N1=C-C-1``)
                    # close different atoms; keep the ring whole rather than lose it.
                    specs = [(c, _attachment_indices(c, include_closed=True)) for c in compiled]
                if len({len(idx) for _, idx in specs}) == 1:
                    return EFGAtomSpec(specs, accept_pattern, len(specs[0][1]))
        return EFGAtomSpec([(accept_pattern, [0])], None, 1)
    attach = _attachment_indices(accept_pattern)
    return EFGAtomSpec([(accept_pattern, attach)], None, len(attach))


def _efg_atom_tuples(mol, spec: EFGAtomSpec) -> list[tuple[int, ...]]:
    """Distinct attachment-atom tuples of ``spec``'s matches in ``mol``, in match order.

    Every mapping of the pattern is enumerated (``uniquify=False``), so a symmetric group yields
    each equivalent ordering of its attachment atoms -- otherwise which ring atom lands in which
    argument would be an accident of RDKit's search, and a rule using a position would hold on
    some molecules and not others. Mappings that differ only inside the dropped placeholders
    collapse to one tuple.
    """
    roots = None
    if spec.root_filter is not None:
        roots = {m[0] for m in mol.GetSubstructMatches(spec.root_filter, maxMatches=_MAX_MATCHES)}
    tuples: dict[tuple[int, ...], None] = {}
    for pattern, attach in spec.patterns:
        for match in mol.GetSubstructMatches(pattern, uniquify=False, maxMatches=_MAX_MATCHES):
            if roots is not None and match[0] not in roots:
                continue
            tuples[tuple(match[i] for i in attach)] = None
    return list(tuples)


def get_efg_atom_fgs(
    processed_df: "pd.DataFrame",
    efg_path: str = DEFAULT_EFG_PATH,
    cache_path: str | None = None,
) -> dict[int, list[tuple[str, tuple[int, ...]]]]:
    """Atom-anchored EFG matches: ``{molecule id: [(efga_<name>, (atom_idx, ...)), ...]}``.

    Same accept/reject semantics as :func:`get_efg_fgs` (reject patterns veto molecule-wide,
    all accept patterns must match), but instead of a single presence flag this emits one
    tuple of *attachment atoms* per match of the group's first accept pattern (see
    :func:`_attachment_indices`). A group's tuples all have the same length, its arity
    (:func:`efg_atom_arities`). Indices are 0-based RDKit indices; callers format them with
    ``get_atom_id`` (which adds 1).
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    patterns = [(pred, reject, accept, _efg_atom_spec(accept[0]) if accept else None)
                for pred, reject, accept in _load_efg_patterns(efg_path)]

    matches_by_mol: dict[int, list[tuple[str, tuple[int, ...]]]] = {}
    for row in tqdm.tqdm(processed_df.itertuples(), total=len(processed_df), desc="Matching EFG FGs (atom-anchored)"):
        occurrences: list[tuple[str, tuple[int, ...]]] = []
        for pred, reject_pats, accept_pats, spec in patterns:
            if spec is None:  # no accept pattern: nothing to anchor on
                continue
            if any(row.mol.HasSubstructMatch(p) for p in reject_pats):
                continue
            if not all(row.mol.HasSubstructMatch(p) for p in accept_pats):
                continue
            gat = _efg_atom_name(pred)
            occurrences += [(gat, t) for t in _efg_atom_tuples(row.mol, spec)]
        matches_by_mol[row.Index] = occurrences

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(matches_by_mol, fh)

    return matches_by_mol


def efg_atom_arities(efg_path: str = DEFAULT_EFG_PATH) -> dict[str, tuple[int, list[str]]]:
    """``{efga_<name>: (arity, [argument SMARTS, ...])}`` for every group with an accept pattern.

    The argument SMARTS is the query of the pattern atom each argument binds (for an expanded
    recursive pattern, of its first alternative), so a reader can tell which atom is which.
    """
    out = {}
    for pred, _reject, accept in _load_efg_patterns(efg_path):
        if not accept:
            continue
        spec = _efg_atom_spec(accept[0])
        pattern, attach = spec.patterns[0]
        out[_efg_atom_name(pred)] = (spec.arity, [pattern.GetAtomWithIdx(i).GetSmarts() for i in attach])
    return out


def efg_atom_fg_matches(rows, cache_path: str | None = None) -> dict:
    """``{molecule id: [(efga_<name>, (atom_idx, ...)), ...]}`` -- :func:`get_efg_atom_fgs`."""
    return get_efg_atom_fgs(rows, cache_path=cache_path)


def efg_atom_fg_vocabulary(efg_path: str = DEFAULT_EFG_PATH) -> list[tuple[str, str]]:
    """``[(efga_<name>, description)]`` for every anchorable EFG group.

    The description is the group's :func:`efg_fg_vocabulary` description plus its signature, e.g.
    ``Carboxylic acids: <text>; efga_carboxylic_acids(A1), A1: [C&X3]``, since the arity varies per
    group and a rule has to know which argument is which atom.
    """
    arities = efg_atom_arities(efg_path)
    vocab = []
    for pred, desc in efg_fg_vocabulary(efg_path):
        gat = _efg_atom_name(pred)
        if gat not in arities:  # no accept pattern -> never emitted
            continue
        arity, arg_smarts = arities[gat]
        args = [f"A{i + 1}" for i in range(arity)]
        sig = f"{gat}({','.join(args)}), " + ", ".join(f"{a}: {q}" for a, q in zip(args, arg_smarts))
        vocab.append((gat, f"{desc.rstrip('.')}; {sig}"))
    return vocab


def efg_descriptions(path: str = DEFAULT_EFG_DESCRIPTIONS_PATH) -> dict[str, str]:
    """``{EFG group name: description}`` from ``path``; empty if the file does not exist."""
    import csv

    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return {r["Name"]: r["Description"] for r in csv.DictReader(fh, delimiter="\t") if r["Description"]}


# ---------------------------------------------------------------------------
# Seeding the LLM rule library with a functional-group set
#
# A seed source turns a functional-group set into reusable library predicates. Each predicate
# carries the group as a molecule-level presence fact under a reserved prefix; its extension is
# the RDKit match, supplied at grounding rather than derived from a clause. Both consumers
# (ilp_problem_builder, generate_auxiliary_rules) react to the prefix, so a library can be
# seeded with either set (or neither) without any flag at build time.
# ---------------------------------------------------------------------------


def chembl_fg_vocabulary() -> list[tuple[str, str]]:
    """``(predicate_name, description)`` for every unique ChEMBL functional group.

    ``predicate_name`` is ``chembl_fg_<sanitized>`` where the ``<sanitized>`` part matches what
    :func:`get_chembl_fgs` emits per molecule, so a seeded predicate lines up with the injected
    ``chembl_fg_<name>(M)`` facts.
    """
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.CHEMBL)
    catalog = FilterCatalog.FilterCatalog(params)

    vocab: dict[str, str] = {}
    for i in range(catalog.GetNumEntries()):
        desc = catalog.GetEntryWithIdx(i).GetDescription()
        name = _sanitize_fg_name(desc)
        if name and name not in vocab:
            vocab[name] = desc
    return [(f"{CHEMBL_FG_FACT_PREFIX}{name}", desc) for name, desc in sorted(vocab.items())]


def efg_fg_vocabulary(efg_path: str = DEFAULT_EFG_PATH) -> list[tuple[str, str]]:
    """``(predicate_name, description)`` for every Extended Functional Group.

    ``predicate_name`` is the ``efg_<sanitized>`` name :func:`get_efg_fgs` emits per molecule.
    The description is ``<group name>: <text>`` with the text from :func:`efg_descriptions`, or
    the bare group name for a group without one. Reads only the ``Name`` column, so no SMARTS are
    compiled.
    """
    import csv

    descriptions = efg_descriptions()
    vocab: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(efg_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header
        for idx, row in enumerate(reader):
            name = row[0]
            text = descriptions.get(name)
            vocab.append((_efg_pred_name(name, idx, seen), f"{name}: {text}" if text else name))
    return vocab


def chembl_fg_matches(rows, cache_path: str | None = None) -> dict:
    """``{molecule id: [chembl_fg_<name>, ...]}`` — :func:`get_chembl_fgs` with the prefix applied."""
    raw = get_chembl_fgs(rows, cache_path=cache_path)
    return {k: [f"{CHEMBL_FG_FACT_PREFIX}{n}" for n in v] for k, v in raw.items()}


def efg_fg_matches(rows, cache_path: str | None = None) -> dict:
    """``{molecule id: [efg_<name>, ...]}`` — :func:`get_efg_fgs` (already ``efg_``-prefixed)."""
    return get_efg_fgs(rows, cache_path=cache_path)


class FGSeedSource(NamedTuple):
    """One functional-group seed source (a key of :data:`FG_SEED_SOURCES`).

    ``matcher(rows, cache_path)`` returns ``{molecule id: [...]}``: for a molecule-level source
    (``atom_level`` False) the values are prefixed predicate names; for an atom-level source they
    are ``(predicate_name, atom_idx)`` pairs. ``atom_level`` selects the fact emitter
    (:func:`fg_atom_fact_lines` vs :func:`fg_fact_lines`) at every injection site. ``usage`` tells
    the model how to call the source's predicates (see :func:`fg_seed_usage`).
    """
    prefix: str
    matcher: "callable"
    vocabulary: "callable"
    cache: str
    atom_level: bool = False
    usage: str = ""


_PRESENCE_USAGE = "{prefix}<name>(M): molecule M contains the functional group (a matched fact, no clause body)."

FG_SEED_SOURCES: dict[str, FGSeedSource] = {
    "chembl_fgs": FGSeedSource(CHEMBL_FG_FACT_PREFIX, chembl_fg_matches, chembl_fg_vocabulary, "chembl_fgs.pkl",
                               usage=_PRESENCE_USAGE.format(prefix=CHEMBL_FG_FACT_PREFIX)),
    "efg": FGSeedSource(EFG_FACT_PREFIX, efg_fg_matches, efg_fg_vocabulary, "efg_fgs.pkl",
                        usage=_PRESENCE_USAGE.format(prefix=EFG_FACT_PREFIX)),
    "efg_atoms": FGSeedSource(EFG_ATOM_FACT_PREFIX, efg_atom_fg_matches, efg_atom_fg_vocabulary, EFG_ATOM_CACHE,
                              atom_level=True,
                              usage=f"{EFG_ATOM_FACT_PREFIX}<name>(A1,..,Ak): one fact per occurrence of the "
                                    "functional group (matched, no clause body); the arguments are its attachment "
                                    "atoms, as in the candidate's signature. Join to the molecule via has_atom(M,A1)."),
}
# Every reserved seed prefix, as a tuple for ``str.startswith`` / ``referenced_fact_predicates``.
FG_SEED_PREFIXES: tuple[str, ...] = tuple(src.prefix for src in FG_SEED_SOURCES.values())


def is_fg_seed_name(name: str) -> bool:
    """True if ``name`` is a seeded functional-group predicate (matched, not clause-derived)."""
    return name.startswith(FG_SEED_PREFIXES)


def fg_seed_usage(names) -> str:
    """How to call the seeded functional-group predicates among ``names``: one line per source.

    Empty if none of ``names`` is a seed. The registered prefixes are disjoint under
    ``str.startswith``, so each name belongs to at most one source.
    """
    used = {src.prefix: src.usage for name in names for src in FG_SEED_SOURCES.values()
            if name.startswith(src.prefix)}
    return "\n".join(f"- {usage}" for usage in used.values())


def fg_fact_lines(matches: dict, ids, needed: set[str] | None = None) -> list[str]:
    """``<name>(<id>).`` presence facts from a matcher's output, for the given molecule ids.

    ``matches`` maps a molecule id to its matched (full, prefixed) predicate names. ``needed``
    optionally restricts output to those names. Ids are matched by string so an int- or
    str-keyed frame both work.
    """
    by_str = {str(k): v for k, v in matches.items()}
    lines: list[str] = []
    for i in ids:
        # A group can match a molecule several times; one presence fact is enough.
        for name in dict.fromkeys(by_str.get(str(i), [])):
            if needed is None or name in needed:
                lines.append(f"{name}({i}).")
    return lines


def fg_atom_fact_lines(matches: dict, ids, needed: set[str] | None = None) -> list[str]:
    """``<name>(a<mol>_<idx+1>,...).`` atom-anchored facts from an atom-level matcher's output.

    ``matches`` maps a molecule id to a list of ``(predicate_name, (atom_idx, ...))`` occurrences
    (see :func:`get_efg_atom_fgs`). The atom ids follow :func:`chebILP.utils.get_atom_id`, so these
    facts join to the molecule through the ``has_atom(M, a<mol>_<idx+1>)`` already in the BK.
    """
    by_str = {str(k): v for k, v in matches.items()}
    lines: list[str] = []
    for i in ids:
        for name, atom_idxs in by_str.get(str(i), []):
            if needed is None or name in needed:
                lines.append(f"{name}({','.join(f'a{i}_{j + 1}' for j in atom_idxs)}).")
    return lines


def seed_fg_library(library_dir: str, source_key: str, overwrite: bool = False) -> int:
    """Write one header-only library predicate per functional group of ``source_key``.

    ``source_key`` is a key of :data:`FG_SEED_SOURCES` (``"chembl_fgs"`` or ``"efg"``). Only the
    ``programs/`` files are written (no ``class_map.json``), so the generator still runs every
    class and simply has the groups available to reuse. Returns the number of files written.
    """
    from chebILP.ilp_path_manager import get_aux_programs_dir

    if source_key not in FG_SEED_SOURCES:
        raise ValueError(f"unknown seed source {source_key!r}; choices: {sorted(FG_SEED_SOURCES)}")
    src = FG_SEED_SOURCES[source_key]

    # How to call a seeded predicate is stated once in the prompt (``fg_seed_usage``), not per
    # file: the same boilerplate on hundreds of entries only dilutes retrieval.
    programs_dir = get_aux_programs_dir(base_dir=library_dir)
    written = 0
    for name, description in src.vocabulary():
        path = os.path.join(programs_dir, f"{name}.pl")
        if os.path.exists(path) and not overwrite:
            continue
        desc = " ".join(str(description).split()) or name
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"% PREDICATE_NAME: {name}\n"
                    f"% DESCRIPTION: {desc}\n")
        written += 1
    return written


def get_chebi_fgs(
    processed_df: "pd.DataFrame",
    smarts_path: str = os.path.join("data", "chebi_fg_smarts.csv"),
    cache_path: str | None = None,
) -> dict[int, list[str]]:
    """Return a dict mapping molecule (ChEBI) IDs to lists of ChEBI FG predicate names.

    Parameters
    ----------
    processed_df : pd.DataFrame
        Must have a ``mol`` column (RDKit ``Mol`` objects) and be indexed by
        integer ChEBI IDs.
    smarts_path : str
        Path to a CSV with columns ``group_id`` and ``smarts``.
    cache_path : str or None
        If given, results are pickled to / loaded from this path.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    import pandas as pd

    smarts_df = pd.read_csv(smarts_path)
    compiled = [
        (f"chebi_fg_{r['group_id']}", Chem.MolFromSmarts(r["smarts"]))
        for _, r in smarts_df.iterrows()
    ]

    fg_matches_by_mol: dict[int, list[str]] = {}
    for row in tqdm.tqdm(
        processed_df.itertuples(),
        total=len(processed_df),
        desc="Matching ChEBI FGs",
    ):
        fg_matches_by_mol[row.Index] = []
        for fg_name, pattern in compiled:
            if row.mol.HasSubstructMatch(pattern):
                fg_matches_by_mol[row.Index].append(fg_name)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(fg_matches_by_mol, fh)

    return fg_matches_by_mol
