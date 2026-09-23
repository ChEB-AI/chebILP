"""Functional-group matching helpers.

Provides ``get_chembl_fgs`` and ``get_chebi_fgs`` which replicate the
identically-named methods formerly on ``chemlog.preprocessing.chebi_data.ChEBIData``.
Results are cached to pickle files so expensive SMARTS matching is only
performed once per dataset.
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

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

DEFAULT_EFG_PATH = os.path.join("data", "efg_functional_groups.tsv")


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
    Reads only the ``Name`` column, so no SMARTS are compiled.
    """
    import csv

    vocab: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(efg_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header
        for idx, row in enumerate(reader):
            name = row[0]
            vocab.append((_efg_pred_name(name, idx, seen), name))
    return vocab


def chembl_fg_matches(rows, cache_path: str | None = None) -> dict:
    """``{molecule id: [chembl_fg_<name>, ...]}`` — :func:`get_chembl_fgs` with the prefix applied."""
    raw = get_chembl_fgs(rows, cache_path=cache_path)
    return {k: [f"{CHEMBL_FG_FACT_PREFIX}{n}" for n in v] for k, v in raw.items()}


def efg_fg_matches(rows, cache_path: str | None = None) -> dict:
    """``{molecule id: [efg_<name>, ...]}`` — :func:`get_efg_fgs` (already ``efg_``-prefixed)."""
    return get_efg_fgs(rows, cache_path=cache_path)


# key -> (fact prefix, matcher(rows, cache_path)->{id:[full_name,...]}, vocabulary(), cache filename)
FG_SEED_SOURCES: dict[str, tuple] = {
    "chembl_fgs": (CHEMBL_FG_FACT_PREFIX, chembl_fg_matches, chembl_fg_vocabulary, "chembl_fgs.pkl"),
    "efg": (EFG_FACT_PREFIX, efg_fg_matches, efg_fg_vocabulary, "efg_fgs.pkl"),
}
# Every reserved seed prefix, as a tuple for ``str.startswith`` / ``referenced_fact_predicates``.
FG_SEED_PREFIXES: tuple[str, ...] = tuple(prefix for prefix, *_ in FG_SEED_SOURCES.values())


def is_fg_seed_name(name: str) -> bool:
    """True if ``name`` is a seeded functional-group predicate (matched, not clause-derived)."""
    return name.startswith(FG_SEED_PREFIXES)


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


def seed_fg_library(library_dir: str, source_key: str, overwrite: bool = False) -> int:
    """Write one header-only library predicate per functional group of ``source_key``.

    ``source_key`` is a key of :data:`FG_SEED_SOURCES` (``"chembl_fgs"`` or ``"efg"``). Only the
    ``programs/`` files are written (no ``class_map.json``), so the generator still runs every
    class and simply has the groups available to reuse. Returns the number of files written.
    """
    from chebILP.ilp_path_manager import get_aux_programs_dir

    if source_key not in FG_SEED_SOURCES:
        raise ValueError(f"unknown seed source {source_key!r}; choices: {sorted(FG_SEED_SOURCES)}")
    _prefix, _matcher, vocabulary, _cache = FG_SEED_SOURCES[source_key]

    programs_dir = get_aux_programs_dir(base_dir=library_dir)
    written = 0
    for name, description in vocabulary():
        path = os.path.join(programs_dir, f"{name}.pl")
        if os.path.exists(path) and not overwrite:
            continue
        desc = " ".join(str(description).split()) or name
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"% PREDICATE_NAME: {name}\n"
                    f"% DESCRIPTION: {desc} ({source_key} functional group, molecule-level presence)\n")
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
