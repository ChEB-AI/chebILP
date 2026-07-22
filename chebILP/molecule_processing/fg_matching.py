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


def _sanitize_fg_name(raw_name: str) -> str:
    """Convert a ChEMBL FilterCatalog description to a Prolog-safe predicate name."""
    name = raw_name.lower().replace(" ", "_").replace("-", "_")
    name = name.replace("/", "_or_").replace(">", "_more_than_").replace("<", "_less_than_")
    name = "".join(c for c in name if c.isalnum() or c == "_")
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
