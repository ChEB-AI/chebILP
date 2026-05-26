"""
Dataset preparation: download ChEBI data, build and cache the hierarchy graph
and molecule DataFrame, select label classes, and create train/val/test splits.

Typical usage (CLI):
    python -m chebILP prepare_dataset --chebi_version 248 --min_pos_samples 50

After running, the following files are ready for use by the ILP pipeline:
    data/chebi_v248/chebi_graph.pkl          – networkx DiGraph (hierarchy subgraph)
    data/chebi_v248/molecules.pkl            – pandas DataFrame  (index = ChEBI ID)
    data/chebi_v248/min50/labels.txt         – one selected class ID per line
    data/chebi_v248/min50/splits.csv         – mol_id,split  (train/validation/test)
"""

import argparse
import os
import pickle
from typing import Optional

import networkx as nx
import pandas as pd

from chebi_utils import (
    build_chebi_graph,
    build_labeled_dataset,
    create_multilabel_splits,
    download_chebi_obo,
    download_chebi_sdf,
    extract_molecules,
    get_hierarchy_subgraph,
)


class ChEBIDataset:
    """
    Encapsulates a ChEBI dataset: downloads raw files on demand, builds and
    caches the hierarchy graph and molecule DataFrame, selects label classes,
    and creates stratified train/val/test splits.
    """

    def __init__(
        self,
        chebi_version: int,
        data_dir: Optional[str] = None,
    ):
        self.chebi_version = chebi_version
        self.data_dir = data_dir or os.path.join("data", f"chebi_v{chebi_version}")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.obo_path = os.path.join(self.raw_dir, "chebi.obo")
        self.sdf_path = os.path.join(self.raw_dir, "chebi.sdf.gz")
        self.graph_path = os.path.join(self.data_dir, "chebi_graph.pkl")
        self.molecules_path = os.path.join(self.data_dir, "molecules.pkl")

        os.makedirs(self.raw_dir, exist_ok=True)
        self._download_if_needed()
        self.chebi_graph: nx.DiGraph = self._load_or_build_graph()
        self.molecules: pd.DataFrame = self._load_or_build_molecules()
        self._labeled_df: Optional[pd.DataFrame] = None  # cache set by select_labels

    # ── Setup helpers ──────────────────────────────────────────────────────

    def _download_if_needed(self) -> None:
        if not os.path.exists(self.obo_path):
            print(f"Downloading chebi.obo v{self.chebi_version}...")
            download_chebi_obo(self.chebi_version, dest_dir=self.data_dir)
        if not os.path.exists(self.sdf_path):
            print(f"Downloading chebi.sdf.gz v{self.chebi_version}...")
            download_chebi_sdf(self.chebi_version, dest_dir=self.data_dir)

    def _load_or_build_graph(self) -> nx.DiGraph:
        if os.path.exists(self.graph_path):
            print(f"Loading ChEBI graph from {self.graph_path}...")
            with open(self.graph_path, "rb") as f:
                return pickle.load(f)
        print("Building ChEBI hierarchy graph...")
        graph = get_hierarchy_subgraph(build_chebi_graph(self.obo_path))
        with open(self.graph_path, "wb") as f:
            pickle.dump(graph, f)
        print(f"  Saved to {self.graph_path} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
        return graph

    def _load_or_build_molecules(self) -> pd.DataFrame:
        if os.path.exists(self.molecules_path):
            print(f"Loading molecules from {self.molecules_path}...")
            return pd.read_pickle(self.molecules_path)
        print("Extracting molecules from SDF (this may take several minutes)...")
        molecules = extract_molecules(self.sdf_path)
        molecules.index = molecules["chebi_id"].astype(str)
        molecules.index.name = None
        molecules.to_pickle(self.molecules_path)
        print(f"  Saved {len(molecules)} molecules to {self.molecules_path}")
        return molecules

    # ── Public API ─────────────────────────────────────────────────────────

    def select_labels(
        self,
        min_pos_samples: int = 50,
        output_path: Optional[str] = None,
    ) -> list[str]:
        """
        Return ChEBI class IDs with at least *min_pos_samples* descendant
        molecules.  Writes ``labels.txt`` if *output_path* is given.
        The resulting labeled DataFrame is cached for :meth:`create_splits`.
        """
        labeled_df, labels = build_labeled_dataset(
            chebi_graph=self.chebi_graph,
            molecules=self.molecules,
            min_molecules=min_pos_samples,
        )
        self._labeled_df = labeled_df

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                f.write("\n".join(labels) + "\n")
            print(f"Saved {len(labels)} labels to {output_path}")
        return labels

    def create_splits(
        self,
        labels: list[str],
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create a stratified train/val/test split at the molecule level.

        Uses the labeled DataFrame cached by :meth:`select_labels`; re-builds
        it if not cached.  Molecules that belong to none of the selected label
        classes are distributed randomly across splits.

        Returns a DataFrame with columns ``[mol_id, split]``.
        """
        if self._labeled_df is not None:
            labeled_df = self._labeled_df
        else:
            labeled_df, _ = build_labeled_dataset(
                chebi_graph=self.chebi_graph,
                molecules=self.molecules,
                min_molecules=1,
            )

        label_set = set(labels)
        label_cols = [c for c in labeled_df.columns[2:] if c in label_set]
        if not label_cols:
            raise ValueError("None of the provided labels appear in the labeled DataFrame.")

        split_input = labeled_df[["chebi_id", "mol"] + label_cols].copy()
        train_ratio = round(1.0 - val_ratio - test_ratio, 10)
        raw_splits = create_multilabel_splits(
            split_input,
            label_start_col=2,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        records: list[tuple[str, str]] = []
        for split_name, key in [("train", "train"), ("validation", "val"), ("test", "test")]:
            records.extend((mid, split_name) for mid in raw_splits[key]["chebi_id"].tolist())
        splits_df = pd.DataFrame(records, columns=["mol_id", "split"])

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            splits_df.to_csv(output_path, index=False)
            n = splits_df["split"].value_counts()
            print(
                f"Saved splits to {output_path}  "
                f"({n.get('train', 0)} train / {n.get('validation', 0)} val / {n.get('test', 0)} test)"
            )
        return splits_df

    @classmethod
    def prepare(
        cls,
        chebi_version: int,
        data_dir: Optional[str] = None,
        min_pos_samples: int = 50,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        labels_path: Optional[str] = None,
        splits_path: Optional[str] = None,
    ) -> "ChEBIDataset":
        """Download, build caches, select labels, and save splits in one call."""
        dataset = cls(chebi_version=chebi_version, data_dir=data_dir)
        processed = os.path.join(dataset.data_dir, f"min{min_pos_samples}")
        labels_out = labels_path or os.path.join(processed, "labels.txt")
        splits_out = splits_path or os.path.join(processed, "splits.csv")

        labels = dataset.select_labels(min_pos_samples=min_pos_samples, output_path=labels_out)
        dataset.create_splits(
            labels=labels,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            output_path=splits_out,
        )
        return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ChEBI data and build all dataset artefacts.",
    )
    parser.add_argument("--chebi_version", type=int, required=True,
                        help="ChEBI ontology version (e.g. 248).")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Root directory for raw and cached files "
                             "(default: data/chebi_v{version}).")
    parser.add_argument("--min_pos_samples", type=int, default=50,
                        help="Minimum descendant molecules a class must have "
                             "to be included as a label (default: 50).")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of molecules for validation (default: 0.1).")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Fraction of molecules for test (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splits (default: 42).")
    parser.add_argument("--labels_path", type=str, default=None,
                        help="Output path for labels.txt "
                             "(default: {data_dir}/min{n}/labels.txt).")
    parser.add_argument("--splits_path", type=str, default=None,
                        help="Output path for splits.csv "
                             "(default: {data_dir}/min{n}/splits.csv).")
    args = parser.parse_args()

    ChEBIDataset.prepare(
        chebi_version=args.chebi_version,
        data_dir=args.data_dir,
        min_pos_samples=args.min_pos_samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        labels_path=args.labels_path,
        splits_path=args.splits_path,
    )


if __name__ == "__main__":
    main()
