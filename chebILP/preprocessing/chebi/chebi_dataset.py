"""
Dataset preparation: download ChEBI data, build and cache the hierarchy graph
and molecule DataFrame, select label classes, and create train/val/test splits.

Example usage:
    python -m chebILP prepare_dataset chebi --version 251

After running, the following files are ready for use by the ILP pipeline:
    data/chebi_v251/chebi_graph.pkl                 – networkx DiGraph (hierarchy subgraph)
    data/chebi_v251/processed_25_3_STAR/molecules.pkl    – pandas DataFrame  (index = ChEBI ID)
    data/chebi_v251/processed_25_3_STAR/labels.txt       – one selected class ID per line
    data/chebi_v251/processed_25_3_STAR/splits.csv       – id,split  (train/validation/test)
"""

import os
import pickle

import networkx as nx
import pandas as pd
from rdkit import Chem

from chebILP.preprocessing.base_dataset import BaseDataset, PromptSpec
from chebILP.preprocessing.chebi import llm_prompt
from chebILP.preprocessing.chebi.fg_matching import (
    FG_SEED_SOURCES,
    fg_atom_fact_lines,
    fg_fact_lines,
    fg_seed_usage,
)
from chebILP.preprocessing.chebi.mol_to_fol import (
    build_background_chemlog,
    build_computed_facts,
)

from chebi_utils import (
    build_chebi_graph,
    build_labeled_dataset,
    download_chebi_obo,
    download_chebi_sdf,
    extract_molecules,
    get_hierarchy_subgraph,
)


class ChEBIDataset(BaseDataset):
    """
    Encapsulates a ChEBI dataset: downloads raw files on demand, builds and
    caches the hierarchy graph and molecule DataFrame, selects label classes,
    and creates stratified train/val/test splits.
    """

    DATASET_NAME: str = "chebi"

    def __init__(
        self,
        version: str,
        base_dir: str = "data",
        min_pos_samples: int = 25,
        three_star_only: bool = True,
    ):
        # processed_dir, which the base constructor reads, depends on three_star_only
        self.three_star_only = three_star_only
        super().__init__(
            version=version, base_dir=base_dir, min_pos_samples=min_pos_samples
        )

        self.obo_path = self._raw("chebi.obo")
        self.sdf_path = self._raw("chebi.sdf.gz")
        self.graph_path = os.path.join(self.data_dir, "chebi_graph.pkl")
        self.molecules_path = os.path.join(self.processed_dir, "molecules.pkl")

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            self.data_dir,
            f"processed_{self.min_pos_samples}{'_3_STAR' if self.three_star_only else ''}",
        )

    def download(self) -> None:
        if not os.path.exists(self.obo_path):
            print(f"Downloading chebi.obo v{self.version}...")
            download_chebi_obo(self.version, dest_dir=self.raw_dir)
        if not os.path.exists(self.sdf_path):
            print(f"Downloading chebi.sdf.gz v{self.version}...")
            download_chebi_sdf(self.version, dest_dir=self.raw_dir)

    def _load_or_build_graph(self) -> nx.DiGraph:
        if os.path.exists(self.graph_path):
            print(f"Loading ChEBI graph from {self.graph_path}...")
            with open(self.graph_path, "rb") as f:
                return pickle.load(f)
        print("Building ChEBI hierarchy graph...")
        graph = get_hierarchy_subgraph(
            build_chebi_graph(self.obo_path, top_class="23367")
        )
        with open(self.graph_path, "wb") as f:
            pickle.dump(graph, f)
        print(
            f"  Saved to {self.graph_path} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
        )
        return graph

    def _load_or_build_molecules(self) -> pd.DataFrame:
        if os.path.exists(self.molecules_path):
            print(f"Loading molecules from {self.molecules_path}...")
            return pd.read_pickle(self.molecules_path)
        print("Extracting molecules from SDF (this may take several minutes)...")
        molecules = extract_molecules(self.sdf_path)
        molecules.index = molecules["chebi_id"].astype(str)
        molecules.index.name = None
        # filter for 3-star subset
        if self.three_star_only:
            molecules = molecules[molecules["STAR"] == "3"]
        molecules.to_pickle(self.molecules_path)
        print(f"  Saved {len(molecules)} molecules to {self.molecules_path}")
        return molecules

    def select_labels(self) -> list[str]:
        """
        Return ChEBI class IDs with at least *min_pos_samples* descendant
        molecules.  Writes ``labels.txt`` if *output_path* is given.
        The resulting labeled DataFrame is cached for :meth:`create_splits`.
        The returned list is sorted by hierarchy level (superclasses first)
        """
        labeled_df, labels = build_labeled_dataset(
            chebi_graph=self.graph,
            molecules=self.molecules,
            min_molecules=self.min_pos_samples,
        )
        self._labeled_df = labeled_df.rename(
            columns={"chebi_id": "id", "mol": "sample"}
        )

        labels = self.sort_labels(labels)

        if self.labels_file:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.labels_file)), exist_ok=True
            )
            with open(self.labels_file, "w") as f:
                f.write("\n".join(labels) + "\n")
            print(f"Saved {len(labels)} labels to {self.labels_file}")
        return labels

    def process_data(self) -> None:
        self.graph = self._load_or_build_graph()
        self.molecules = self._load_or_build_molecules()

    def build_background(self, rows):
        return build_background_chemlog(rows)

    def build_computed_facts(self, rows) -> list[str]:
        return build_computed_facts(rows)

    def _fg_matches(self, source_key: str, rows: pd.DataFrame) -> dict:
        """RDKit matches of a seed source, from the dataset-wide cache when ``rows`` are dataset molecules."""
        src = FG_SEED_SOURCES[source_key]
        cache = self.__dict__.setdefault("_fg_match_cache", {})
        if source_key not in cache:
            molecules = getattr(self, "molecules", None)
            if molecules is None or not rows.index.isin(molecules.index).all():
                return src.matcher(rows)
            cache[source_key] = src.matcher(
                molecules, cache_path=os.path.join(self.processed_dir, src.cache)
            )
        matches = cache[source_key]
        if not all(str(i) in matches for i in rows.index):
            return src.matcher(rows)
        return matches

    def load_seed_matches(self, names) -> None:
        for key, src in FG_SEED_SOURCES.items():
            if any(n.startswith(src.prefix) for n in names):
                self._fg_matches(key, self.molecules)

    def seed_fact_lines(self, names, rows) -> list[str]:
        """Facts of the seeded FG predicates ``names`` (``chembl_fg_*``, ``efg_*``, ``efga_*``) over ``rows``."""
        ids = [str(i) for i in rows.index]
        lines: list[str] = []
        for key, src in FG_SEED_SOURCES.items():
            want = {n for n in names if n.startswith(src.prefix)}
            if want:
                emit = fg_atom_fact_lines if src.atom_level else fg_fact_lines
                lines += emit(self._fg_matches(key, rows), ids, needed=want)
        return lines

    def candidate_notes(self, names) -> str:
        usage = fg_seed_usage(names)
        return (
            f"Functional-group candidates are called as follows:\n{usage}\n"
            if usage
            else ""
        )

    def class_label(self, class_id: str) -> str:
        return f"CHEBI:{class_id}"

    def describe_example(self, example_id: str) -> str:
        return Chem.MolToSmiles(self.molecules.at[example_id, "mol"])

    @property
    def prompt_spec(self) -> PromptSpec:
        return PromptSpec(
            system_prompt=llm_prompt.SYSTEM_PROMPT,
            existing_predicates=llm_prompt.EXISTING_PREDICATES,
            computed_predicates=llm_prompt.COMPUTED_PREDICATES,
            example_noun="molecule",
            example_format="SMILES",
            max_siblings=8,
            relative_definitions=True,
            count_hidden_siblings=False,
            examples_as_siblings=True,
            sibling_header="Sibling classes (look for the contrast, e.g. they may differ only by a count or chain length):",
        )
