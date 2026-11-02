from dataclasses import dataclass
from typing import Optional

from abc import ABC, abstractmethod
import os
import networkx as nx
import pandas as pd

from chebi_utils import create_multilabel_splits


@dataclass(frozen=True)
class PromptSpec:
    """Dataset-specific text of the LLM rule-generation prompt."""

    system_prompt: str
    existing_predicates: str
    computed_predicates: str
    # "molecule", "reaction", "domain": what one example is
    example_noun: str
    # how ``describe_example`` renders an example, e.g. "SMILES"
    example_format: str
    # siblings listed in the prompt; None lists all
    max_siblings: Optional[int] = None
    # show the definitions of parents and siblings, not only their names
    relative_definitions: bool = False
    # add "... and N more" when siblings are cut at max_siblings
    count_hidden_siblings: bool = True
    # list examples attached directly to a parent among the siblings
    examples_as_siblings: bool = False
    sibling_header: str = "Sibling classes (the negatives come from these):"


class BaseDataset(ABC):
    """A labelled dataset whose classes form a hierarchy.

    Subclasses set ``molecules`` (one row per example, index = example id, column ``mol``) and
    ``graph`` (an ``nx.DiGraph`` with child -> parent edges, examples as leaves, nodes carrying
    ``name`` and ``definition``) in :meth:`process_data`.
    """

    DATASET_NAME: str = "base_dataset"

    def __init__(self, version: str, base_dir: str = "data", min_pos_samples: int = 25):
        self.version = version
        self.base_dir = base_dir
        self.min_pos_samples = min_pos_samples

        self.labels_file = os.path.join(self.processed_dir, "labels.txt")
        self.splits_file = os.path.join(self.processed_dir, "splits.csv")

        # cached labeled DataFrame after select_labels()
        # todo this should be loaded from disk if available
        self._labeled_df: Optional[pd.DataFrame] = None
        self._closure: Optional[nx.DiGraph] = None

    def _raw(self, name: str) -> str:
        return os.path.join(self.raw_dir, name)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_dir, f"{self.DATASET_NAME}_v{self.version}")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.data_dir, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.data_dir, f"processed_{self.min_pos_samples}")

    def download(self) -> None:
        pass

    def process_data(self) -> None:
        pass

    def select_labels(self) -> list[str]:
        """
        Identify the label IDs for the dataset. Criterion is the self.min_pos_samples threshold.
        ILP problems / LLM predicate generation follows the order of this list, thus
        it should be sorted (e.g., by hierarchy level for ChEBI)."""
        pass

    def load_labels(self) -> list[str]:
        with open(self.labels_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def create_splits(
        self,
        labels: list[str],
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Create a stratified train/val/test split at the molecule level.

        Uses the labeled DataFrame cached by :meth:`select_labels`; re-builds
        it if not cached.  Molecules that belong to none of the selected label
        classes are distributed randomly across splits.

        Returns a DataFrame with columns ``[id, split]``.
        """
        labeled_df = self._labeled_df
        if labeled_df is None:
            raise ValueError(
                "Labeled DataFrame not available. Run `select_labels()` first."
            )
        label_set = set(labels)
        label_cols = [c for c in labeled_df.columns[2:] if c in label_set]
        if not label_cols:
            raise ValueError(
                "None of the provided labels appear in the labeled DataFrame."
            )

        split_input = labeled_df[["id", "sample"] + label_cols].copy()
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
        for split_name, split in raw_splits.items():
            records.extend((mid, split_name) for mid in split["id"].tolist())
        splits_df = pd.DataFrame(records, columns=["id", "split"])

        if self.splits_file:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.splits_file)), exist_ok=True
            )
            splits_df.to_csv(self.splits_file, index=False)
            n = splits_df["split"].value_counts()
            print(
                f"Saved splits to {self.splits_file}  "
                f"({n.get('train', 0)} train / {n.get('validation', 0)} val / {n.get('test', 0)} test)"
            )
        return splits_df

    def load_splits_from_csv(self) -> pd.DataFrame:
        """
        Load a previously saved splits CSV file from the processed directory.  Returns a DataFrame with
        columns ``[id, split]``.
        """
        if not os.path.exists(self.splits_file):
            raise FileNotFoundError(
                f"Splits file not found: {self.splits_file}. "
                f"Run `python -m chebILP prepare_dataset {self.DATASET_NAME}` to create it."
            )
        splits_df = pd.read_csv(self.splits_file, dtype={"id": str})
        if "id" not in splits_df.columns or "split" not in splits_df.columns:
            raise ValueError(
                f"Splits CSV must contain 'id' and 'split' columns: {self.splits_file}"
            )
        if not all(
            s in splits_df["split"].unique() for s in ["train", "validation", "test"]
        ):
            raise ValueError(
                f"Splits CSV must contain 'train', 'validation', and 'test' splits: {self.splits_file}"
            )
        return splits_df

    @classmethod
    def prepare(
        cls,
        version: str,
        base_dir: str = "data",
        min_pos_samples: int = 25,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> "BaseDataset":
        """Download, build caches, select labels, and save splits in one call."""
        dataset = cls(
            version=version, base_dir=base_dir, min_pos_samples=min_pos_samples
        )

        os.makedirs(dataset.raw_dir, exist_ok=True)
        dataset.download()
        os.makedirs(dataset.processed_dir, exist_ok=True)
        dataset.process_data()

        labels = dataset.select_labels()
        dataset.create_splits(
            labels=labels,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        return dataset

    # --- hierarchy ---------------------------------------------------------------

    @property
    def hierarchy_closure(self) -> nx.DiGraph:
        if self._closure is None:
            self._closure = nx.transitive_closure_dag(self.graph)
        return self._closure

    def _examples_below(self, node: str) -> set[str]:
        """Example ids at or below ``node``."""
        below = {str(d) for d in self.hierarchy_closure.predecessors(node)} | {
            str(node)
        }
        return below & self._example_ids

    @property
    def _example_ids(self) -> set[str]:
        ids = getattr(self, "_example_id_cache", None)
        if ids is None:
            ids = self._example_id_cache = {str(i) for i in self.molecules.index}
        return ids

    def get_positive_examples(self, target_id: str) -> set[str]:
        """Example ids at or below ``target_id`` (a target that is itself an example counts)."""
        return self._examples_below(target_id)

    def get_negative_examples(self, target_id: str) -> set[str]:
        """Direct neighbours: examples below every direct parent of ``target_id``, minus its positives."""
        parent_spaces = [
            self._examples_below(p) for p in self.graph.successors(target_id)
        ]
        if not parent_spaces:
            return set()
        return set.intersection(*parent_spaces) - self.get_positive_examples(target_id)

    def class_label(self, class_id: str) -> str:
        """Human-readable id of a class, as shown in prompts and logs."""
        return str(class_id)

    def _node_info(self, node: str) -> dict:
        data = self.graph.nodes.get(node, {})
        return {
            "id": node,
            "label": self.class_label(node),
            "name": data.get("name") or self.class_label(node),
            "definition": data.get("definition"),
        }

    def class_info(self, class_id: str) -> dict:
        """Name, definition, parents and siblings (the parents' other child classes) of a class."""
        skip = set() if self.prompt_spec.examples_as_siblings else self._example_ids
        parents, siblings = [], {}
        for pid in self.graph.successors(class_id):
            parents.append(self._node_info(pid))
            for sid in self.graph.predecessors(pid):
                if sid != class_id and sid not in siblings and str(sid) not in skip:
                    siblings[sid] = self._node_info(sid)
        return {
            **self._node_info(class_id),
            "parents": parents,
            "siblings": list(siblings.values()),
        }

    def sort_labels(self, labels: list[str]) -> list[str]:
        """``labels`` with every superclass before its subclasses, otherwise in their given order."""
        missing = set(labels) - set(self.graph)
        if missing:
            raise ValueError(f"Some labels are not in the class hierarchy: {missing}")
        label_set = set(labels)
        order_graph = nx.DiGraph()
        order_graph.add_nodes_from(labels)
        for label in labels:
            for superclass in nx.descendants(self.graph, label) & label_set:
                order_graph.add_edge(superclass, label)
        position = {label: i for i, label in enumerate(labels)}
        return list(
            nx.lexicographical_topological_sort(order_graph, key=position.__getitem__)
        )

    # --- examples ----------------------------------------------------------------

    def example_sizes(self) -> pd.Series:
        """Size of each example (index = example id): atoms of its ``mol``."""
        sizes = getattr(self, "_sizes", None)
        if sizes is None:
            sizes = self._sizes = self.molecules["mol"].map(lambda m: m.GetNumAtoms())
        return sizes

    def select_train_subset(self, ids: list[str], max_samples: int) -> list[str]:
        """The ``max_samples`` smallest of ``ids``; ties resolve by the order of ``ids``."""
        if len(ids) <= max_samples:
            return list(ids)
        sizes = self.example_sizes()
        return list(sizes[list(ids)].nsmallest(max_samples, keep="first").index)

    def get_example_weight(self, example_id: str) -> float:
        return 1.0

    # --- encoding ----------------------------------------------------------------

    @abstractmethod
    def build_background(
        self, rows: pd.DataFrame
    ) -> tuple[list[str], list[tuple[str, int]]]:
        """BK fact lines for the examples in ``rows`` and their body predicates ``(name, arity)``."""

    def build_computed_facts(self, rows: pd.DataFrame) -> list[str]:
        """Numeric facts offered to LLM rules only; never written to ``bk.pl``."""
        return []

    def seed_fact_lines(self, names: set[str], rows: pd.DataFrame) -> list[str]:
        """Facts over ``rows`` of library predicates ``names`` that are not ASP rules (see ``is_fg_seed_name``)."""
        return []

    def load_seed_matches(self, names: set[str]) -> None:
        """Load the dataset-wide data ``seed_fact_lines`` needs for ``names``, so it survives dropping ``molecules``."""

    # --- prompting ---------------------------------------------------------------

    def candidate_notes(self, names) -> str:
        """Extra prompt text about the offered reuse candidates ``names``."""
        return ""

    @abstractmethod
    def describe_example(self, example_id: str) -> str:
        """One-line text form of an example for prompts."""

    @property
    @abstractmethod
    def prompt_spec(self) -> PromptSpec:
        """Prompt text for LLM rule generation."""
