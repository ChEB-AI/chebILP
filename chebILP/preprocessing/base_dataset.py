from typing import Optional

from abc import ABC
import os
import pandas as pd

from chebi_utils import create_multilabel_splits


class BaseDataset(ABC):
    DATASET_NAME: str = "base_dataset"

    def __init__(self, version: str, base_dir: str = "data", min_pos_samples: int = 25):
        self.version = version
        self.base_dir = base_dir
        self.min_pos_samples = min_pos_samples

        self.labels_file = os.path.join(self.processed_dir, "labels.txt")
        self.splits_file = os.path.join(self.processed_dir, "splits.csv")

        # cached labeled DataFrame after select_labels()
        self._labeled_df: Optional[pd.DataFrame] = None

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
        pass

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
