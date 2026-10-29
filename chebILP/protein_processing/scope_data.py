"""
SCOPe protein domains labelled with their fold.

Samples are the domains of the ASTRAL <95% identity subset, labelled with their fold
``scop_<cl>_<cf>``. Folds carry a ``name`` (the ``dir.des`` description) and a ``definition``
(the ``dir.com`` comments, e.g. "core: 6 helices; folded leaf, partly opened").

Example usage:
    python -m chebILP prepare_dataset scope --version 2.08

Writes to ``data/scope_v<version>/``:
    domains.pkl                     – DataFrame, index = domain id; ``mol`` holds the DSSP structure
    folds.pkl                       – DataFrame of fold names and definitions
    processed_<min_pos>/labels.txt  – selected fold ids
    processed_<min_pos>/splits.csv  – id,split; grouped per fold by superfamily, family or domain
    processed_<min_pos>/holdout.csv – label,holdout; the level each fold's split keeps apart
"""

import os
import re
import tarfile
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import tqdm

from chebILP.preprocessing.base_dataset import BaseDataset
from chebILP.protein_processing.dssp import domain_structure

SCOPE_URL = "https://scop.berkeley.edu/downloads"
ASTRAL_SUBSET = 95  # use subset with <= 95% sequence identity
# Folds need this many positives in each split to be trained and tested.
MIN_POS_PER_SPLIT = 3
# Coordinate files held in memory at once while streaming the archive.
DSSP_BATCH_SIZE = 4096
# What a fold's split keeps apart, strictest first.
HOLDOUT_LEVELS = ("superfamily", "family", "domain")


def raw_files(version: str) -> dict[str, str]:
    files = {
        f"dir.{kind}.scope.{version}-stable.txt": f"{SCOPE_URL}/parse/dir.{kind}.scope.{version}-stable.txt"
        for kind in ("des", "cla", "com")
    }
    files[pdbstyle_archive(version)] = (
        f"{SCOPE_URL}/pdbstyle/{pdbstyle_archive(version)}"
    )
    return files


def pdbstyle_archive(version: str) -> str:
    return f"pdbstyle-sel-gs-bib-{ASTRAL_SUBSET}-{version}.tgz"


def fold_id(sccs: str) -> str:
    """``c.1.8.4`` -> ``scop_c_1``."""
    return "scop_" + "_".join(sccs.split(".")[:2])


def domain_id(sid: str) -> str:
    """A SCOPe sid as an ASP constant: lower case, ``.`` -> ``_``."""
    return re.sub(r"[^a-z0-9_]", "_", sid.lower())


def parse_des(path: str) -> dict[int, tuple[str, str, str]]:
    """``sunid -> (level, sccs, description)``."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            sunid, level, sccs, _sid, description = line.rstrip("\n").split("\t")
            out[int(sunid)] = (level, sccs, description)
    return out


def parse_com(path: str) -> dict[int, list[str]]:
    """``sunid -> [comment, ...]``."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            sunid, *comments = [c.strip() for c in line.rstrip("\n").split("!")]
            out[int(sunid)] = [c for c in comments if c]
    return out


def parse_cla(path: str) -> pd.DataFrame:
    """One row per domain: sid, pdb, region, sccs, px and the ancestor sunids (cl, cf, sf, fa, dm, sp)."""
    records = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            sid, pdb, region, sccs, px, ancestors = line.rstrip("\n").split("\t")
            record = {"sid": sid, "pdb": pdb, "region": region, "sccs": sccs}
            record.update(
                {k: int(v) for k, v in (kv.split("=") for kv in ancestors.split(","))}
            )
            records.append(record)
    return pd.DataFrame(records)


def _structure(sid: str, pdb_text: str) -> Optional[dict]:
    try:
        return domain_structure(pdb_text.splitlines())
    except Exception as e:  # malformed coordinates; the domain is dropped
        print(f"DSSP failed on {sid}: {e}")
        return None


class ScopeDataset(BaseDataset):
    """SCOPe domains with fold labels.

    ``molecules`` is the domain frame (``mol`` holds the secondary-structure dict from
    :func:`~chebILP.protein_processing.dssp.domain_structure`). ``classes`` holds each fold's
    ``name``, ``sccs`` and ``definition``. Splits keep each fold's superfamilies together where
    the fold allows, else its families (see :meth:`create_splits`).
    """

    DATASET_NAME: str = "scope"

    def __init__(
        self,
        version: str = "2.08",
        base_dir: str = "data",
        min_pos_samples: int = 25,
        workers: int = 8,
    ):
        super().__init__(
            version=version, base_dir=base_dir, min_pos_samples=min_pos_samples
        )
        self.workers = workers
        self.domains_path = os.path.join(self.data_dir, "domains.pkl")
        self.classes_path = os.path.join(self.data_dir, "folds.pkl")
        self.holdout_file = os.path.join(self.processed_dir, "holdout.csv")

    def _parse_file(self, kind: str) -> str:
        return self._raw(f"dir.{kind}.scope.{self.version}-stable.txt")

    def download(self) -> None:
        for name, url in raw_files(self.version).items():
            if not os.path.exists(self._raw(name)):
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, self._raw(name))

    def process_data(self) -> None:
        self.molecules = self._load_or_build_domains()
        self.classes = self._load_or_build_classes()

    def _structures(self) -> dict[str, Optional[dict]]:
        """DSSP structure of every domain in the ASTRAL subset archive by sid, read from the archive without extracting it."""
        structures = {}

        def run(batch, pool):
            sids, texts = zip(*batch)
            structures.update(
                zip(sids, pool.map(_structure, sids, texts, chunksize=16))
            )

        with (
            tarfile.open(self._raw(pdbstyle_archive(self.version)), "r|gz") as tar,
            ProcessPoolExecutor(self.workers) as pool,
        ):
            batch = []
            for member in tqdm.tqdm(tar, desc="DSSP", unit=" files"):
                name = os.path.basename(member.name)
                if member.isfile() and name.endswith(".ent"):
                    batch.append((name[:-4], tar.extractfile(member).read().decode()))
                    if len(batch) == DSSP_BATCH_SIZE:
                        run(batch, pool)
                        batch = []
            if batch:
                run(batch, pool)
        return structures

    def _load_or_build_domains(self) -> pd.DataFrame:
        if os.path.exists(self.domains_path):
            print(f"Loading domains from {self.domains_path}...")
            return pd.read_pickle(self.domains_path)
        cla = parse_cla(self._parse_file("cla"))
        structures = self._structures()
        cla = cla[cla["sid"].isin(structures)]
        print(f"{len(cla)} domains in the ASTRAL <{ASTRAL_SUBSET}% subset")

        records, dropped = [], Counter()
        for row in cla.itertuples(index=False):
            structure = structures[row.sid]
            if structure is None or len(structure["sses"]) < 2:
                dropped["fewer than 2 SSEs"] += 1
                continue
            records.append(
                {
                    "id": domain_id(row.sid),
                    "mol": structure,
                    "sid": row.sid,
                    "pdb": row.pdb,
                    "region": row.region,
                    "sccs": row.sccs,
                    "fold": fold_id(row.sccs),
                    "superfamily": ".".join(row.sccs.split(".")[:3]),
                }
            )
        domains = pd.DataFrame(records).set_index("id")
        domains.index.name = None
        assert domains.index.is_unique, "domain ids collide after sanitising sids"
        print(f"Kept {len(domains)} domains; dropped: {dict(dropped)}")
        domains.to_pickle(self.domains_path)
        return domains

    def _load_or_build_classes(self) -> pd.DataFrame:
        if os.path.exists(self.classes_path):
            return pd.read_pickle(self.classes_path)
        des = parse_des(self._parse_file("des"))
        com = parse_com(self._parse_file("com"))
        folds = {
            fold_id(sccs): (description, sccs, "; ".join(com.get(sunid, [])) or None)
            for sunid, (level, sccs, description) in des.items()
            if level == "cf"
        }
        nodes = sorted(set(self.molecules["fold"]))
        classes = pd.DataFrame(
            [folds[n] for n in nodes],
            columns=["name", "sccs", "definition"],
            index=nodes,
        )
        classes.to_pickle(self.classes_path)
        return classes

    def select_labels(self) -> list[str]:
        """Folds with at least ``min_pos_samples`` domains."""
        counts = self.molecules["fold"].value_counts()
        labels = sorted(counts.index[counts >= self.min_pos_samples])
        self._labeled_df = pd.concat(
            [
                pd.DataFrame(
                    {"id": self.molecules.index, "sample": self.molecules["mol"].values}
                ),
                pd.DataFrame(
                    {lbl: (self.molecules["fold"] == lbl).values for lbl in labels}
                ),
            ],
            axis=1,
        )
        self._write_labels(labels)
        return labels

    def _write_labels(self, labels: list[str]) -> None:
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.labels_file, "w") as f:
            f.write("\n".join(labels) + "\n")
        print(f"Saved {len(labels)} labels to {self.labels_file}")

    def create_splits(
        self,
        labels: list[str],
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Split each fold on its own, at the strictest holdout level that leaves it testable.

        Folds are disjoint, so each fold's domains are split separately: by superfamily (testing
        fold recognition), else by family (testing superfamily generalisation), else per
        domain. A level is used if :func:`assign_groups` gives every split ``MIN_POS_PER_SPLIT``
        positives. Domains of unselected folds are split by superfamily. ``holdout.csv``
        records each label's level; folds failing at domain level are dropped from ``labels.txt``.
        """
        ratios = {
            "train": round(1.0 - val_ratio - test_ratio, 10),
            "validation": val_ratio,
            "test": test_ratio,
        }
        rng = np.random.default_rng(seed)
        d = self.molecules
        group_of = {
            "superfamily": d["superfamily"],
            "family": d["sccs"],
            "domain": d.index.to_series(index=d.index),
        }
        split_of, holdout = {}, {}
        for fold, members in d.groupby("fold").groups.items():
            for level in HOLDOUT_LEVELS if fold in labels else ["superfamily"]:
                sizes = group_of[level][members].value_counts().sort_index()
                assignment = assign_groups(sizes, ratios, rng)
                if (
                    fold not in labels
                    or min_split_size(sizes, assignment) >= MIN_POS_PER_SPLIT
                ):
                    break
            else:
                continue  # untestable even per domain; dropped below
            if fold in labels:
                holdout[fold] = level
            split_of.update(
                (m, assignment[g]) for m, g in group_of[level][members].items()
            )
        # domains of dropped folds still serve as negatives
        for m in d.index.difference(split_of):
            split_of[m] = rng.choice(list(ratios), p=list(ratios.values()))

        dropped = [f for f in labels if f not in holdout]
        if dropped:
            print(
                f"Dropped {len(dropped)} folds with fewer than {MIN_POS_PER_SPLIT} positives "
                f"in some split: {', '.join(dropped)}"
            )
        labels = [f for f in labels if f in holdout]
        self._write_labels(labels)
        holdout = pd.Series({f: holdout[f] for f in labels}, name="holdout")
        holdout.rename_axis("label").to_csv(self.holdout_file)
        print("Holdout levels:", holdout.value_counts().to_dict())

        splits = pd.DataFrame(
            [(m, split_of[m]) for m in d.index], columns=["id", "split"]
        )
        splits.to_csv(self.splits_file, index=False)
        n = splits["split"].value_counts()
        print(
            f"Saved splits to {self.splits_file}  ({n.get('train', 0)} train / "
            f"{n.get('validation', 0)} val / {n.get('test', 0)} test)"
        )
        return splits


def assign_groups(
    sizes: pd.Series, ratios: dict[str, float], rng: np.random.Generator
) -> dict:
    """Assign groups (``sizes``: group -> domain count) to splits, largest first, each to the split furthest below its share."""
    order = sizes.sample(frac=1, random_state=rng).sort_values(
        ascending=False, kind="stable"
    )
    total, filled, out = sizes.sum(), dict.fromkeys(ratios, 0), {}
    for group, n in order.items():
        split = max(ratios, key=lambda s: ratios[s] * total - filled[s])
        out[group] = split
        filled[split] += n
    return out


def min_split_size(sizes: pd.Series, assignment: dict) -> int:
    filled = Counter()
    for group, n in sizes.items():
        filled[assignment[group]] += n
    return min(filled[s] for s in ("train", "validation", "test"))
