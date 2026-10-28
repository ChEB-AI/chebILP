"""
SCOPe protein domains labelled with their fold, laid out like a ChEBI dataset.

Each domain is a leaf ``d<sid>`` with an ``is_a`` edge to its fold ``scop_<cl>_<cf>``, which
sits under its class ``scop_<cl>`` and the root ``scop``. Fold and class nodes carry ``name``
(the ``dir.des`` description) and ``definition`` (the ``dir.com`` comments, e.g. "core: 6
helices; folded leaf, partly opened").

Only human-curated domains of the true classes a–g are kept. SCOPe has no flag for that,
so a domain is dropped when its family or protein node is an "automated matches" node, or
its own ``dir.com`` entry reads "automated match to d…". Examples are the curated domains
of the ASTRAL <40% identity subset.

Example usage:
    python -m chebILP.protein_processing.scope_data --min_pos_samples 25

Writes to ``data/scope_<version>/``:
    scope_graph.pkl            – networkx DiGraph (domains + folds + classes)
    domains.pkl                – DataFrame, index = domain id; ``mol`` holds the DSSP structure
    Scope<min_pos>/labels.txt  – selected fold/class ids
    Scope<min_pos>/splits.csv  – id,split; all domains of one superfamily share a split
"""

import argparse
import os
import pickle
import re
import tarfile
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import networkx as nx
import pandas as pd
import tqdm
from chebi_utils import build_labeled_dataset, create_multilabel_splits

from chebILP.protein_processing.dssp import domain_structure

SCOPE_URL = "https://scop.berkeley.edu/downloads"
SCOP_ROOT = "scop"
CURATED_CLASSES = "abcdefg"
AUTOMATED_NODE = "automated matches"
AUTOMATED_COMMENT = "automated match to"


def raw_files(version: str, subset: int) -> dict[str, str]:
    files = {f"dir.{kind}.scope.{version}-stable.txt": f"{SCOPE_URL}/parse/dir.{kind}.scope.{version}-stable.txt"
             for kind in ("des", "cla", "com")}
    name = f"pdbstyle-sel-gs-bib-{subset}-{version}.tgz"
    files[name] = f"{SCOPE_URL}/pdbstyle/{name}"
    return files


def node_id(sccs: str) -> str:
    """``c.1.8.4`` -> ``scop_c_1`` (fold); ``c`` -> ``scop_c`` (class)."""
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
            record.update({k: int(v) for k, v in (kv.split("=") for kv in ancestors.split(","))})
            records.append(record)
    return pd.DataFrame(records)


def curated_domains(cla: pd.DataFrame, des: dict, com: dict) -> pd.DataFrame:
    """Domains of classes a–g with no automated-match marker on their family, protein or own entry."""
    counts = {"all": len(cla)}
    cla = cla[cla["sccs"].str[0].isin(list(CURATED_CLASSES))]
    counts["classes a-g"] = len(cla)
    automated_node = lambda sunid: des[sunid][2] == AUTOMATED_NODE
    cla = cla[~cla["fa"].map(automated_node) & ~cla["dm"].map(automated_node)]
    counts["no automated family/protein"] = len(cla)
    cla = cla[~cla["px"].map(lambda px: any(c.startswith(AUTOMATED_COMMENT) for c in com.get(px, [])))]
    counts["no automated match comment"] = len(cla)
    print("Curation filter:", counts)
    return cla


def _structure(path: str) -> Optional[dict]:
    try:
        return domain_structure(path)
    except Exception as e:  # malformed coordinates; the domain is dropped
        print(f"DSSP failed on {path}: {e}")
        return None


class ScopeDataset:
    """Curated SCOPe domains with fold labels, exposing the attributes ``ILPProblemBuilder`` reads from ``ChEBIDataset``.

    ``molecules`` is the domain frame (``mol`` holds the secondary-structure dict from
    :func:`~chebILP.protein_processing.dssp.domain_structure`) and ``chebi_graph`` the fold graph.
    """

    def __init__(self, scope_version: str = "2.08", base_dir: str = "data", min_pos_samples: int = 25,
                 subset: int = 40, workers: int = 8):
        self.scope_version = scope_version
        self.subset = subset
        self.min_pos_samples = min_pos_samples
        self.workers = workers
        self.data_dir = os.path.join(base_dir, f"scope_{scope_version.replace('.', '')}")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, f"Scope{min_pos_samples}")
        self.graph_path = os.path.join(self.data_dir, "scope_graph.pkl")
        self.domains_path = os.path.join(self.data_dir, "domains.pkl")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        if os.path.exists(self.graph_path) and os.path.exists(self.domains_path):
            with open(self.graph_path, "rb") as f:
                self.chebi_graph: nx.DiGraph = pickle.load(f)
            self.molecules: pd.DataFrame = pd.read_pickle(self.domains_path)
        else:
            self._download_if_needed()
            self.molecules, self.chebi_graph = self._build()
        self._labeled_df: Optional[pd.DataFrame] = None

    def _raw(self, name: str) -> str:
        return os.path.join(self.raw_dir, name)

    def _download_if_needed(self) -> None:
        for name, url in raw_files(self.scope_version, self.subset).items():
            if not os.path.exists(self._raw(name)):
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, self._raw(name))

    def _extract(self, sids: set[str]) -> dict[str, str]:
        """Coordinate files of ``sids`` present in the ASTRAL subset archive, extracted once into ``raw/pdbstyle``."""
        out_dir = self._raw("pdbstyle")
        tgz = self._raw(f"pdbstyle-sel-gs-bib-{self.subset}-{self.scope_version}.tgz")
        paths = {}
        if os.path.isdir(out_dir):
            for root, _, files in os.walk(out_dir):
                paths.update({f[:-4]: os.path.join(root, f) for f in files if f[:-4] in sids})
            return paths
        os.makedirs(out_dir)
        with tarfile.open(tgz) as tar:
            for member in tqdm.tqdm(tar, desc="Extracting coordinates"):
                sid = os.path.basename(member.name)[:-4]
                if member.isfile() and member.name.endswith(".ent") and sid in sids:
                    path = os.path.join(out_dir, sid[2:4], sid + ".ent")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(tar.extractfile(member).read())
                    paths[sid] = path
        return paths

    def _build(self) -> tuple[pd.DataFrame, nx.DiGraph]:
        v = self.scope_version
        des = parse_des(self._raw(f"dir.des.scope.{v}-stable.txt"))
        com = parse_com(self._raw(f"dir.com.scope.{v}-stable.txt"))
        cla = curated_domains(parse_cla(self._raw(f"dir.cla.scope.{v}-stable.txt")), des, com)

        paths = self._extract(set(cla["sid"]))
        cla = cla[cla["sid"].isin(paths)]
        print(f"{len(cla)} curated domains in the ASTRAL <{self.subset}% subset")

        ordered = list(cla["sid"])
        with ProcessPoolExecutor(self.workers) as pool:
            structures = list(tqdm.tqdm(pool.map(_structure, [paths[s] for s in ordered], chunksize=16),
                                        total=len(ordered), desc="DSSP"))
        records, dropped = [], Counter()
        for row, structure in zip(cla.itertuples(index=False), structures):
            if structure is None or len(structure["sses"]) < 2:
                dropped["fewer than 2 SSEs"] += 1
                continue
            did = domain_id(row.sid)
            records.append({
                "id": did, "chebi_id": did, "mol": structure, "sid": row.sid, "pdb": row.pdb,
                "region": row.region, "sccs": row.sccs, "fold": node_id(row.sccs),
                "superfamily": ".".join(row.sccs.split(".")[:3]), "family": row.sccs,
                "protein": des[row.dm][2], "species": des[row.sp][2],
            })
        domains = pd.DataFrame(records).set_index("id")
        domains.index.name = None
        assert domains.index.is_unique, "domain ids collide after sanitising sids"
        print(f"Kept {len(domains)} domains; dropped: {dict(dropped)}")

        graph = self._build_graph(domains, des, com)
        domains.to_pickle(self.domains_path)
        with open(self.graph_path, "wb") as f:
            pickle.dump(graph, f)
        return domains, graph

    def _build_graph(self, domains: pd.DataFrame, des: dict, com: dict) -> nx.DiGraph:
        by_sccs = {(level, sccs): sunid for sunid, (level, sccs, _) in des.items() if level in ("cl", "cf")}
        graph = nx.DiGraph()
        graph.add_node(SCOP_ROOT, name="protein domain", definition=None)

        def add_node(level: str, sccs: str, parent: str) -> str:
            node = node_id(sccs)
            if node not in graph:
                sunid = by_sccs[(level, sccs)]
                comments = com.get(sunid, [])
                graph.add_node(node, name=des[sunid][2], sccs=sccs,
                               definition="; ".join(comments) if comments else None)
                graph.add_edge(node, parent, relation="is_a")
            return node

        for did, row in domains.iterrows():
            cl, cf = row["sccs"].split(".")[:2]
            cls = add_node("cl", cl, SCOP_ROOT)
            fold = add_node("cf", f"{cl}.{cf}", cls)
            graph.add_node(did, name=row["sid"], definition=None)
            graph.add_edge(did, fold, relation="is_a")
        return graph

    # ── Public API (mirrors ChEBIDataset) ─────────────────────────────────

    def select_labels(self, output_path: Optional[str] = None) -> list[str]:
        """Folds and classes with at least ``min_pos_samples`` domains (the synthetic root excluded)."""
        labeled_df, labels = build_labeled_dataset(self.chebi_graph, self.molecules, min_molecules=self.min_pos_samples)
        labels = [lbl for lbl in labels if lbl.startswith("scop_")]
        self._labeled_df = labeled_df[["chebi_id", "mol"] + labels]
        if output_path:
            self._write_labels(labels, output_path)
        return labels

    @staticmethod
    def _write_labels(labels: list[str], output_path: str) -> None:
        with open(output_path, "w") as f:
            f.write("\n".join(labels) + "\n")
        print(f"Saved {len(labels)} labels to {output_path}")

    def create_splits(self, labels: list[str], val_ratio=0.2, test_ratio=0.2, seed=42,
                      output_path: Optional[str] = None) -> pd.DataFrame:
        """Stratified split over superfamilies, so a test domain has no homologue in train.

        Domains of one superfamily share an evolutionary origin; a fold is the structure
        common to several superfamilies, so holding whole superfamilies out tests fold
        recognition rather than homology.
        """
        if self._labeled_df is None:
            self.select_labels()
        labeled = self._labeled_df.set_index("chebi_id")
        members = self.molecules.groupby("superfamily").groups
        groups = sorted(members)
        group_df = pd.DataFrame(
            [{"chebi_id": str(i), "mol": None, **labeled.loc[list(members[sf]), labels].any().to_dict()}
             for i, sf in enumerate(groups)]
        )
        raw = create_multilabel_splits(group_df, label_start_col=2, train_ratio=round(1 - val_ratio - test_ratio, 10),
                                       val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        records = [(did, split) for split, df in raw.items()
                   for gid in df["chebi_id"] for did in members[groups[int(gid)]]]
        splits_df = pd.DataFrame(records, columns=["id", "split"])
        if output_path:
            splits_df.to_csv(output_path, index=False)
            n = splits_df["split"].value_counts()
            print(f"Saved splits to {output_path} ({n.get('train', 0)} train / "
                  f"{n.get('validation', 0)} val / {n.get('test', 0)} test domains, {len(groups)} superfamilies)")
        return splits_df

    def labels_in_every_split(self, labels: list[str], splits: pd.DataFrame, min_per_split: int = 3) -> list[str]:
        """The labels with at least ``min_per_split`` positives in each split.

        Grouping by superfamily puts a fold with one or two superfamilies wholly into one or two
        splits; such a fold cannot be trained and tested.
        """
        split_of = dict(zip(splits["id"], splits["split"]))
        labeled = self._labeled_df.set_index("chebi_id")
        kept, dropped = [], []
        for label in labels:
            n = Counter(split_of[d] for d in labeled.index[labeled[label]] if d in split_of)
            (kept if all(n[s] >= min_per_split for s in ("train", "validation", "test")) else dropped).append(label)
        if dropped:
            print(f"Dropped {len(dropped)} labels with fewer than {min_per_split} positives in some split: {', '.join(dropped)}")
        return kept

    def load_splits_from_csv(self) -> pd.DataFrame:
        splits_path = os.path.join(self.processed_dir, "splits.csv")
        if not os.path.exists(splits_path):
            raise FileNotFoundError(f"Splits file not found: {splits_path}. "
                                    f"Run `python -m chebILP.protein_processing.scope_data` to create it.")
        return pd.read_csv(splits_path, dtype={"id": str})

    @classmethod
    def prepare(cls, scope_version: str = "2.08", base_dir: str = "data", min_pos_samples: int = 25,
                val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42, min_per_split: int = 3,
                workers: int = 8) -> "ScopeDataset":
        dataset = cls(scope_version=scope_version, base_dir=base_dir, min_pos_samples=min_pos_samples, workers=workers)
        labels = dataset.select_labels()
        splits = dataset.create_splits(labels, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
                                       output_path=os.path.join(dataset.processed_dir, "splits.csv"))
        labels = dataset.labels_in_every_split(labels, splits, min_per_split)
        dataset._write_labels(labels, os.path.join(dataset.processed_dir, "labels.txt"))
        return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scope_version", default="2.08")
    parser.add_argument("--base_dir", default="data")
    parser.add_argument("--min_pos_samples", type=int, default=25)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_per_split", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    ScopeDataset.prepare(args.scope_version, args.base_dir, args.min_pos_samples, args.val_ratio, args.test_ratio,
                         args.seed, args.min_per_split, args.workers)


if __name__ == "__main__":
    main()
