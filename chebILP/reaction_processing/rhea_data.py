"""
Rhea reactions labelled with EC classes, laid out like a ChEBI dataset.

The EC hierarchy down to the sub-subclass plays the role of the ChEBI class hierarchy:
each reaction is a leaf node ``r<master id>`` with ``is_a`` edges to the sub-subclass of
each of its EC numbers, which sits under ``ec_<a>_<b>`` -> ``ec_<a>`` -> root ``ec``. EC
nodes carry ``name`` (the ENZYME ``enzclass.txt`` heading) and ``definition`` (heading plus
the IUBMB ExplorEnz notes of the node and its ancestors), as ChEBI nodes do.

Example usage:
    python -m chebILP.reaction_processing.rhea_data --min_pos_samples 25

Writes to ``data/rhea_v<version>/``:
    ec_graph.pkl                     – networkx DiGraph (reactions + EC classes)
    reactions.pkl                    – DataFrame, index = reaction id ``r<master>``
    Rhea<min_pos>/labels.txt         – selected EC class ids
    Rhea<min_pos>/splits.csv         – id,split; all reactions of one EC number share a split
"""

import argparse
import os
import pickle
import re
import tarfile
import urllib.request
from collections import defaultdict
from typing import Optional

import networkx as nx
import pandas as pd
from rdkit import Chem, RDLogger
from chebi_utils import build_labeled_dataset, create_multilabel_splits

RHEA_FTP = "https://ftp.expasy.org/databases/rhea"
RAW_FILES = {
    "rhea2ec.tsv": f"{RHEA_FTP}/tsv/rhea2ec.tsv",
    "rhea-directions.tsv": f"{RHEA_FTP}/tsv/rhea-directions.tsv",
    "rhea-chebi-smiles.tsv": f"{RHEA_FTP}/tsv/rhea-chebi-smiles.tsv",
    "rhea-rxn.tar.gz": f"{RHEA_FTP}/ctfiles/rhea-rxn.tar.gz",
    "enzclass.txt": "https://ftp.expasy.org/databases/enzyme/enzclass.txt",
    "enzyme-data.sql.gz": "https://www.enzyme-database.org/downloads/enzyme-data.sql.gz",
}
EC_ROOT = "ec"
_SQL_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "0": ""}


def ec_node_id(ec_number: str, depth: int = 3) -> str:
    """``3.1.1.4`` -> ``ec_3_1_1`` (the first ``depth`` levels)."""
    return "ec_" + "_".join(ec_number.split(".")[:depth])


def reaction_node_id(master_id) -> str:
    return f"r{master_id}"


def parse_enzclass(path: str) -> dict[str, str]:
    """``enzclass.txt`` headings, keyed by EC node id."""
    headings = {}
    pattern = re.compile(r"^\s*(\d+)\.\s*([\d-]+)\.\s*([\d-]+)\.-\s+(.*?)\s*$")
    with open(path, encoding="latin-1") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            levels = [lvl for lvl in m.group(1, 2, 3) if lvl != "-"]
            headings["ec_" + "_".join(levels)] = m.group(4).rstrip(".")
    return headings


def _sql_rows(values: str) -> list[list]:
    """Field values of a MySQL ``INSERT ... VALUES (...),(...);`` tail."""
    rows, row, i = [], None, 0
    while i < len(values):
        ch = values[i]
        if ch == "(" and row is None:
            row, field, i = [], "", i + 1
            continue
        if row is None:
            i += 1
            continue
        if ch == "'":
            j, buf = i + 1, []
            while values[j] != "'":
                if values[j] == "\\":
                    j += 1
                    buf.append(_SQL_ESCAPES.get(values[j], values[j]))
                else:
                    buf.append(values[j])
                j += 1
            field, i = "".join(buf), j + 1
            continue
        if ch in ",)":
            row.append(None if field == "NULL" else field)
            field = ""
            if ch == ")":
                rows.append(row)
                row = None
            i += 1
            continue
        field += ch
        i += 1
    return rows


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def _fix_double_utf8(text: str) -> str:
    """Undo UTF-8 that was encoded twice (``â€™`` -> ``’``), as in some ExplorEnz notes."""
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def parse_explorenz_notes(path: str) -> dict[str, str]:
    """IUBMB explanatory notes from the ExplorEnz SQL dump's ``class`` table, keyed by EC node id."""
    import gzip

    notes = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("INSERT INTO `class` VALUES"):
                continue
            for _id, cls, sub, subsub, _heading, note, _changed in _sql_rows(line.split("VALUES", 1)[1]):
                if not note or not note.strip():
                    continue
                levels = [lvl for lvl in (cls, sub, subsub) if lvl not in (None, "0")]
                notes["ec_" + "_".join(levels)] = _strip_html(_fix_double_utf8(note))
    return notes


def parse_rxn_participants(tar_path: str, wanted: set[int]) -> dict[int, tuple[list[str], list[str]]]:
    """Left/right participant ids (``CHEBI:x`` / ``POLYMER:x``) of the directional reactions ``wanted``.

    Participants keep the order of the rxn file and are deduplicated, so stoichiometry is lost.
    """
    out = {}
    with tarfile.open(tar_path) as tar:
        for member in tar:
            stem = os.path.basename(member.name)
            if not stem.endswith(".rxn") or int(stem[:-4]) not in wanted:
                continue
            lines = tar.extractfile(member).read().decode().split("\n")
            n_left, n_right = int(lines[4][:3]), int(lines[4][3:6])
            ids = [lines[i + 1].strip() for i, line in enumerate(lines) if line == "$MOL"]
            assert len(ids) == n_left + n_right, stem
            out[int(stem[:-4])] = (list(dict.fromkeys(ids[:n_left])), list(dict.fromkeys(ids[n_left:])))
    return out


class RheaDataset:
    """Rhea reactions with EC labels, exposing the attributes ``ILPProblemBuilder`` reads from ``ChEBIDataset``.

    ``molecules`` is the reaction frame (one row per reaction; ``mol`` holds all participants
    combined, so its atom count sizes the reaction) and ``chebi_graph`` the EC graph.
    """

    def __init__(self, rhea_version: int = 142, base_dir: str = "data", min_pos_samples: int = 25):
        self.rhea_version = rhea_version
        self.min_pos_samples = min_pos_samples
        self.data_dir = os.path.join(base_dir, f"rhea_v{rhea_version}")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, f"Rhea{min_pos_samples}")
        self.graph_path = os.path.join(self.data_dir, "ec_graph.pkl")
        self.reactions_path = os.path.join(self.data_dir, "reactions.pkl")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._download_if_needed()
        if os.path.exists(self.graph_path) and os.path.exists(self.reactions_path):
            with open(self.graph_path, "rb") as f:
                self.chebi_graph: nx.DiGraph = pickle.load(f)
            self.molecules: pd.DataFrame = pd.read_pickle(self.reactions_path)
        else:
            self.molecules, self.chebi_graph = self._build()
        self._labeled_df: Optional[pd.DataFrame] = None

    @property
    def ec_graph(self) -> nx.DiGraph:
        return self.chebi_graph

    def _raw(self, name: str) -> str:
        return os.path.join(self.raw_dir, name)

    def _download_if_needed(self) -> None:
        for name, url in RAW_FILES.items():
            if not os.path.exists(self._raw(name)):
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, self._raw(name))

    def _build(self) -> tuple[pd.DataFrame, nx.DiGraph]:
        RDLogger.DisableLog("rdApp.*")
        rhea2ec = pd.read_csv(self._raw("rhea2ec.tsv"), sep="\t", dtype=str)
        ec_by_master = rhea2ec.groupby("MASTER_ID")["ID"].apply(lambda s: sorted(set(s))).to_dict()
        directions = pd.read_csv(self._raw("rhea-directions.tsv"), sep="\t", dtype=int)
        lr_by_master = dict(zip(directions["RHEA_ID_MASTER"].astype(str), directions["RHEA_ID_LR"]))
        smiles = pd.read_csv(self._raw("rhea-chebi-smiles.tsv"), sep="\t", header=None, names=["chebi", "smiles"])
        smiles_by_chebi = dict(zip(smiles["chebi"], smiles["smiles"]))

        masters = [m for m in ec_by_master if m in lr_by_master]
        participants = parse_rxn_participants(self._raw("rhea-rxn.tar.gz"), {lr_by_master[m] for m in masters})

        mol_cache: dict[str, Optional[Chem.Mol]] = {}

        def to_mol(pid: str) -> Optional[Chem.Mol]:
            if pid not in mol_cache:
                smi = smiles_by_chebi.get(pid)
                mol_cache[pid] = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            return mol_cache[pid]

        records, dropped = [], defaultdict(int)
        for master in masters:
            lr = lr_by_master[master]
            if lr not in participants:
                dropped["no rxn file"] += 1
                continue
            left, right = participants[lr]
            if any(not p.startswith("CHEBI:") for p in left + right):
                dropped["polymer participant"] += 1
                continue
            mols = [to_mol(p) for p in left + right]
            if any(m is None for m in mols):
                dropped["participant without parsable SMILES"] += 1
                continue
            combined = mols[0]
            for m in mols[1:]:
                combined = Chem.CombineMols(combined, m)
            records.append({
                "id": reaction_node_id(master),
                "chebi_id": reaction_node_id(master),
                "mol": combined,
                "reactants": [(p, to_mol(p)) for p in left],
                "products": [(p, to_mol(p)) for p in right],
                "ec_numbers": ec_by_master[master],
                "has_wildcard": any(a.GetAtomicNum() == 0 for m in mols for a in m.GetAtoms()),
                "smiles": ".".join(Chem.MolToSmiles(m) for m in mols[:len(left)]) + ">>"
                          + ".".join(Chem.MolToSmiles(m) for m in mols[len(left):]),
            })
        reactions = pd.DataFrame(records).set_index("id")
        reactions.index.name = None
        print(f"Kept {len(reactions)} of {len(masters)} EC-labelled reactions; dropped: {dict(dropped)}")

        graph = self._build_graph(reactions)
        reactions.to_pickle(self.reactions_path)
        with open(self.graph_path, "wb") as f:
            pickle.dump(graph, f)
        return reactions, graph

    def _build_graph(self, reactions: pd.DataFrame) -> nx.DiGraph:
        headings = parse_enzclass(self._raw("enzclass.txt"))
        notes = parse_explorenz_notes(self._raw("enzyme-data.sql.gz"))
        graph = nx.DiGraph()
        graph.add_node(EC_ROOT, name="enzyme-catalysed reaction", definition=None)

        def add_ec(ec_number: str) -> str:
            parent = EC_ROOT
            for depth in (1, 2, 3):
                node = ec_node_id(ec_number, depth)
                if node not in graph:
                    graph.add_node(node, name=headings.get(node, "EC " + ec_number.rsplit(".", 4 - depth)[0]))
                    graph.add_edge(node, parent, relation="is_a")
                parent = node
            return parent

        for rid, row in reactions.iterrows():
            graph.add_node(rid, name=f"RHEA:{rid[1:]}", smiles=row["smiles"], definition=None)
            for ec in row["ec_numbers"]:
                graph.add_edge(rid, add_ec(ec), relation="is_a")

        for node in graph.nodes:
            if node.startswith("ec_"):
                graph.nodes[node]["definition"] = ec_definition(node, graph, notes)
        return graph

    # ── Public API (mirrors ChEBIDataset) ─────────────────────────────────

    def select_labels(self, output_path: Optional[str] = None) -> list[str]:
        """EC classes with at least ``min_pos_samples`` reactions (the synthetic root excluded)."""
        labeled_df, labels = build_labeled_dataset(self.chebi_graph, self.molecules, min_molecules=self.min_pos_samples)
        labels = [lbl for lbl in labels if lbl.startswith("ec_")]
        self._labeled_df = labeled_df[["chebi_id", "mol"] + labels]
        if output_path:
            with open(output_path, "w") as f:
                f.write("\n".join(labels) + "\n")
            print(f"Saved {len(labels)} labels to {output_path}")
        return labels

    def create_splits(self, labels: list[str], val_ratio=0.2, test_ratio=0.2, seed=42,
                      output_path: Optional[str] = None) -> pd.DataFrame:
        """Stratified split over EC-number groups, so all reactions of one enzyme land in one split.

        Reactions sharing an EC number are often the same transformation on homologous
        substrates; splitting them apart would leak. Groups are the connected components of
        reactions linked by a shared EC number, stratified by the union of their labels.
        """
        if self._labeled_df is None:
            self.select_labels()
        labeled = self._labeled_df.set_index("chebi_id")

        groups = nx.Graph()
        for rid, ecs in self.molecules["ec_numbers"].items():
            groups.add_node(rid)
            for ec in ecs:
                groups.add_edge(rid, "EC:" + ec)
        components = [sorted(n for n in comp if n.startswith("r")) for comp in nx.connected_components(groups)]
        group_df = pd.DataFrame(
            [{"chebi_id": str(i), "mol": None, **labeled.loc[members, labels].any().to_dict()}
             for i, members in enumerate(components)]
        )
        raw = create_multilabel_splits(group_df, label_start_col=2, train_ratio=round(1 - val_ratio - test_ratio, 10),
                                       val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        records = [(rid, split) for split, df in raw.items()
                   for gid in df["chebi_id"] for rid in components[int(gid)]]
        splits_df = pd.DataFrame(records, columns=["id", "split"])
        if output_path:
            splits_df.to_csv(output_path, index=False)
            n = splits_df["split"].value_counts()
            print(f"Saved splits to {output_path} ({n.get('train', 0)} train / "
                  f"{n.get('validation', 0)} val / {n.get('test', 0)} test reactions, {len(components)} EC groups)")
        return splits_df

    def load_splits_from_csv(self) -> pd.DataFrame:
        splits_path = os.path.join(self.processed_dir, "splits.csv")
        if not os.path.exists(splits_path):
            raise FileNotFoundError(f"Splits file not found: {splits_path}. "
                                    f"Run `python -m chebILP.reaction_processing.rhea_data` to create it.")
        return pd.read_csv(splits_path, dtype={"id": str})

    @classmethod
    def prepare(cls, rhea_version: int = 142, base_dir: str = "data", min_pos_samples: int = 25,
                val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42) -> "RheaDataset":
        dataset = cls(rhea_version=rhea_version, base_dir=base_dir, min_pos_samples=min_pos_samples)
        labels = dataset.select_labels(output_path=os.path.join(dataset.processed_dir, "labels.txt"))
        dataset.create_splits(labels, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
                              output_path=os.path.join(dataset.processed_dir, "splits.csv"))
        return dataset


def ec_definition(node: str, graph: nx.DiGraph, notes: dict[str, str]) -> str:
    """Heading path from the EC class down to ``node``, followed by the IUBMB notes along that path."""
    path = [node]
    while path[-1] != EC_ROOT:
        path.append(next(graph.successors(path[-1])))
    path = path[-2::-1]
    label = lambda n: "EC " + n[3:].replace("_", ".")
    text = " > ".join(f"{label(n)}: {graph.nodes[n]['name']}" for n in path) + "."
    for n in path:
        if n in notes:
            text += f" {label(n)}: {notes[n]}"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rhea_version", type=int, default=142)
    parser.add_argument("--base_dir", default="data")
    parser.add_argument("--min_pos_samples", type=int, default=25)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    RheaDataset.prepare(args.rhea_version, args.base_dir, args.min_pos_samples, args.val_ratio, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
