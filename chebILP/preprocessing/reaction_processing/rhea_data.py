"""
Rhea reactions labelled with EC classes.

Samples are reactions. Each is labelled with the EC classes ``ec_<a>``, ``ec_<a>_<b>`` and
``ec_<a>_<b>_<c>`` of its EC numbers. EC classes carry a ``name`` (the IUBMB ExplorEnz heading)
and a ``definition`` (the headings plus the ExplorEnz notes of the class and its ancestors).

Example usage:
    python -m chebILP.preprocessing.reaction_processing.rhea_data --version 142

Writes to ``data/rhea_v<version>/``:
    reactions.pkl                    – DataFrame, index = reaction id ``r<master>``
    ec_classes.pkl                   – DataFrame of EC class names and definitions
    processed_<min_pos>/labels.txt   – selected EC class ids
    processed_<min_pos>/splits.csv   – id,split
"""

import os
import re
import gzip
import html
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
from chebILP.preprocessing.base_dataset import BaseDataset
from rdkit import Chem, RDLogger
from chebi_utils.read_molecule import parse_molblock
from chebi_utils.sdf_extractor import _iter_sdf_records, _parse_sdf_record

RHEA_FTP = "https://ftp.expasy.org/databases/rhea"
RAW_FILES = {
    "rhea.rdf.gz": f"{RHEA_FTP}/rdf/rhea.rdf.gz",
    "rhea.sdf.gz": f"{RHEA_FTP}/ctfiles/rhea.sdf.gz",
    "enzyme-data.xml.gz": "https://www.enzyme-database.org/downloads/enzyme-data.xml.gz",
}


def ec_node_id(ec_number: str, depth: int = 3) -> str:
    """``3.1.1.4`` -> ``ec_3_1_1`` (the first ``depth`` levels)."""
    return "ec_" + "_".join(ec_number.split(".")[:depth])


def reaction_node_id(master_id) -> str:
    return f"r{master_id}"


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text))).strip()


def _fix_double_utf8(text: str) -> str:
    """Undo UTF-8 that was encoded twice (``â€™`` -> ``’``), as in some ExplorEnz notes."""
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def parse_explorenz_classes(path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Headings and IUBMB notes from the ExplorEnz XML dump's ``class`` table, keyed by EC node id."""
    headings, notes = {}, {}
    with gzip.open(path) as f:
        for _, elem in ET.iterparse(f):
            if elem.tag != "table_data":
                continue
            if elem.get("name") == "class":
                for row in elem.iter("row"):
                    fields = {fld.get("name"): fld.text or "" for fld in row}
                    levels = [
                        fields[k]
                        for k in ("class", "subclass", "subsubclass")
                        if fields[k] != "0"
                    ]
                    node = "ec_" + "_".join(levels)
                    headings[node] = _strip_html(fields["heading"]).rstrip(".")
                    if fields["note"].strip():
                        notes[node] = _strip_html(_fix_double_utf8(fields["note"]))
            elem.clear()
    return headings, notes


_RDF_ABOUT = re.compile(
    r'<rdf:Description rdf:about="http://rdf\.rhea-db\.org/([^"]+)"'
)
_RDF_ACCESSION = re.compile(r"<rh:accession>([^<]+)</rh:accession>")
_RDF_REF = re.compile(
    r'<rh:(contains|compound|reactivePart) rdf:resource="http://rdf\.rhea-db\.org/([^"]+)"'
)
_RDF_CHEBI = re.compile(
    r'<rh:chebi rdf:resource="http://purl\.obolibrary\.org/obo/CHEBI_(\d+)"'
)
_RDF_EC = re.compile(r'<rh:ec rdf:resource="http://purl\.uniprot\.org/enzyme/([^"]+)"')


def parse_rdf_reactions(
    rdf_gz_path: str,
) -> tuple[dict[str, tuple[list[str], list[str]]], dict[str, list[str]]]:
    """Participants and EC numbers of every Rhea master reaction, both keyed by master id.

    Participants are the left/right accessions (``CHEBI:x`` / ``POLYMER:x``). A generic compound
    (``GENERIC:x``) is replaced by the ChEBI entries of all its reactive parts, which is what
    ``rhea.sdf.gz`` holds structures for. They keep file order and are deduplicated, so
    stoichiometry is lost. Masters without an EC number are absent from the second dict.
    """
    accession, reactive_parts, rp_chebi = {}, defaultdict(list), {}
    side_participants, participant_compound = defaultdict(list), {}
    ec_numbers = defaultdict(set)
    about = None
    with gzip.open(rdf_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if m := _RDF_ABOUT.search(line):
                about = m.group(1)
            elif m := _RDF_EC.search(line):
                ec_numbers[about].add(m.group(1))
            elif m := _RDF_ACCESSION.search(line):
                accession[about] = m.group(1)
            elif m := _RDF_REF.search(line):
                kind, target = m.groups()
                if kind == "contains":
                    side_participants[about].append(target)
                elif kind == "compound":
                    participant_compound[about] = target
                else:
                    reactive_parts[about].append(target)
            elif (m := _RDF_CHEBI.search(line)) and "_rp" in about:
                rp_chebi[about] = f"CHEBI:{m.group(1)}"

    def resolve(participant: str) -> list[str]:
        compound = participant_compound[participant]
        if accession[compound].startswith("GENERIC:"):
            return [rp_chebi[rp] for rp in reactive_parts[compound]]
        return [accession[compound]]

    out = {}
    for name in side_participants:
        if name.endswith("_L") and name[:-2] + "_R" in side_participants:
            master = name[:-2]
            out[master] = tuple(
                list(
                    dict.fromkeys(
                        c for p in side_participants[master + s] for c in resolve(p)
                    )
                )
                for s in ("_L", "_R")
            )
    return out, {k: sorted(v) for k, v in ec_numbers.items()}


def load_sdf_mols(sdf_gz_path: str) -> dict[str, Chem.Mol]:
    """RDKit molecules of ``rhea.sdf.gz`` without explicit hydrogens, keyed by ``ACCESSION``
    (``CHEBI:x`` / ``POLYMER:x``)."""
    mols = {}
    for record in _iter_sdf_records(sdf_gz_path):
        props, molblock = _parse_sdf_record(record)
        mol = parse_molblock(molblock, props["ACCESSION"])
        if mol is not None:
            mols[props["ACCESSION"]] = Chem.RemoveHs(mol, sanitize=False)
    return mols


class RheaDataset(BaseDataset):
    """Rhea reactions with EC labels.

    ``molecules`` is the reaction frame (one row per reaction; ``mol`` holds all participants
    combined, so its atom count sizes the reaction). A reaction is labelled with every EC class
    (depth 1-3) of its EC numbers. ``classes`` holds each EC class's ``name`` and ``definition``.
    """

    DATASET_NAME: str = "rhea"

    def __init__(
        self, version: str = "142", base_dir: str = "data", min_pos_samples: int = 25
    ):
        super().__init__(
            version=version, base_dir=base_dir, min_pos_samples=min_pos_samples
        )
        self.reactions_path = os.path.join(self.data_dir, "reactions.pkl")
        self.classes_path = os.path.join(self.data_dir, "ec_classes.pkl")

    def _raw(self, name: str) -> str:
        return os.path.join(self.raw_dir, name)

    def download(self) -> None:
        for name, url in RAW_FILES.items():
            if not os.path.exists(self._raw(name)):
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, self._raw(name))

    def process_data(self) -> None:
        self.molecules = self._load_or_build_reactions()
        self.classes = self._load_or_build_classes()

    def _load_or_build_reactions(self) -> pd.DataFrame:
        if os.path.exists(self.reactions_path):
            print(f"Loading reactions from {self.reactions_path}...")
            return pd.read_pickle(self.reactions_path)
        RDLogger.DisableLog("rdApp.*")
        participants, ec_by_master = parse_rdf_reactions(self._raw("rhea.rdf.gz"))
        mols_by_acc = load_sdf_mols(self._raw("rhea.sdf.gz"))

        records, dropped = [], defaultdict(int)
        for master, ec_numbers in ec_by_master.items():
            if master not in participants:
                dropped["not in rdf"] += 1
                continue
            left, right = participants[master]
            if any(not p.startswith("CHEBI:") for p in left + right):
                dropped["polymer participant"] += 1
                continue
            mols = [mols_by_acc.get(p) for p in left + right]
            if any(m is None for m in mols):
                dropped["participant without structure"] += 1
                continue
            combined = mols[0]
            for m in mols[1:]:
                combined = Chem.CombineMols(combined, m)
            records.append(
                {
                    "id": reaction_node_id(master),
                    "mol": combined,
                    "reactants": list(zip(left, mols[: len(left)])),
                    "products": list(zip(right, mols[len(left) :])),
                    "ec_numbers": ec_numbers,
                    "has_wildcard": any(
                        a.GetAtomicNum() == 0 for m in mols for a in m.GetAtoms()
                    ),
                    "smiles": ".".join(Chem.MolToSmiles(m) for m in mols[: len(left)])
                    + ">>"
                    + ".".join(Chem.MolToSmiles(m) for m in mols[len(left) :]),
                }
            )
        reactions = pd.DataFrame(records).set_index("id")
        reactions.index.name = None
        print(
            f"Kept {len(reactions)} of {len(ec_by_master)} EC-labelled reactions; dropped: {dict(dropped)}"
        )
        reactions.to_pickle(self.reactions_path)
        return reactions

    def _load_or_build_classes(self) -> pd.DataFrame:
        if os.path.exists(self.classes_path):
            return pd.read_pickle(self.classes_path)
        headings, notes = parse_explorenz_classes(self._raw("enzyme-data.xml.gz"))
        nodes = sorted(
            {c for ecs in self.molecules["ec_numbers"] for c in ec_classes(ecs)}
        )
        classes = pd.DataFrame(
            {
                "name": [headings.get(n, ec_label(n)) for n in nodes],
                "definition": [ec_definition(n, headings, notes) for n in nodes],
            },
            index=nodes,
        )
        classes.to_pickle(self.classes_path)
        return classes

    def select_labels(self) -> list[str]:
        """EC classes with at least ``min_pos_samples`` reactions."""
        reaction_classes = self.molecules["ec_numbers"].map(ec_classes)
        counts = pd.Series(
            [c for classes in reaction_classes for c in classes]
        ).value_counts()
        labels = sorted(counts.index[counts >= self.min_pos_samples])

        self._labeled_df = pd.concat(
            [
                pd.DataFrame(
                    {"id": self.molecules.index, "sample": self.molecules["mol"].values}
                ),
                pd.DataFrame(
                    [
                        [lbl in classes for lbl in labels]
                        for classes in reaction_classes
                    ],
                    columns=labels,
                ),
            ],
            axis=1,
        )
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.labels_file, "w") as f:
            f.write("\n".join(labels) + "\n")
        print(f"Saved {len(labels)} labels to {self.labels_file}")
        return labels


def ec_classes(ec_numbers: list[str]) -> set[str]:
    """EC class ids (depth 1-3) of a reaction's EC numbers: ``["3.1.1.4"]`` -> ``{ec_3, ec_3_1, ec_3_1_1}``."""
    return {
        ec_node_id(ec, depth)
        for ec in ec_numbers
        for depth in (1, 2, 3)
        if "-" not in ec.split(".")[:depth]
    }


def ec_label(node: str) -> str:
    return "EC " + node[3:].replace("_", ".")


def ec_definition(node: str, headings: dict[str, str], notes: dict[str, str]) -> str:
    """Heading path from the EC class down to ``node``, followed by the IUBMB notes along that path."""
    levels = node[3:].split("_")
    path = ["ec_" + "_".join(levels[: i + 1]) for i in range(len(levels))]
    text = (
        " > ".join(
            f"{ec_label(n)}: {headings[n]}" if n in headings else ec_label(n)
            for n in path
        )
        + "."
    )
    for n in path:
        if n in notes:
            text += f" {ec_label(n)}: {notes[n]}"
    return text
