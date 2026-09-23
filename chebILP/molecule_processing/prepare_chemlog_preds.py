"""
ChemLog predictions for the validation and test splits, in the tensor format
``ensemble_eval.load_dl_preds()`` reads.

    python -m chebILP.molecule_processing.prepare_chemlog_preds \\
        --chebi_version 251 \\
        --output_dir data/chemlog_v251

Produces, per split:
    <split>_preds_chemlog.npy            float32 (n_molecules x n_classes)
    <split>_preds_chemlog_metadata.json  {"mol_order": [...], "class_labels": [...]}

ChemLog is the rule-based model python-chebifier contributes to its ensemble
(``chebifier.prediction_models.chemlog_predictor.ChemlogAllPredictor``). That class
cannot be imported here — chebifier's ``prediction_models`` package pulls in the neural
predictors — so the same three sub-classifiers are driven directly:

    peptides            chemlog, 'algo' strategy, over 16 peptide classes
    x molecular entity  chemlog_extra, one class per chemical element present
    organo-x compound   chemlog_extra, one class per element bonded to carbon

Columns are the classes a ChemLog rule decides directly, so every cell is a 1.0 or a
0.0 the rules stand behind. NaN appears only for a molecule ChemLog could not read.
Unlike chebifier's ensemble member, no positive is propagated up the ChEBI hierarchy:
a superclass no rule decides gets no column.
"""

import argparse
import json
import os

import networkx as nx
import numpy as np
from rdkit import Chem

# The classes chemlog's peptide classifier decides, from ChemlogPeptidesPredictor minus
# 25696 and 25697: chemlog's resolve_chebi_classes emits 22563 / 36961 for those charge
# cases, so no rule can ever put a molecule in them.
# fmt: off
PEPTIDE_LABELS = [
    "15841", "16670", "24866", "25676", "27369", "46761", "47923", "48030", "48545",
    "60194", "60334", "60466", "64372", "65061", "90799", "155837",
]
# fmt: on


def to_mol(molecule):
    """Kekulise a copy of the molecule, as chemlog's classifiers expect. Mirrors
    chebifier.utils.to_mol, which keeps a molecule that fails to kekulise."""
    if molecule is None:
        return None
    molecule = Chem.Mol(molecule)
    try:
        Chem.Kekulize(molecule)
    except Chem.KekulizeException as e:
        print(f"Failed to kekulise {Chem.MolToSmiles(molecule)}: {e}")
    return molecule


class ChemLogPredictor:
    """The three chemlog classifiers of chebifier's ``chemlog`` ensemble member,
    predicting a ``{chebi_id: 0/1}`` dict per molecule."""

    def __init__(self, chebi_graph: nx.DiGraph, chebi_version: int):
        from chemlog.cli import CLASSIFIERS
        from chemlog_extra.alg_classification.by_element_classification import (
            OrganoXCompoundClassifier,
            XMolecularEntityClassifier,
        )

        # Both classifiers derive their element-to-class mapping from the ChEBI graph and
        # cache it under data/chebi_v<version>/ relative to the working directory.
        self.element_classifiers = [
            XMolecularEntityClassifier(chebi_graph=chebi_graph, chebi_version=chebi_version),
            OrganoXCompoundClassifier(chebi_graph=chebi_graph, chebi_version=chebi_version),
        ]
        self.peptide_classifiers = {key: cls() for key, cls in CLASSIFIERS["algo"].items()}

    @property
    def classes(self) -> list[str]:
        """Every class the rules decide, so that the column layout does not depend on
        which classes a particular split happens to hit."""
        decided = set(PEPTIDE_LABELS)
        for classifier in self.element_classifiers:
            decided.update(classifier.element_class_mapping.values())
        return sorted(decided, key=int)

    def predict_peptides(self, mol) -> dict[str, int]:
        from chemlog.cli import strategy_call

        predicted = strategy_call("algo", self.peptide_classifiers, mol)["chebi_classes"]
        return {label: int(label in predicted) for label in PEPTIDE_LABELS}

    def predict_elements(self, mols: list) -> list[dict[str, int]]:
        merged: list[dict[str, int]] = [dict() for _ in mols]
        for classifier in self.element_classifiers:
            for row, result in zip(merged, classifier.classify(list(mols))):
                row.update({cls: int(hit) for cls, hit in result.items()})
        return merged

    def predict(self, mols: list) -> list[dict[str, int] | None]:
        """One dict per molecule, None where the molecule is missing."""
        kekulized = [to_mol(mol) for mol in mols]
        usable = [i for i, mol in enumerate(kekulized) if mol is not None]

        results: list[dict[str, int] | None] = [None] * len(mols)
        element_preds = self.predict_elements([kekulized[i] for i in usable])
        for i, elements in zip(usable, element_preds):
            results[i] = elements
        for i in usable:
            results[i].update(self.predict_peptides(kekulized[i]))
        return results

    def on_finish(self):
        for classifier in self.peptide_classifiers.values():
            classifier.on_finish()


def to_dense(predictions: list, classes: list[str]) -> np.ndarray:
    """Stack the per-molecule dicts into a float32 matrix over *classes*, NaN where
    ChemLog made no statement."""
    col_of = {cls: idx for idx, cls in enumerate(classes)}

    scores = np.full((len(predictions), len(classes)), np.nan, dtype="float32")
    for row, pred in enumerate(predictions):
        if not pred:
            continue
        for cls, value in pred.items():
            scores[row, col_of[cls]] = value
    return scores


def prepare_chemlog_preds(
    chebi_version: int,
    output_dir: str,
    splits: list[str],
    base_dir: str = "data",
    three_star_only: bool = True,
    min_pos_samples: int = 25,
    batch_size: int = 500,
    limit: int | None = None,
):
    import tqdm

    from chebILP.molecule_processing.data_preparation import ChEBIDataset

    dataset = ChEBIDataset(
        chebi_version=chebi_version,
        three_star_only=three_star_only,
        base_dir=base_dir,
        min_pos_samples=min_pos_samples,
    )
    molecules = dataset.molecules
    splits_df = dataset.load_splits_from_csv()
    splits_df["id"] = splits_df["id"].astype(str)

    predictor = ChemLogPredictor(chebi_graph=dataset.chebi_graph, chebi_version=chebi_version)

    class_labels = predictor.classes
    print(f"ChemLog decides {len(class_labels)} classes")

    os.makedirs(output_dir, exist_ok=True)
    for split in splits:
        split_ids = splits_df.loc[splits_df["split"] == split, "id"]
        mol_order = [i for i in split_ids if i in molecules.index]
        if limit:
            mol_order = mol_order[:limit]
        print(f"\n{split}: {len(mol_order)} molecules")

        predictions: list = []
        for start in tqdm.tqdm(range(0, len(mol_order), batch_size), desc=split):
            batch = mol_order[start : start + batch_size]
            predictions.extend(predictor.predict(molecules.loc[batch, "mol"].tolist()))

        scores = to_dense(predictions, class_labels)
        npy_path = os.path.join(output_dir, f"{split}_preds_chemlog.npy")
        meta_path = os.path.join(output_dir, f"{split}_preds_chemlog_metadata.json")
        np.save(npy_path, scores)
        with open(meta_path, "w") as f:
            json.dump({"mol_order": mol_order, "class_labels": class_labels}, f, indent=2)

        print(f"  Saved: {npy_path}  (shape: {scores.shape}, {int((scores == 1).sum())} positive cells)")
        print(f"  Saved: {meta_path}")

    predictor.on_finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce ChemLog predictions for the validation and test splits."
    )
    parser.add_argument("--chebi_version", "-v", type=int, default=251, help="ChEBI version.")
    parser.add_argument("--base_dir", type=str, default="data", help="Dataset base directory.")
    parser.add_argument("--include_two_star", "-2", action="store_true",
                        help="Include two-star molecules (default: three-star only).")
    parser.add_argument("--min_pos_samples", type=int, default=25,
                        help="Minimum positive samples per class; selects the processed subdirectory.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/chemlog_v<chebi_version>).")
    parser.add_argument("--splits", type=str, nargs="+", default=["validation", "test"],
                        help="Splits to predict on.")
    parser.add_argument("--batch_size", type=int, default=500, help="Molecules per progress step.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only predict the first N molecules per split.")

    args = parser.parse_args()

    prepare_chemlog_preds(
        chebi_version=args.chebi_version,
        output_dir=args.output_dir or os.path.join("data", f"chemlog_v{args.chebi_version}"),
        splits=args.splits,
        base_dir=args.base_dir,
        three_star_only=not args.include_two_star,
        min_pos_samples=args.min_pos_samples,
        batch_size=args.batch_size,
        limit=args.limit,
    )
