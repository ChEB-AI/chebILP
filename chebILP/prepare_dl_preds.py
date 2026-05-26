"""
EXP-006: Extract DL predictions to numpy format for ensemble evaluation.

Run this script ONCE in an environment with torch installed:

    python -m chebILP prepare_dl_preds \\
        --preds_file data/preds/chebi25_3STAR_v248_ul8u3pnm_epoch=198_val.pt \\
        --checkpoint  data/checkpoints/chebi25_3STAR_v248_ul8u3pnm_epoch=198.ckpt \\
        --splits_csv  data/chebi_v248/splits_chebi248.csv \\
        --target_classes data/ilp_classes.txt \\
        --output_dir  data/preds

Produces:
    val_preds_chebi25.npy          float32 array (n_val_samples × n_target_classes)
    val_preds_chebi25_metadata.json  {"mol_order": [...], "class_labels": [...]}

The metadata file describes row/column ordering so the .npy file can be loaded
without torch (numpy only) by ensemble_eval.load_dl_preds().

Tensor layout in the .pt file:
  rows  = validation molecules in splits_csv order
  cols  = classification_labels stored in the checkpoint (ckpt['classification_labels'])
"""

import json
import os


def extract_dl_preds(
    preds_pt_path: str,
    checkpoint_path: str,
    splits_csv_path: str,
    output_file: str,
    subset: str = "validation",
):
    """
    Load the .pt prediction tensor, filter to target_class_ids, and save as
    numpy + JSON metadata.

    Args:
        preds_pt_path:    path to the val .pt tensor
        checkpoint_path:  path to the .ckpt file (contains classification_labels)
        splits_csv_path:  path to the splits CSV
        target_class_ids: ChEBI class IDs to keep (all others are discarded)
        output_file:      path to save the numpy array and metadata JSON
    """
    import torch
    import numpy as np

    print("Loading prediction tensor...")
    preds = torch.load(preds_pt_path, weights_only=False)
    print(f"  shape: {preds.shape}  dtype: {preds.dtype}")

    print("Loading classification labels from checkpoint...")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    all_labels: list[str] = ckpt["classification_labels"]
    print(f"  {len(all_labels)} classes in model")

    print(f"Reading {subset} molecule order from splits CSV...")
    val_mol_order = []
    with open(splits_csv_path, "r") as f:
        for line in f.readlines()[1:]:
            mol_id, split = line.strip().split(",")
            if split == subset:
                val_mol_order.append(mol_id)
    print(f"  {len(val_mol_order)} {subset} molecules")

    assert len(val_mol_order) == preds.shape[0], (
        f"Row count mismatch: {len(val_mol_order)} mol IDs vs {preds.shape[0]} tensor rows"
    )
    
    print(f"  Extracting predictions for {len(all_labels)} classes")

    arr = preds.detach().cpu().numpy().astype("float32")

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    npy_path = output_file
    meta_path = output_file.replace(".npy", "_metadata.json")

    np.save(npy_path, arr)
    with open(meta_path, "w") as f:
        json.dump({"mol_order": val_mol_order, "class_labels": all_labels}, f, indent=2)

    print(f"Saved: {npy_path}  (shape: {arr.shape})")
    print(f"Saved: {meta_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract DL predictions to numpy format for ensemble evaluation.")
    parser.add_argument("--preds_file", type=str, required=True, help="Path to the .pt predictions tensor.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.ckpt).")
    parser.add_argument("--splits_csv", type=str, required=True, help="Path to the splits CSV file.")
    parser.add_argument("--output_file", type=str, default=os.path.join("data", "preds", "val_preds_chebi25.npy"), help="Path to save the numpy array (val_preds_chebi25.npy) and metadata JSON.")
    parser.add_argument("--subset", type=str, default="validation", help="Subset of molecules to extract predictions for.")

    args = parser.parse_args()

    extract_dl_preds(
        args.preds_file,
        args.checkpoint,
        args.splits_csv,
        args.output_file,
        subset=args.subset
    )