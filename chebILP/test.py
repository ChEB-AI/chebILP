

import json
import os
import time
from chebILP.ilp_path_manager import get_bk_path, get_exs_path
from typing import Literal, Optional

import numpy as np
import pandas as pd

def build_ilp_preds_tensor(
    programs: dict[str, str],
    molecules_df: pd.DataFrame,
    mol_order: list[str],
    output_npy_path: str,
    output_meta_path: str,
) -> pd.DataFrame:
    """
    Evaluate ILP programs against every molecule in mol_order and save the
    result as a boolean numpy tensor.

    Builds background knowledge from scratch using build_background_chemlog
    (not from existing bk.pl problem files). Intended for generating full
    split prediction tensors used by ensemble_construct.

    Args:
        programs:         class_id → Prolog program string (only classes with
                          a learned program; null programs must be excluded).
        molecules_df:     DataFrame with 'mol' column; index = molecule IDs
                          (string ChEBI IDs). Built from the raw SDF via
                          extract_molecules().
        mol_order:        ordered list of molecule IDs to evaluate.
        output_npy_path:  destination .npy file for the boolean tensor.
        output_meta_path: destination JSON file for mol_order / class_labels.

    Returns a boolean DataFrame (index=mol_order, columns=sorted class IDs).
    """
    import tqdm
    from chebILP.clingo_eval import evaluate_with_clingo
    from chebILP.mol2ilp import build_background_chemlog

    ilp_class_ids = sorted(programs.keys())
    n_mols = len(mol_order)
    n_classes = len(ilp_class_ids)
    print(f"Building ILP predictions tensor: {n_mols} molecules × {n_classes} classes")

    print("Building background facts for all molecules...")
    all_bg_facts: list[str] = []
    present_mol_ids: list[str] = []
    missing = 0
    for mol_id in tqdm.tqdm(mol_order, desc="Background knowledge"):
        mol_row = molecules_df[molecules_df.index == mol_id].dropna(subset=["mol"])
        if mol_row.empty:
            missing += 1
        else:
            lines, _ = build_background_chemlog(mol_row)
            all_bg_facts.extend(lines)
            present_mol_ids.append(mol_id)
    if missing:
        print(f"  {missing} molecules missing from molecules_df — predicted False")

    all_rules = [line for prog in programs.values() for line in prog.split("\n") if line.strip()]
    all_target_labels = [f"chebi_{cls_id}" for cls_id in ilp_class_ids]

    print("Running Clingo for all ILP classes...")
    try:
        positives = evaluate_with_clingo(all_rules, all_bg_facts, all_target_labels, present_mol_ids)
    except Exception as e:
        print(f"Clingo evaluation failed: {e}")
        positives = {}

    mol_idx = {mol_id: i for i, mol_id in enumerate(mol_order)}
    tensor = np.zeros((n_mols, n_classes), dtype=bool)
    for col_idx, cls_id in enumerate(ilp_class_ids):
        label = f"chebi_{cls_id}"
        for mol_id in positives.get(label, []):
            if mol_id in mol_idx:
                tensor[mol_idx[mol_id], col_idx] = True

    np.save(output_npy_path, tensor)
    with open(output_meta_path, "w") as f:
        json.dump({"mol_order": mol_order, "class_labels": ilp_class_ids}, f)
    print(f"Saved ILP predictions tensor: {output_npy_path} ({n_mols} × {n_classes})")

    return pd.DataFrame(tensor, index=mol_order, columns=ilp_class_ids)


def test_chebi_classes(run_to_evaluate, problem_dir: str, predicate_set: str, results_dir, selection_mode: Optional[str], selection_k: Optional[int], test_on: Literal["validation", "test"] = "test", verbose: bool = False, **kwargs):

    with open(os.path.join(results_dir, "config.yml"), "a+") as f:
        f.write(f"problem_dir: {problem_dir}\n")
        f.write(f"run_to_evaluate: {run_to_evaluate}\n")

    with open(os.path.join(run_to_evaluate, "results.json"), "r") as f:
        results = []
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line}")
    
    for row in results:
        chebi_id = row["chebi_id"]
        prog_str = row["program"]
        if prog_str is None:
            print(f"No program found for ChEBI:{chebi_id}, skipping...")
            continue
        conf_matrix = None
        start_time = time.perf_counter()
        print(f"Testing ChEBI:{chebi_id}...")
        from chebILP.clingo_eval import run_ilp_validation_clingo
        try:
            conf_matrix, details = run_ilp_validation_clingo(
                chebi_id, prog_str,
                exs_file=get_exs_path(chebi_id, split=test_on, base_dir=problem_dir),
                bk_file=get_bk_path(chebi_id, predicate_set=predicate_set, split=test_on, base_dir=problem_dir,
                                    selection_mode=selection_mode, selection_k=selection_k),
                return_details=True,
            )
            if verbose:
                balanced_acc = 0.5 * (conf_matrix["TP"] / (conf_matrix["TP"] + conf_matrix["FN"]) if (conf_matrix["TP"] + conf_matrix["FN"]) > 0 else 0) + 0.5 * (conf_matrix["TN"] / (conf_matrix["TN"] + conf_matrix["FP"]) if (conf_matrix["TN"] + conf_matrix["FP"]) > 0 else 0)
                f1_score = conf_matrix["TP"] / (conf_matrix["TP"] + 0.5 * (conf_matrix["FP"] + conf_matrix["FN"])) if (conf_matrix["TP"] + conf_matrix["FP"] + conf_matrix["FN"]) > 0 else 0.0
                print(f"  F1: {f1_score:.2f}, BA: {balanced_acc:.2f} TPs: {conf_matrix['TP']}, FPs: {conf_matrix['FP']}, TNs: {conf_matrix['TN']}, FNs: {conf_matrix['FN']}")
                for outcome in ("FP", "FN"):
                    subset = [d for d in details if d["outcome"] == outcome][:10]
                    print(f"  {'False Positive' if outcome == 'FP' else 'False Negative'} samples (showing up to 10):")
                    for d in subset:
                        print(f"    {d['id']} (true label: {d['true_label']})")
            
            with open(os.path.join(results_dir, "results.json"), "a+") as f:
                result_entry = {
                    "chebi_id": chebi_id,
                    "time_taken": time.perf_counter() - start_time,
                    "program": prog_str,
                    f"{test_on}_score": conf_matrix,
                    f"{test_on}_details": details
                }
                f.write(json.dumps(result_entry) + "\n")

        except Exception as e:
            print(f"Error testing ChEBI:{chebi_id}: {e}")
            with open(os.path.join(results_dir, "results.json"), "a+") as f:
                result_entry = {
                    "chebi_id": chebi_id,
                    "time_taken": time.perf_counter() - start_time,
                    "program": prog_str,
                    f"{test_on}_score": None,
                    f"{test_on}_details": None,
                    "error": str(e)
                }
                f.write(json.dumps(result_entry) + "\n")
