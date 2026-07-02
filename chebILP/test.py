

import json
import os
import time
from chebILP.ilp_path_manager import get_bk_path, get_exs_path
from typing import Literal, Optional

import numpy as np
import pandas as pd


def _silent_clingo_logger(code, message):
    """Swallow Clingo's grounding messages.
    """
    pass


def build_ilp_preds_tensor(
    programs: dict[str, str],
    molecules_df: pd.DataFrame,
    mol_order: list[str],
    output_npy_path: str,
    output_meta_path: str,
    predicate_set: str = "atoms",
    aux_timeout: float | None = None,
    problem_dir: str | None = None,
    label_timeout: float = 60.0,
) -> pd.DataFrame:
    """
    Evaluate ILP programs against every molecule in mol_order and save the
    result as a boolean numpy tensor.

    Builds background knowledge from scratch (not from existing bk.pl problem
    files), using the same predicate-set logic as ILPProblemBuilder.build_bk so
    the programs are evaluated against the BK they were learned on. Intended for
    generating full split prediction tensors used by ensemble_construct.

    Args:
        programs:         class_id → Prolog program string (only classes with
                          a learned program; null programs must be excluded).
        molecules_df:     DataFrame with 'mol' column; index = molecule IDs
                          (string ChEBI IDs). Built from the raw SDF via
                          extract_molecules().
        mol_order:        ordered list of molecule IDs to evaluate.
        output_npy_path:  destination .npy file for the boolean tensor.
        output_meta_path: destination JSON file for mol_order / class_labels.
        predicate_set:    which background-knowledge predicate set to build
                          (must match the run being predicted for).
        aux_timeout:      per-predicate timeout for LLM auxiliary predicates
                          (llm_generated_fgs only); None → library default.
        problem_dir:      ILP problem directory (needed to load auxiliary
                          predicates for llm_generated_fgs).
        label_timeout:    Clingo solving timeout in seconds (default 60).

    Returns a boolean DataFrame (index=mol_order, columns=sorted class IDs).
    """
    import tqdm
    import clingo
    from chebILP.auxiliary_predicates import DEFAULT_AUX_TIMEOUT
    from chebILP.ilp_problem_builder import build_full_background

    if aux_timeout is None:
        aux_timeout = DEFAULT_AUX_TIMEOUT

    ilp_class_ids = sorted(programs.keys())
    n_mols = len(mol_order)
    n_classes = len(ilp_class_ids)
    print(f"Building ILP predictions tensor: {n_mols} molecules × {n_classes} classes "
          f"(predicate_set={predicate_set})")

    # Molecules we actually have; anything missing stays False in the tensor.
    present_rows = molecules_df[molecules_df.index.isin(set(mol_order))].dropna(subset=["mol"])
    present_mol_ids = [str(i) for i in present_rows.index]
    missing = len(set(mol_order)) - len(set(present_mol_ids))
    if missing:
        print(f"  {missing} molecules missing from molecules_df — predicted False")

    # Column index per target label, and row index per molecule id.
    col_of = {f"chebi_{cls_id}": j for j, cls_id in enumerate(ilp_class_ids)}
    mol_idx = {mol_id: i for i, mol_id in enumerate(mol_order)}
    tensor = np.zeros((n_mols, n_classes), dtype=bool)

    # Parse-check every program once and join the valid ones into a single block; each
    # program's only derived head is its unique chebi_{id} target, so they never interfere.
    valid_programs = []
    for cls_id in ilp_class_ids:
        prog = "\n".join(line for line in programs[cls_id].split("\n") if line.strip())
        try:
            clingo.Control(logger=_silent_clingo_logger).add("t", [], prog)  # syntax check only
            valid_programs.append(prog)
        except RuntimeError as e:
            print(f"  Skipping ChEBI:{cls_id} — failed to parse program: {e}")
    programs_str = "\n".join(valid_programs)

    # Auxiliary predicates (llm_generated_fgs) are molecule-independent definitions, so
    # gather them once (first definition of each aux_ name wins, matching
    # load_auxiliary_predicates' within-class de-duplication); their extensions are then
    # computed per molecule inside the loop. Only keep predicates actually referenced by
    # some program — computing the rest would just waste time per molecule.
    aux_predicates = None
    if predicate_set == "llm_generated_fgs":
        import re
        from chebILP.auxiliary_predicates import load_auxiliary_predicates
        used_aux_names = set(re.findall(r"\baux_\w+", programs_str))
        seen: dict[str, object] = {}
        for cls_id in tqdm.tqdm(ilp_class_ids, desc="Loading auxiliary predicates"):
            for pred in load_auxiliary_predicates(cls_id, problem_dir=problem_dir):
                if pred.name in used_aux_names:
                    seen.setdefault(pred.name, pred)
        aux_predicates = list(seen.values())
        print(f"Loaded {len(aux_predicates)} auxiliary predicate(s) used across "
              f"{len(used_aux_names)} referenced name(s)")

    # Molecules are independent, so evaluate them one at a time: this keeps each Clingo
    # grounding tiny (one molecule's background knowledge + all programs) instead of
    # grounding the whole split at once.
    n_timeout = 0
    n_error = 0
    for row in tqdm.tqdm(present_rows.itertuples(), total=len(present_rows), desc="Molecules"):
        row_df = present_rows.loc[[row.Index]]
        bg_facts = build_full_background(
            row_df, predicate_set=predicate_set,
            aux_predicates=aux_predicates, aux_timeout=aux_timeout,
        )
        fired: set[str] = set()

        def _on_model(model):
            for atom in model.symbols(atoms=True):
                if atom.name in col_of:
                    fired.add(atom.name)

        try:
            ctl = clingo.Control(logger=_silent_clingo_logger)
            ctl.add("base", [], "\n".join(bg_facts) + "\n" + programs_str)
            ctl.ground([("base", [])])
            if label_timeout and label_timeout > 0:
                with ctl.solve(on_model=_on_model, async_=True) as handle:
                    if not handle.wait(label_timeout):
                        handle.cancel()
                        n_timeout += 1
            else:
                ctl.solve(on_model=_on_model)
        except Exception as e:
            n_error += 1
            print(f"  Clingo evaluation failed for molecule {row.Index}: {e}")
            continue

        i = mol_idx.get(str(row.Index))
        if i is not None:
            for name in fired:
                tensor[i, col_of[name]] = True

    if n_timeout:
        print(f"  {n_timeout} molecule(s) hit the {label_timeout}s solving timeout — those rows may be incomplete")
    if n_error:
        print(f"  {n_error} molecule(s) failed to evaluate — predicted False")

    np.save(output_npy_path, tensor)
    with open(output_meta_path, "w") as f:
        json.dump({"mol_order": mol_order, "class_labels": ilp_class_ids}, f)
    print(f"Saved ILP predictions tensor: {output_npy_path} ({n_mols} × {n_classes})")

    return pd.DataFrame(tensor, index=mol_order, columns=ilp_class_ids)


def predict_smiles(
    smiles_list: list[str],
    rules_file: str,
    target_predicates: list[str],
    verbose: bool = False,
) -> list[dict]:
    """
    Predict which target predicates each molecule (given as a SMILES string) satisfies.

    Runs the same evaluation as ``test`` (build background knowledge, run the rule set
    through Clingo, collect the target predicates entailed for the molecule), but it does
    not depend on a previous run directory or pre-built problem files: background knowledge
    is built from scratch for the supplied SMILES instead of running a validation/test split.

    Each molecule is evaluated in its own Clingo run so that the time it takes can be
    recorded individually.

    Args:
        smiles_list:       SMILES strings of the molecules to classify.
        rules_file:        Path to a file containing one or more logic programs
                           (Prolog/ASP rule clauses). Comment (``%``) and blank lines
                           are ignored.
        target_predicates: Head predicate names to check, e.g. ``"chebi_35341"``.
        verbose:           If True, print the satisfied predicates for each molecule.

    Returns:
        list of dicts, one per input SMILES, each with keys:
          - ``"SMILES"``: the input SMILES string.
          - ``"labels"``: list of target predicates the molecule satisfies (empty if none
            or if the SMILES is invalid).
          - ``"time"``: time (seconds) the Clingo evaluation took, or None for invalid SMILES.
    """
    from rdkit import Chem
    from chebILP.ilp_problem_builder import build_background_chemlog
    from chebILP.clingo_eval import evaluate_with_clingo

    with open(rules_file, "r") as f:
        rules = [line.strip() for line in f if line.strip() and not line.strip().startswith("%")]

    results: list[dict] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES (skipped): {smiles!r}")
            results.append({"SMILES": smiles, "labels": [], "time": None})
            continue

        mol_id = "m0"
        mol_df = pd.DataFrame([{"mol": mol}], index=[mol_id])
        background_facts, _ = build_background_chemlog(mol_df)

        start_time = time.perf_counter()
        try:
            positives = evaluate_with_clingo(rules, background_facts, target_predicates, [mol_id])
        except Exception as e:
            print(f"Clingo evaluation failed for {smiles!r}: {e}")
            positives = {}
        elapsed = time.perf_counter() - start_time

        labels = [target for target in target_predicates if mol_id in positives.get(target, [])]
        results.append({"SMILES": smiles, "labels": labels, "time": elapsed})

    if verbose:
        for entry in results:
            labels = entry["labels"]
            time_str = f"{entry['time']:.3f}s" if entry["time"] is not None else "n/a"
            print(f"{entry['SMILES']} [{time_str}]: {', '.join(labels) if labels else '(none)'}")

    return results


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
