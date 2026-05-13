import os
import traceback
from typing import Literal
import typing
import json
import time
import argparse

from chebILP.mol2ilp import ILPProblemBuilder, tee_output, AVAILABLE_PREDICATE_SETS
from chebILP.learn_fgs import FGILPProblemBuilder
from chebILP.ilp_classifier import run_ilp_training_subprocess, run_ilp_validation_subprocess
from chebILP.ilp_path_manager import get_exs_path, get_bk_path, get_bias_path




def learn_chebi_classes(classes_list, ilp_builder: ILPProblemBuilder, results_dir, timeout=20, selection_mode:Literal["claude", "random", "top_k"]|None=None, selection_k:int|None=None, mdl_weight_fn=1, mdl_weight_fp=1, mdl_weight_size=1):

        # Build settings parameters for Popper
        settings_parameters = {
            "noisy": True,
            "anytime_solver": "nuwls",
            "timeout": timeout,
            "mdl_weight_fn": mdl_weight_fn,
            "mdl_weight_fp": mdl_weight_fp,
            "mdl_weight_size": mdl_weight_size,
        }

        with open(os.path.join(results_dir, "config.yml"), "a+") as f:
            f.write(f"problem_dir: {ilp_builder.problem_dir}\n")
            f.write("popper_settings:\n")
            for key, value in settings_parameters.items():
                f.write(f"\t{key}: {value}\n")

        ilp_builder.build_bias(classes_list, selection_mode=selection_mode, selection_k=selection_k)

        for chebi_id in classes_list:
            start_time = time.perf_counter()
            # Run training in subprocess (isolated Prolog session)
            print(f"Training ChEBI:{chebi_id}")
            exs_path = get_exs_path(chebi_id, split="train", base_dir=ilp_builder.problem_dir)
            bk_path = get_bk_path(chebi_id, split="train", base_dir=ilp_builder.problem_dir, predicate_set=ilp_builder.predicate_set, selection_mode=selection_mode, selection_k=selection_k)
            bias_path = get_bias_path(chebi_id, split="train", base_dir=ilp_builder.problem_dir, predicate_set=ilp_builder.predicate_set, selection_mode=selection_mode, selection_k=selection_k, max_vars=ilp_builder.max_vars, max_body=ilp_builder.max_body, max_clauses=ilp_builder.max_clauses)
            if not os.path.exists(exs_path) or not os.path.exists(bk_path) or not os.path.exists(bias_path):
                print(f"Missing files for ChEBI:{chebi_id} - skipping. exs_path: {exs_path}, bk_path: {bk_path}, bias_path: {bias_path}")
                continue
            train_result = run_ilp_training_subprocess(exs_path, bk_path, bias_path, settings_parameters, log_dir=results_dir)
            prog_str = train_result["prog_str"]  # string representation for display/storage
            score = train_result["score"]
            if score:
                tp, fn, tn, fp = score[0], score[1], score[2], score[3]
                f1 = (2*tp / (2*tp + fn + fp)) if (tp + fn + fp) > 0 else 0.0
                print(f"ChEBI:{chebi_id} - F1: {f1:.2f} (TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn})")
            if prog_str:
                print(f"    Learned program:\n    {prog_str}")
            else:
                print(f"ChEBI:{chebi_id} - No program learned.")

            conf_matrix = None
            if prog_str is not None:
                # Run validation in subprocess (isolated Prolog session)
                print(f"Validating ChEBI:{chebi_id}...")
                try:
                    from chebILP.clingo_eval import run_ilp_validation_clingo
                    conf_matrix = run_ilp_validation_clingo(
                        chebi_id, prog_str,
                        exs_file=get_exs_path(chebi_id, split="validation", base_dir=ilp_builder.problem_dir),
                        bk_file=get_bk_path(chebi_id, predicate_set=ilp_builder.predicate_set, split="validation", base_dir=ilp_builder.problem_dir),
                    )
                    f1 = (2*conf_matrix["TP"] / (2*conf_matrix["TP"] + conf_matrix["FP"] + conf_matrix["FN"])) if (conf_matrix["TP"] + conf_matrix["FP"] + conf_matrix["FN"]) > 0 else 0.0
                    print(f"    Validation F1: {f1:.2f} (TP: {conf_matrix['TP']}, FP: {conf_matrix['FP']}, TN: {conf_matrix['TN']}, FN: {conf_matrix['FN']})")
                except Exception as e:
                    print(f"Validation failed for ChEBI:{chebi_id} with error: {e}")
                    traceback.print_exc()
                    conf_matrix = None

            with open(os.path.join(results_dir, "results.json"), "a+") as f:
                result_entry = {
                    "chebi_id": chebi_id,
                    "train_score": {"TP": tp, "FP": fp, "TN": tn, "FN": fn} if score else None,
                    "time_taken": time.perf_counter() - start_time,
                    "program": prog_str,
                    "validation_score": conf_matrix,
                }
                f.write(json.dumps(result_entry) + "\n")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_classes(labels_file: str) -> list[str]:
    with open(labels_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _make_ilp_builder(args) -> ILPProblemBuilder:
    if isinstance(args, dict):
        fg_mode = args["fg_mode"]
        chebi_version = int(args["chebi_version"])
        chebi_split = args["chebi_split"]
        predicate_set = args["predicate_set"]
        max_vars = int(args.get("max_vars", 6))
        max_body = int(args.get("max_body", 8))
        max_clauses = int(args.get("max_clauses", 2))
    else:
        fg_mode = args.fg_mode
        chebi_version = args.chebi_version
        chebi_split = args.chebi_split
        predicate_set = args.predicate_set
        max_vars = args.max_vars
        max_body = args.max_body
        max_clauses = args.max_clauses

    if fg_mode:
        return FGILPProblemBuilder(
            chebi_version=chebi_version,
            chebi_split=chebi_split,
            dataset_path=os.path.join("data", "chebi_fgs_dataset.pkl"),
            predicate_set=predicate_set,
            max_vars=max_vars,
            max_body=max_body,
            max_clauses=max_clauses,
        )
    return ILPProblemBuilder(
        chebi_version=chebi_version,
        chebi_split=chebi_split,
        muggleton=False,
        predicate_set=predicate_set,
        max_vars=max_vars,
        max_body=max_body,
        max_clauses=max_clauses,
    )


def _make_results_dir(fg_mode: bool) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("data", "results", f"run_fgs_{timestamp}" if fg_mode else f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w+") as f:
        f.write("")
    return results_dir


# ── Subcommand handlers ─────────────────────────────────────────────────────


def _handle_build_samples(args):
    classes = _load_classes(args.labels_file)
    ilp_builder = _make_ilp_builder(args)
    ilp_builder.build_examples(classes, min_pos_samples=args.min_pos_samples, max_pos_samples=args.max_pos_samples, min_neg_samples=args.min_neg_samples, max_neg_samples=args.max_neg_samples)


def _handle_build_bk(args):
    classes = _load_classes(args.labels_file)
    ilp_builder = _make_ilp_builder(args)
    ilp_builder.build_bk(classes)


def _handle_learn(args):
    classes = _load_classes(args.labels_file)
    results_dir = _make_results_dir(args.fg_mode)
    log_path = os.path.join(results_dir, "run.log")

    # write config file
    with open(os.path.join(results_dir, "config.yml"), "w+") as f:
        f.write("args:\n")
        for arg in vars(args):
            f.write(f"  {arg}: {getattr(args, arg)}\n")

    with tee_output(log_path):
        ilp_builder = _make_ilp_builder(args)
        learn_chebi_classes(
            classes, ilp_builder, results_dir,
            timeout=args.timeout,
            selection_mode=args.selection_mode,
            selection_k=args.selection_k,
            mdl_weight_fn=args.mdl_weight_fn,
            mdl_weight_fp=args.mdl_weight_fp,
            mdl_weight_size=args.mdl_weight_size,
        )


def _handle_select_predicates(args):
    from chebILP.select_predicates import select_predicates_for_classes

    with open(args.labels_file, "r") as f:
        chebi_ids = [int(line.strip()) for line in f if line.strip()]

    print(f"Processing {len(chebi_ids)} ChEBI classes...")
    results = select_predicates_for_classes(
        chebi_ids=chebi_ids,
        chebi_version=args.chebi_version,
        problem_dir=args.problem_dir,
        predicate_set=args.predicate_set,
        selection_mode=args.selection_mode,
        selection_k=args.selection_k,
    )
    successful = sum(1 for v in results.values() if v is not None)
    print(f"\nCompleted: {successful}/{len(chebi_ids)} classes processed successfully")


def _handle_ensemble_predict(args):
    from chebILP.ensemble_eval import EnsemblePredictor, load_dl_preds
    from chebi_utils import build_chebi_graph, get_hierarchy_subgraph, extract_molecules
    import networkx as nx

    with open(args.label_stats) as f:
        # chebi_id,num_pos,num_neg,has_negatives
        label_stats_lines = [line.strip().split(",") for line in f if line.strip()][1:]
        label_stats = {
            line[0]: {"num_pos": int(line[1]), "num_neg": int(line[2]), "has_negatives": line[3].lower() == "true"}
            for line in label_stats_lines
        }
    print(f"Label stats: {len(label_stats)} labels, {sum(1 for v in label_stats.values() if not v['has_negatives'])} without negatives")

    dl_preds = load_dl_preds(args.dl_preds_npy, args.dl_preds_meta)
    print(f"DL predictions: {dl_preds.shape[0]} molecules x {dl_preds.shape[1]} labels")

    ilp_val_run_dirs = {}
    for run_path in args.ilp_val_runs:
        ilp_val_run_dirs[os.path.basename(run_path.rstrip("/\\"))] = run_path

    ilp_run_dirs = {}
    for run_path in args.ilp_runs:
        ilp_run_dirs[os.path.basename(run_path.rstrip("/\\"))] = run_path

    data_dir = os.path.join("data", f"chebi_v{args.chebi_version}")
    obo_path = os.path.join(data_dir, "raw", "chebi.obo")
    sdf_path = os.path.join(data_dir, "raw", "chebi.sdf.gz")
    chebi_graph = get_hierarchy_subgraph(build_chebi_graph(obo_path))

    chebi_graph = get_hierarchy_subgraph(build_chebi_graph(obo_path))
    
    with open(args.dl_val_scores) as f:
        # class_id,TP,FP,TN,FN
        dl_val_scores_lines = [line.strip().split(",") for line in f if line.strip()][1:]
        dl_val_scores = {
            line[0]: {"TP": int(line[1]) if line[1] != "" else 0, "FP": int(line[2]) if line[2] != "" else 0, "TN": int(line[3]) if line[3] != "" else 0, "FN": int(line[4]) if line[4] != "" else 0}
            for line in dl_val_scores_lines
        }


    predictor = EnsemblePredictor(
        ilp_val_runs=ilp_val_run_dirs,
        ilp_runs=ilp_run_dirs,
        dl_preds=dl_preds,
        chebi_graph=chebi_graph,
        label_stats=label_stats,
        dl_val_scores=dl_val_scores,
        classifier_chain_mode=not args.native_mode,
        model_selection_metric=args.model_selection_metric,
    )

    mol_ids = []
    with open(args.chebi_split) as f:
        for line in f.readlines()[1:]:
            mol_id, split = line.strip().split(",")
            if split == args.predict_on:
                mol_ids.append(mol_id)
    print(f"Predicting on {len(mol_ids)} '{args.predict_on}' molecules...")

    molecules_df = None
    if args.load_molecules:
        from chebi_utils import extract_molecules
        sdf_path = os.path.join(data_dir, "raw", "chebi.sdf.gz")
        molecules_df = extract_molecules(sdf_path)
        molecules_df.index = molecules_df["chebi_id"].astype(str)
        molecules_df.index.name = None
        print(f"Loaded {len(molecules_df)} molecules for Clingo fallback")

    predictions_df = predictor.predict_set(mol_ids, molecules_df)

    import numpy as np, json as _json
    arr = predictions_df.to_numpy().astype("float32")
    meta = {"mol_order": list(predictions_df.index), "class_labels": list(predictions_df.columns)}

    npy_path = args.output if args.output.endswith(".npy") else args.output + ".npy"
    meta_path = npy_path.replace(".npy", "_metadata.json")

    os.makedirs(os.path.dirname(npy_path) or ".", exist_ok=True)
    np.save(npy_path, arr)
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)

    n_pos = int(arr.sum())
    print(f"Saved predictions: {npy_path}  (shape {arr.shape}, {n_pos} positive assignments)")
    print(f"Saved metadata:    {meta_path}")


def _handle_test(args):
    from chebILP.test import test_chebi_classes

    # load config from the run to evaluate
    with open(os.path.join(args.run_to_evaluate, "config.yml"), "r") as f:
        config = {}
        for line in f:
            if ": " in line:
                key, value = line.strip().split(": ", 1)
                config[key] = value
    assert "predicate_set" in config and "selection_mode" in config and "selection_k" in config and "problem_dir" in config and "fg_mode" in config, \
        "Config file must contain predicate_set, selection_mode, selection_k, problem_dir, and fg_mode"

    config["fg_mode"] = config["fg_mode"] == "True"
    config["selection_mode"] = config["selection_mode"] if config["selection_mode"] != "None" else None
    config["selection_k"] = int(config["selection_k"]) if config["selection_k"] != "None" else None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("data", f"results_{args.test_on}", f"run_fgs_{timestamp}" if config["fg_mode"] else f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w+") as f:
        f.write("")

    log_path = os.path.join(results_dir, "run.log")

    with open(os.path.join(results_dir, "config.yml"), "w+") as f:
        f.write("args:\n")
        for arg in vars(args):
            f.write(f"  {arg}: {getattr(args, arg)}\n")

    with tee_output(log_path):
        test_chebi_classes(args.run_to_evaluate, config["problem_dir"], config["predicate_set"], results_dir, selection_mode=config["selection_mode"], selection_k=config["selection_k"], test_on=args.test_on, verbose=args.verbose)


# ── Argument parsing ─────────────────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by all subcommands that build an ILPProblemBuilder."""
    parser.add_argument("--labels_file", type=str, required=True, help="Path to the labels file (one ChEBI ID per line).")
    parser.add_argument("--chebi_split", type=str, required=True, help="Path to the ChEBI split file.")
    parser.add_argument("--fg_mode", action="store_true", help="Learn functional groups instead of ChEBI classes.")
    parser.add_argument("--chebi_version", type=int, default=248, help="ChEBI version to use.")
    parser.add_argument("--predicate_set", type=str, default="atoms", choices=typing.get_args(AVAILABLE_PREDICATE_SETS), help="Which predicate set to use for background knowledge.")
    parser.add_argument("--max_vars", type=int, default=6, help="Maximum number of variables in learned rules.")
    parser.add_argument("--max_body", type=int, default=8, help="Maximum number of body literals in learned rules.")
    parser.add_argument("--max_clauses", type=int, default=2, help="Maximum number of clauses in the learned program.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ILP classification CLI for ChEBI classes using Popper.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── build_samples ────────────────────────────────────────────────────
    sp_samples = subparsers.add_parser(
        "build_samples",
        help="Build positive/negative example files (exs.pl) for the given ChEBI classes.",
    )
    _add_common_args(sp_samples)
    sp_samples.add_argument("--min_pos_samples", type=int, default=25, help="Minimum positive samples per class.")
    sp_samples.add_argument("--max_pos_samples", type=int, default=200, help="Maximum positive samples per class.")
    sp_samples.add_argument("--min_neg_samples", type=int, default=25, help="Minimum negative samples per class.")
    sp_samples.add_argument("--max_neg_samples", type=int, default=200, help="Maximum negative samples per class.")

    sp_samples.set_defaults(func=_handle_build_samples)

    # ── build_bk ─────────────────────────────────────────────────────────
    sp_bk = subparsers.add_parser(
        "build_bk",
        help="Build background knowledge (bk.pl) and bias template (bias.pl) for the given ChEBI classes.",
    )
    _add_common_args(sp_bk)
    sp_bk.set_defaults(func=_handle_build_bk)

    # ── learn ────────────────────────────────────────────────────────────
    sp_learn = subparsers.add_parser(
        "learn",
        help="Run ILP learning (training + validation) for the given ChEBI classes.",
    )
    _add_common_args(sp_learn)
    sp_learn.add_argument("--timeout", type=int, default=20, help="Timeout for ILP solver in seconds.")
    sp_learn.add_argument("--max_pos_samples", type=int, default=200, help="Maximum positive samples per class.")
    sp_learn.add_argument("--max_neg_samples", type=int, default=200, help="Maximum negative samples per class.")
    sp_learn.add_argument("--selection_mode", type=str, default=None, choices=["claude", "random", "top_k"], help="Mode for selecting body predicates in bias file.")
    sp_learn.add_argument("--selection_k", type=int, default=10, help="Number of predicates selection with selection_mode (required if selection_mode is set).")
    sp_learn.add_argument("--mdl_weight_fn", type=int, default=1, help="Weight β for false negatives in MDL cost (default: 1).")
    sp_learn.add_argument("--mdl_weight_fp", type=int, default=1, help="Weight γ for false positives in MDL cost (default: 1).")
    sp_learn.add_argument("--mdl_weight_size", type=int, default=1, help="Weight α for program size in MDL cost (default: 1).")
    sp_learn.set_defaults(func=_handle_learn)

    # ── select_predicates ────────────────────────────────────────────────
    sp_select = subparsers.add_parser(
        "select_predicates",
        help="Select predicates for ChEBI classes (via Claude, random, or top-k frequency).",
    )
    sp_select.add_argument("--labels_file", type=str, required=True, help="Path to file with ChEBI IDs (one per line).")
    sp_select.add_argument("--chebi_version", type=int, default=248, help="ChEBI version to use.")
    sp_select.add_argument("--problem_dir", type=str, default=None, help="Base directory for ILP problems.")
    sp_select.add_argument("--predicate_set", type=str, default="atoms", choices=typing.get_args(AVAILABLE_PREDICATE_SETS), help="Which predicate set to use.")
    sp_select.add_argument("--selection_mode", type=str, default="claude", choices=["claude", "random", "top_k"], help="How to select predicates.")
    sp_select.add_argument("--selection_k", type=int, default=10, help="Number of predicates to select.")
    sp_select.set_defaults(func=_handle_select_predicates)

    # ── test ─────────────────────────────────────────────────────────────
    sp_test = subparsers.add_parser(
        "test",
        help="Evaluate learned programs on the test set using results from a previous run.",
    )
    sp_test.add_argument("--run_to_evaluate", type=str, required=True, help="Path to a previous run directory (must contain results.json and config.yml).")
    sp_test.add_argument("--test_on", type=str, default="test", choices=["val", "test"], help="Split to evaluate on: 'test' (default) or 'val' (validation).")
    sp_test.add_argument("--verbose", action="store_true", help="Log classification result for up to 10 positive and negative samples per class.")
    sp_test.set_defaults(func=_handle_test)

    # ── ensemble_predict ─────────────────────────────────────────────────
    sp_ep = subparsers.add_parser(
        "ensemble_predict",
        help="(EXP-006) Run the hierarchical ensemble predictor on a data split.",
    )
    sp_ep.add_argument("--chebi_version", type=int, default=248)
    sp_ep.add_argument("--chebi_split", type=str, required=True,
                       help="Path to the splits CSV file.")
    sp_ep.add_argument("--dl_val_scores", type=str, default=os.path.join("data", "preds", "dl_val_direct_neighbor_scores.csv"),
                       help="DL validation scores on direct neighbors (generated with get_dl_direct_neighbor_scores).")
    sp_ep.add_argument("--ilp_val_runs", type=str, nargs="+", default=[],
                       help="Validation-run directories (output of 'test' with test_on=validation); used for model selection and cache.")
    sp_ep.add_argument("--ilp_runs", type=str, nargs="+", default=[],
                       help="ILP run directories providing the programs used for prediction; their validation_details also populate the cache.")
    sp_ep.add_argument("--dl_preds_npy", type=str, default=os.path.join("data", "preds", "test_preds_chebi25.npy"),
                       help="DL predictions .npy for the target split (validation or test).")
    sp_ep.add_argument("--dl_preds_meta", type=str, default=os.path.join("data", "preds", "test_preds_chebi25_metadata.json"),
                       help="Metadata JSON for --dl_preds_npy.")
    sp_ep.add_argument("--label_stats", type=str, default=os.path.join("data", "chebi_v248", "ChEBI25_3_STAR", "processed", "class_stats.csv"),
                       help="Path to the class statistics CSV file (contains list of label classes + info about which classes have negative samples and which do not).")
    sp_ep.add_argument("--predict_on", type=str, default="validation",
                       choices=["train", "validation", "test"],
                       help="Which split to predict on (default: validation).")
    sp_ep.add_argument("--load_molecules", action="store_true",
                       help="Load molecule structures for Clingo fallback on uncached molecules (required for test-set ILP predictions).")
    sp_ep.add_argument("--output", type=str,
                       default=os.path.join("data", "results_correct_val", "ensemble_predictions.npy"),
                       help="Output .npy path; a matching _metadata.json is written alongside.")
    sp_ep.add_argument("--model_selection_metric", type=str, default="f1",
                       choices=["balanced_acc", "f1", "weighted_f1"],
                       help="Metric for model selection: 'f1'/'balanced_acc' pick the single best model; 'weighted_f1' blends all models weighted by F1.")
    sp_ep.add_argument("--native_mode", "-n", action="store_true",
                       help="Skips classifier chain mode (only predict a class if all parents are predicted positive, the default).")
    sp_ep.set_defaults(func=_handle_ensemble_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
