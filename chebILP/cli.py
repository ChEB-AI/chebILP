import os
import traceback
from typing import Literal
import json
import time
import argparse

from chebILP.mol2ilp import ILPProblemBuilder, tee_output
from chebILP.learn_fgs import FGILPProblemBuilder
from chebILP.ilp_classifier import run_ilp_training_subprocess, run_ilp_validation_subprocess
from chebILP.ilp_path_manager import get_exs_path, get_bk_path, get_bias_path




def learn_chebi_classes(classes_list, ilp_builder: ILPProblemBuilder, results_dir, timeout=20, selection_mode:Literal["claude", "random", "top_k"]|None=None, selection_k:int|None=None):

        # Build settings parameters for Popper
        settings_parameters = {
            "noisy": True,
            "anytime_solver": "nuwls",
            "timeout": timeout,
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
                f1 = (2*score[0] / (2*score[0] + score[1] + score[3])) if (score[0] + score[1] + score[3]) > 0 else 0.0
                print(f"ChEBI:{chebi_id} - F1: {f1:.2f} (TP: {score[0]}, FP: {score[1]}, TN: {score[2]}, FN: {score[3]})")
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
                        bk_file=get_bk_path(chebi_id, predicate_set=ilp_builder.predicate_set, split="validation", base_dir=ilp_builder.problem_dir, selection_mode=selection_mode, selection_k=selection_k),
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
                    "train_score": {"TP": score[0], "FP": score[1], "TN": score[2], "FN": score[3]} if score else None,
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
    if args.fg_mode:
        return FGILPProblemBuilder(
            chebi_version=args.chebi_version,
            chebi_split=args.chebi_split,
            dataset_path=os.path.join("data", "chebi_fgs_dataset.pkl"),
            predicate_set=args.predicate_set,
            max_vars=args.max_vars,
            max_body=args.max_body,
            max_clauses=args.max_clauses,
        )
    return ILPProblemBuilder(
        chebi_version=args.chebi_version,
        chebi_split=args.chebi_split,
        muggleton=False,
        predicate_set=args.predicate_set,
        max_vars=args.max_vars,
        max_body=args.max_body,
        max_clauses=args.max_clauses,
    )


def _make_results_dir(fg_mode: bool) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("ilp", "results", f"run_fgs_{timestamp}" if fg_mode else f"run_{timestamp}")
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
            selection_k=args.top_k,
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
        top_k=args.top_k,
    )
    successful = sum(1 for v in results.values() if v is not None)
    print(f"\nCompleted: {successful}/{len(chebi_ids)} classes processed successfully")


def _handle_test(args):
    from chebILP.test import test_chebi_classes

    # load config from the run to evaluate
    with open(os.path.join(args.run_to_evaluate, "config.yml"), "r") as f:
        config = {}
        for line in f:
            if ": " in line:
                key, value = line.strip().split(": ", 1)
                config[key] = value
    assert "chebi_version" in config and "fg_mode" in config and "predicate_set" in config and "selection_mode" in config and "selection_k" in config and "chebi_split" in config, \
        "Config file must contain chebi_version, fg_mode, predicate_set, selection_mode, selection_k, and chebi_split"

    fg_mode = config["fg_mode"] == "True"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("ilp", "results_test", f"run_fgs_{timestamp}" if fg_mode else f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w+") as f:
        f.write("")

    log_path = os.path.join(results_dir, "run.log")

    # write config file
    with open(os.path.join(results_dir, "config.yml"), "w+") as f:
        f.write("args:\n")
        for arg in vars(args):
            f.write(f"  {arg}: {getattr(args, arg)}\n")

    with tee_output(log_path):
        if fg_mode:
            ilp_builder = FGILPProblemBuilder(
                chebi_version=config["chebi_version"],
                dataset_path=os.path.join("data", "chebi_fgs_dataset.pkl"),
                predicate_set=config["predicate_set"],
                chebi_split=config["chebi_split"]
            )
        else:
            ilp_builder = ILPProblemBuilder(
                chebi_version=config["chebi_version"],
                muggleton=False,
                predicate_set=config["predicate_set"],
                selection_mode=config["selection_mode"],
                selection_k=int(config["selection_k"]) if config["selection_k"] else None,
                chebi_split=config["chebi_split"],
            )
        test_chebi_classes(args.run_to_evaluate, ilp_builder, results_dir, predicate_set=config["predicate_set"])


# ── Argument parsing ─────────────────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by all subcommands that build an ILPProblemBuilder."""
    parser.add_argument("--labels_file", type=str, required=True, help="Path to the labels file (one ChEBI ID per line).")
    parser.add_argument("--chebi_split", type=str, required=True, help="Path to the ChEBI split file.")
    parser.add_argument("--fg_mode", action="store_true", help="Learn functional groups instead of ChEBI classes.")
    parser.add_argument("--chebi_version", type=int, default=248, help="ChEBI version to use.")
    parser.add_argument("--predicate_set", type=str, default="atoms", choices=["atoms", "chembl_fgs", "chebi_fgs", "chebi_fg_rules", "chebi_fg_learned_rules"], help="Which predicate set to use for background knowledge.")
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
    sp_learn.add_argument("--top_k", type=int, default=10, help="Number of predicates selection with selection_mode (required if selection_mode is set).")
    sp_learn.set_defaults(func=_handle_learn)

    # ── select_predicates ────────────────────────────────────────────────
    sp_select = subparsers.add_parser(
        "select_predicates",
        help="Select predicates for ChEBI classes (via Claude, random, or top-k frequency).",
    )
    sp_select.add_argument("--labels_file", type=str, required=True, help="Path to file with ChEBI IDs (one per line).")
    sp_select.add_argument("--chebi_version", type=int, default=248, help="ChEBI version to use.")
    sp_select.add_argument("--problem_dir", type=str, default=None, help="Base directory for ILP problems.")
    sp_select.add_argument("--predicate_set", type=str, default="atoms", choices=["atoms", "chembl_fgs"], help="Which predicate set to use.")
    sp_select.add_argument("--selection_mode", type=str, default="claude", choices=["claude", "random", "top_k"], help="How to select predicates.")
    sp_select.add_argument("--top_k", type=int, default=10, help="Number of predicates to select.")
    sp_select.set_defaults(func=_handle_select_predicates)

    # ── test ─────────────────────────────────────────────────────────────
    sp_test = subparsers.add_parser(
        "test",
        help="Evaluate learned programs on the test set using results from a previous run.",
    )
    sp_test.add_argument("--run_to_evaluate", type=str, required=True, help="Path to a previous run directory (must contain results.json and config.yml).")
    sp_test.set_defaults(func=_handle_test)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
