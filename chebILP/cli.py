import os
import typing
import time
import argparse

from chebILP.utils import tee_output, AVAILABLE_PREDICATE_SETS

from chebILP.ilp_classifier import learn_chebi_classes



# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_classes(labels_file: str) -> list[str]:
    with open(labels_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _make_ilp_builder(args):
    from chebILP.learn_fgs import FGILPProblemBuilder
    from chebILP.ilp_problem_builder import ILPProblemBuilder
    if isinstance(args, dict):
        fg_mode = args["fg_mode"]
        chebi_split = args.get("chebi_split")
        predicate_set = args["predicate_set"]
        max_vars = int(args.get("max_vars", 6))
        max_body = int(args.get("max_body", 8))
        max_clauses = int(args.get("max_clauses", 2))
        chebi_graph_path = args.get("chebi_graph_path")
        molecules_path = args.get("molecules_path")
    else:
        fg_mode = args.fg_mode
        chebi_split = getattr(args, "chebi_split", None)
        predicate_set = args.predicate_set
        max_vars = args.max_vars
        max_body = args.max_body
        max_clauses = args.max_clauses
        chebi_graph_path = getattr(args, "chebi_graph_path", None)
        molecules_path = getattr(args, "molecules_path", None)

    if fg_mode:
        return FGILPProblemBuilder(
            chebi_split=chebi_split,
            chebi_graph_path=chebi_graph_path,
            molecules_path=molecules_path,
            dataset_path=os.path.join("data", "chebi_fgs_dataset.pkl"),
            predicate_set=predicate_set,
            max_vars=max_vars,
            max_body=max_body,
            max_clauses=max_clauses,
        )
    return ILPProblemBuilder(
        chebi_split=chebi_split,
        chebi_graph_path=chebi_graph_path,
        molecules_path=molecules_path,
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
        learn_chebi_classes(
            classes, 
            getattr(args, "problem_dir", None),
            args.predicate_set,
            results_dir,
            timeout=args.timeout,
            selection_mode=args.selection_mode,
            selection_k=args.selection_k,
            max_vars=args.max_vars,
            max_body=args.max_body,
            max_clauses=args.max_clauses,
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


def _load_label_stats(path: str) -> list[str]:
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if path.endswith(".txt"):
        return lines
    # CSV: first column is class ID, skip header
    return [line.split(",")[0] for line in lines[1:]]



def _handle_build_ilp_preds_for_ensemble(args):
    """Build a full ILP predictions tensor for a given split from a run's results.json."""
    import pandas as pd
    from chebILP.test import build_ilp_preds_tensor
    from chebILP.ensemble_eval import load_ilp_results

    results = load_ilp_results(args.run_dir)
    programs = {cid: entry["program"] for cid, entry in results.items() if entry.get("program")}
    print(f"Loaded {len(programs)} ILP programs from {args.run_dir}")

    mol_ids = []
    with open(args.chebi_split) as f:
        for line in f.readlines()[1:]:
            mol_id, split = line.strip().split(",")
            if split == args.predict_on:
                mol_ids.append(mol_id)
    print(f"Building predictions for {len(mol_ids)} '{args.predict_on}' molecules")

    molecules_pkl = getattr(args, "molecules_path", None) or os.path.join(
        "data", f"chebi_v{args.chebi_version}", "molecules.pkl"
    )
    print(f"Loading molecules from {molecules_pkl}...")
    molecules_df = pd.read_pickle(molecules_pkl)

    prefix = "val" if args.predict_on == "validation" else args.predict_on
    output_npy = os.path.join(args.run_dir, f"full_{prefix}_preds.npy")
    output_meta = os.path.join(args.run_dir, f"full_{prefix}_preds_metadata.json")

    build_ilp_preds_tensor(programs, molecules_df, mol_ids, output_npy, output_meta)


def _handle_ensemble_construct(args):
    """Perform model selection and generate the ILP predictions tensor."""
    from chebILP.ensemble_eval import EnsembleConstructor, load_dl_preds

    label_stats = _load_label_stats(args.label_stats)
    print(f"Label stats: {len(label_stats)} labels")

    dl_val_preds = load_dl_preds(args.dl_val_preds_npy, args.dl_val_preds_meta)
    print(f"DL val predictions: {dl_val_preds.shape[0]} molecules x {dl_val_preds.shape[1]} classes")

    ilp_val_run_dirs = {os.path.basename(p.rstrip("/\\")): p for p in args.ilp_val_runs}
    data_dir = os.path.join("data", f"chebi_v{args.chebi_version}")
    import pickle as _pickle
    with open(os.path.join(data_dir, "chebi_graph.pkl"), "rb") as _f:
        chebi_graph = _pickle.load(_f)

    constructor = EnsembleConstructor(
        ilp_val_runs=ilp_val_run_dirs,
        dl_val_preds=dl_val_preds,
        label_stats=label_stats,
        chebi_graph=chebi_graph,
        predict_on=args.predict_on,
    )

    output_base = args.output
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)

    trusted_path = output_base + "_trusted_models.csv"
    with open(trusted_path, "w") as f:
        f.write("chebi_id,model\n")
        for chebi_id, model in constructor.trusted_model.items():
            f.write(f"{chebi_id},{model}\n")
    print(f"Saved trusted models: {trusted_path}")

    mol_ids = []
    with open(args.chebi_split) as f:
        for line in f.readlines()[1:]:
            mol_id, split = line.strip().split(",")
            if split == args.predict_on:
                mol_ids.append(mol_id)
    print(f"Slicing ILP tensor for {len(mol_ids)} '{args.predict_on}' molecules...")

    constructor.slice_ilp_preds(
        mol_order=mol_ids,
        output_npy_path=output_base + "_ilp_preds.npy",
        output_meta_path=output_base + "_ilp_preds_metadata.json",
    )


def _handle_ensemble_aggregate(args):
    """Aggregate pre-computed DL and ILP prediction tensors into ensemble predictions."""
    from chebILP.ensemble_eval import EnsembleAggregator, load_dl_preds, load_ilp_preds
    import numpy as np, json as _json, pandas as pd

    label_stats = _load_label_stats(args.label_stats)

    dl_preds = load_dl_preds(args.dl_preds_npy, args.dl_preds_meta)
    print(f"DL predictions: {dl_preds.shape[0]} molecules x {dl_preds.shape[1]} labels")

    ilp_preds = load_ilp_preds(args.ilp_preds_npy, args.ilp_preds_meta)
    print(f"ILP predictions: {ilp_preds.shape[0]} molecules x {ilp_preds.shape[1]} classes")

    with open(args.trusted_models) as f:
        lines = [line.strip().split(",") for line in f if line.strip()]
    if lines[0] == ["chebi_id", "model"]:
        trusted_model = {line[0]: [model.replace("'", "").replace("[", "").replace("]", "").strip() for model in line[1:]] for line in lines[1:]}
        model_weights_dict = None
        print(f"Loaded trusted models: {len(trusted_model)} classes")
    else:
        tm_df = pd.read_csv(args.trusted_models, dtype=str)
        trusted_model = None
        tm_df = tm_df.set_index("chebi_id")
        model_weights_dict = {
            cls_id: {col: float(val) for col, val in row.items()}
            for cls_id, row in tm_df.iterrows()
        }
        print(f"Loaded model weights: {len(model_weights_dict)} classes")

    data_dir = os.path.join("data", f"chebi_v{args.chebi_version}")
    import pickle as _pickle
    with open(os.path.join(data_dir, "chebi_graph.pkl"), "rb") as _f:
        chebi_graph = _pickle.load(_f)

    aggregator = EnsembleAggregator(
        dl_preds=dl_preds,
        ilp_preds=ilp_preds,
        label_stats=label_stats,
        chebi_graph=chebi_graph,
        trusted_model=trusted_model,
        model_weights=model_weights_dict,
    )

    mol_ids = dl_preds.index.tolist()
    print(f"Predicting on {len(mol_ids)} molecules...")

    predictions_df = aggregator.predict_set(mol_ids)

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

    print(f"Config for test run saved to {os.path.join(results_dir, 'config.yml')}")

    with tee_output(log_path):
        test_chebi_classes(args.run_to_evaluate, config["problem_dir"], config["predicate_set"], results_dir, selection_mode=config["selection_mode"], selection_k=config["selection_k"], test_on=args.test_on, verbose=args.verbose)


# ── Argument parsing ─────────────────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by all subcommands that build an ILPProblemBuilder."""
    parser.add_argument("--labels_file", type=str, required=True, help="Path to the labels file (one ChEBI ID per line).")
    parser.add_argument("--chebi_split", type=str, required=True, help="Path to the ChEBI split file (mol_id,split CSV).")
    parser.add_argument("--fg_mode", action="store_true", help="Learn functional groups instead of ChEBI classes.")
    parser.add_argument("--chebi_graph_path", type=str, required=True,
                        help="Path to chebi_graph.pkl.")
    parser.add_argument("--molecules_path", type=str, default=True,
                        help="Path to molecules.pkl.")
    parser.add_argument("--predicate_set", type=str, default="atoms", choices=typing.get_args(AVAILABLE_PREDICATE_SETS), help="Which predicate set to use for background knowledge.")
   

def _handle_prepare_dataset(args):
    from chebILP.data_preparation import ChEBIDataset
    ChEBIDataset.prepare(
        chebi_version=args.chebi_version,
        three_star_only=not args.include_two_star,
        data_dir=args.data_dir,
        min_pos_samples=args.min_pos_samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        labels_path=args.labels_path,
        splits_path=args.splits_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ILP classification CLI for ChEBI classes using Popper.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── prepare_dataset ──────────────────────────────────────────────────
    sp_pd = subparsers.add_parser(
        "prepare_dataset",
        help="Download ChEBI data and build all dataset artefacts "
             "(graph cache, molecule cache, labels.txt, splits.csv).",
    )
    sp_pd.add_argument("--chebi_version", type=int, required=True,
                       help="ChEBI ontology version (e.g. 248).")
    sp_pd.add_argument("--include_two_star", action="store_true",
                       help="Include classes with 2-star or 3-star annotation status (default: Only 3-star classes).")
    sp_pd.add_argument("--data_dir", type=str, default=None,
                       help="Root directory for raw and cached files "
                            "(default: data/chebi_v{version}).")
    sp_pd.add_argument("--min_pos_samples", type=int, default=50,
                       help="Minimum descendant molecules per label class (default: 50).")
    sp_pd.add_argument("--val_ratio", type=float, default=0.1,
                       help="Fraction of molecules for validation (default: 0.1).")
    sp_pd.add_argument("--test_ratio", type=float, default=0.1,
                       help="Fraction of molecules for test (default: 0.1).")
    sp_pd.add_argument("--seed", type=int, default=42,
                       help="Random seed for splits (default: 42).")
    sp_pd.add_argument("--labels_path", type=str, default=None,
                       help="Output path for labels.txt "
                            "(default: {data_dir}/min{n}/labels.txt).")
    sp_pd.add_argument("--splits_path", type=str, default=None,
                       help="Output path for splits.csv "
                            "(default: {data_dir}/min{n}/splits.csv).")
    sp_pd.set_defaults(func=_handle_prepare_dataset)

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
    sp_learn.add_argument("--labels_file", type=str, required=True, help="Path to the labels file (one ChEBI ID per line).")
    sp_learn.add_argument("--timeout", type=int, default=20, help="Timeout for ILP solver in seconds.")
    sp_learn.add_argument("--predicate_set", type=str, default="atoms", choices=typing.get_args(AVAILABLE_PREDICATE_SETS), help="Which predicate set to use.")
    sp_learn.add_argument("--fg_mode", action="store_true", help="Learn functional groups instead of ChEBI classes.")
    sp_learn.add_argument("--max_pos_samples", type=int, default=200, help="Maximum positive samples per class.")
    sp_learn.add_argument("--max_neg_samples", type=int, default=200, help="Maximum negative samples per class.")
    sp_learn.add_argument("--selection_mode", type=str, default=None, choices=["claude", "random", "top_k"], help="Mode for selecting body predicates in bias file.")
    sp_learn.add_argument("--selection_k", type=int, default=10, help="Number of predicates selection with selection_mode (required if selection_mode is set).")
    sp_learn.add_argument("--max_vars", type=int, default=6, help="Maximum number of variables in learned rules.")
    sp_learn.add_argument("--max_body", type=int, default=8, help="Maximum number of body literals in learned rules.")
    sp_learn.add_argument("--max_clauses", type=int, default=2, help="Maximum number of clauses in the learned program.")
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
    sp_test.add_argument("--test_on", type=str, default="test", choices=["validation", "test"], help="Split to evaluate on: 'test' (default) or 'validation' (validation).")
    sp_test.add_argument("--verbose", action="store_true", help="Log classification result for up to 10 positive and negative samples per class.")
    sp_test.set_defaults(func=_handle_test)

    # ── build_ilp_preds_for_ensemble ─────────────────────────────────────
    sp_bipe = subparsers.add_parser(
        "build_ilp_preds_for_ensemble",
        help="Build a full ILP predictions tensor for a split using programs from a run's results.json. "
             "Builds background knowledge from scratch (not from bk.pl files). "
             "Output is saved as full_val_preds.npy / full_test_preds.npy inside the run directory.",
    )
    sp_bipe.add_argument("--run_dir", type=str, required=True,
                         help="Run directory containing results.json (output of 'learn' or 'test').")
    sp_bipe.add_argument("--predict_on", type=str, default="validation",
                         choices=["validation", "test"],
                         help="Split to build predictions for (default: validation).")
    sp_bipe.add_argument("--chebi_split", type=str, required=True,
                         help="Path to the splits CSV (mol_id,split).")
    sp_bipe.add_argument("--chebi_version", type=int, default=248,
                         help="ChEBI version; used to derive default molecules_path.")
    sp_bipe.add_argument("--molecules_path", type=str, default=None,
                         help="Path to molecules.pkl (default: data/chebi_v{version}/molecules.pkl).")
    sp_bipe.set_defaults(func=_handle_build_ilp_preds_for_ensemble)

    # ── ensemble_construct ───────────────────────────────────────────────
    sp_ec = subparsers.add_parser(
        "ensemble_construct",
        help="Perform model selection and generate the ILP predictions tensor.",
    )
    sp_ec.add_argument("--chebi_version", type=int, default=248)
    sp_ec.add_argument("--chebi_split", type=str, required=True,
                       help="Path to the splits CSV (mol_id,split); used to obtain the molecule list for the ILP tensor.")
    sp_ec.add_argument("--predict_on", type=str, default="validation",
                       choices=["validation", "test"],
                       help="Which split to build the ILP predictions tensor for (default: validation).")
    sp_ec.add_argument("--dl_val_preds_npy", type=str, required=True,
                       help="DL validation predictions .npy (mol × class float scores).")
    sp_ec.add_argument("--dl_val_preds_meta", type=str, required=True,
                       help="Metadata JSON for --dl_val_preds_npy.")
    sp_ec.add_argument("--ilp_val_runs", type=str, nargs="+", default=[],
                       help="Validation-run directories (output of 'test --test_on validation'). "
                            "Each must contain full_val_preds.npy + full_val_preds_metadata.json "
                            "and results.json (for ILP programs).")
    sp_ec.add_argument("--label_stats", type=str, default=os.path.join("data", "chebi_v248", "ChEBI25_3_STAR", "processed", "class_stats.csv"),
                       help="Class statistics CSV (label list + has_negatives flag).")
    sp_ec.add_argument("--output", type=str,
                       default=os.path.join("data", "ensemble_predictions", "ensemble_f1"),
                       help="Base output path. Suffixes _trusted_models.csv, _ilp_preds.npy, _ilp_preds_metadata.json are appended.")
    sp_ec.set_defaults(func=_handle_ensemble_construct)

    # ── ensemble_aggregate ───────────────────────────────────────────────
    sp_ea = subparsers.add_parser(
        "ensemble_aggregate",
        help="Aggregate pre-computed DL and ILP tensors into ensemble predictions.",
    )
    sp_ea.add_argument("--chebi_version", type=int, default=248)
    sp_ea.add_argument("--dl_preds_npy", type=str, required=True,
                       help="DL predictions .npy for the target split.")
    sp_ea.add_argument("--dl_preds_meta", type=str, required=True,
                       help="Metadata JSON for --dl_preds_npy.")
    sp_ea.add_argument("--ilp_preds_npy", type=str, required=True,
                       help="ILP predictions tensor .npy (from ensemble_construct).")
    sp_ea.add_argument("--ilp_preds_meta", type=str, required=True,
                       help="Metadata JSON for --ilp_preds_npy.")
    sp_ea.add_argument("--trusted_models", type=str, required=True,
                       help="Trusted models CSV (_trusted_models.csv) or model weights CSV (_model_weights.csv) from ensemble_construct.")
    sp_ea.add_argument("--label_stats", type=str, default=os.path.join("data", "chebi_v248", "ChEBI25_3_STAR", "processed", "class_stats.csv"),
                       help="Class statistics CSV (label list + has_negatives flag).")
    sp_ea.add_argument("--output", type=str,
                       default=os.path.join("data", "ensemble_predictions", "ensemble_predictions.npy"),
                       help="Output .npy path; a matching _metadata.json is written alongside.")
    sp_ea.set_defaults(func=_handle_ensemble_aggregate)

    # ── explain ──────────────────────────────────────────────────────────────
    sp_explain = subparsers.add_parser(
        "explain",
        help="Explain why a molecule satisfies a learned ILP rule using xclingo.",
    )
    smiles_group = sp_explain.add_mutually_exclusive_group(required=True)
    smiles_group.add_argument("--smiles", type=str, help="SMILES string of the molecule to explain.")
    smiles_group.add_argument("--smiles_file", type=str, help="File containing a single SMILES string.")
    rule_group = sp_explain.add_mutually_exclusive_group(required=True)
    rule_group.add_argument("--rule", type=str, help="ILP rule clause(s) as a string.")
    rule_group.add_argument("--rule_file", type=str, help="File containing ILP rule clause(s).")
    sp_explain.add_argument("--chebi_graph_path", type=str, default=None, help="Path to chebi_graph.pkl for class name and parent lookup (optional).")
    sp_explain.add_argument("--output", type=str, default=None, help="Path to save the molecule visualization image (PNG).")
    sp_explain.add_argument("--verbose", "-v", action="store_true", help="Print the assembled xclingo program before running.")
    sp_explain.set_defaults(func=_handle_explain)

    # ── rule_to_nl ───────────────────────────────────────────────────────────
    sp_rtnl = subparsers.add_parser(
        "rule_to_nl",
        help="Translate a learned ILP rule to a natural language description.",
    )
    rule_group_rtnl = sp_rtnl.add_mutually_exclusive_group(required=True)
    rule_group_rtnl.add_argument("--rule", type=str, help="ILP rule clause(s) as a string.")
    rule_group_rtnl.add_argument("--rule_file", type=str, help="File containing ILP rule clause(s).")
    sp_rtnl.add_argument(
        "--chebi_graph_path", type=str,
        default=None,
        help="Path to chebi_graph.pkl for class name and parent lookup (optional).",
    )
    sp_rtnl.set_defaults(func=_handle_rule_to_nl)

    return parser


def _handle_rule_to_nl(args):
    from chebILP.rule_to_nl import translate_rule

    rule = args.rule
    if rule is None:
        with open(args.rule_file, "r") as f:
            rule = f.read()

    chebi_graph = None
    if args.chebi_graph_path and os.path.exists(args.chebi_graph_path):
        import pickle as _pickle
        with open(args.chebi_graph_path, "rb") as f:
            chebi_graph = _pickle.load(f)
    elif args.chebi_graph_path:
        print(f"Warning: chebi_graph_path '{args.chebi_graph_path}' does not exist. Proceeding without ChEBI graph.")

    print(translate_rule(rule, chebi_graph=chebi_graph))


def _handle_explain(args):
    from chebILP.explain import explain_molecule

    smiles = args.smiles
    if smiles is None:
        with open(args.smiles_file, "r") as f:
            smiles = f.read().strip()

    rule = args.rule
    if rule is None:
        with open(args.rule_file, "r") as f:
            rule = f.read()

    chebi_graph = None
    if args.chebi_graph_path and os.path.exists(args.chebi_graph_path):
        import pickle as _pickle
        with open(args.chebi_graph_path, "rb") as f:
            chebi_graph = _pickle.load(f)
    elif args.chebi_graph_path:
        print(f"Warning: chebi_graph_path '{args.chebi_graph_path}' does not exist. Proceeding without ChEBI graph.")

    satisfies, explanation_text, _ = explain_molecule(
        smiles=smiles,
        rule=rule,
        chebi_graph=chebi_graph,
        output_path=args.output,
        verbose=args.verbose,
    )

    print(explanation_text)

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
