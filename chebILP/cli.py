import os
import time
import argparse

from chebILP.utils import (
    tee_output,
    add_class_arguments,
    resolve_classes,
)
from popper.util import MAX_BODY, MAX_VARS

from chebILP.ilp_classifier import learn_chebi_classes


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_ilp_builder(args):
    from chebILP.preprocessing.base_problem_builder import ILPProblemBuilder
    from chebILP.preprocessing.registry import dataset_from_args

    return ILPProblemBuilder(
        dataset_from_args(args),
        problem_dir=args.problem_dir,
        aux_library_dir=getattr(args, "predicate_dir", None),
        computed_facts=getattr(args, "computed_facts", False),
        write_aleph=not getattr(args, "no_aleph", False),
    )


def _make_results_dir() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("data", "results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w+") as f:
        f.write("")
    return results_dir


def _read_run_config(run_dir: str) -> dict[str, str]:
    """``key: value`` lines of a run's config.yml (nested keys flattened)."""
    config = {}
    config_path = os.path.join(run_dir, "config.yml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                if ": " in line:
                    key, value = line.strip().split(": ", 1)
                    config[key] = value
    return config


def _library_from_config(config: dict[str, str]) -> str | None:
    """The LLM rule library a run was learned with, or None.

    Runs from before predicate sets were removed record ``predicate_set``: ``atoms`` had no
    library, ``llm_generated_rules`` used ``predicate_dir``. Other sets are no longer built.
    """
    library = config.get("predicate_dir")
    library = None if library in (None, "None") else library
    predicate_set = config.get("predicate_set")
    if predicate_set is None or predicate_set == "llm_generated_rules":
        return library
    if predicate_set == "atoms":
        return None
    raise SystemExit(
        f"Run uses predicate_set '{predicate_set}', which is no longer supported "
        "(only atoms and llm_generated_rules)."
    )


# ── Subcommand handlers ─────────────────────────────────────────────────────


def _handle_build_samples(args):
    classes = resolve_classes(args)
    ilp_builder = _make_ilp_builder(args)
    ilp_builder.build_examples(
        classes,
        max_pos_train=args.max_pos_samples,
        max_neg_train=args.max_neg_samples,
        min_pos=args.min_pos_samples,
        min_neg=args.min_neg_samples,
    )


def _handle_build_bk(args):
    classes = resolve_classes(args)
    ilp_builder = _make_ilp_builder(args)
    ilp_builder.build_bk(classes)


def _handle_learn(args):
    classes = resolve_classes(args)
    results_dir = _make_results_dir()
    log_path = os.path.join(results_dir, "run.log")

    # write config file
    with open(os.path.join(results_dir, "config.yml"), "w+") as f:
        f.write("args:\n")
        for arg in vars(args):
            f.write(f"  {arg}: {getattr(args, arg)}\n")

    with tee_output(log_path):
        learn_chebi_classes(
            classes,
            args.problem_dir,
            results_dir,
            timeout=args.timeout,
            max_vars=args.max_vars,
            max_body=args.max_body,
            mdl_weight_fn=args.mdl_weight_fn,
            mdl_weight_fp=args.mdl_weight_fp,
            mdl_weight_size=args.mdl_weight_size,
            seed_hypothesis=args.seed_hypothesis,
            heuristic_guidance=args.heuristic_guidance,
            aux_library_dir=args.predicate_dir,
            tool=args.tool,
            nuwls=args.nuwls,
            aux_pred_cost=args.aux_pred_cost,
            seed_pred_cost=args.seed_pred_cost,
            base_pred_cost=args.base_pred_cost,
        )


def _load_label_stats(path: str) -> list[str]:
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    if path.endswith(".txt"):
        return lines
    # CSV: first column is class ID, skip header
    return [line.split(",")[0] for line in lines[1:]]


def _handle_build_ilp_preds_for_ensemble(args):
    """Build a full ILP predictions tensor for a given split from a run's results.json."""
    from chebILP.evaluation.test import build_ilp_preds_tensor
    from chebILP.evaluation.ensemble_eval import load_ilp_results
    from chebILP.preprocessing.registry import dataset_from_args

    results = load_ilp_results(args.run_dir)
    programs = {
        cid: entry["program"] for cid, entry in results.items() if entry.get("program")
    }
    print(f"Loaded {len(programs)} ILP programs from {args.run_dir}")

    # Recover which rule library the run used so the background knowledge matches what the
    # programs were learned on. An explicit --predicate_dir overrides the config.
    run_config = _read_run_config(args.run_dir)
    aux_library_dir = args.predicate_dir or _library_from_config(run_config)
    print(f"Using rule library: {aux_library_dir}")

    mol_ids = []
    with open(args.chebi_split) as f:
        for line in f.readlines()[1:]:
            mol_id, split = line.strip().split(",")
            if split == args.predict_on:
                mol_ids.append(mol_id)
    print(f"Building predictions for {len(mol_ids)} '{args.predict_on}' examples")

    prefix = "val" if args.predict_on == "validation" else args.predict_on
    output_npy = os.path.join(args.run_dir, f"full_{prefix}_preds.npy")
    output_meta = os.path.join(args.run_dir, f"full_{prefix}_preds_metadata.json")

    computed_facts = (
        getattr(args, "computed_facts", False)
        or run_config.get("computed_facts") == "True"
    )

    build_ilp_preds_tensor(
        programs,
        dataset_from_args(args),
        mol_ids,
        output_npy,
        output_meta,
        aux_library_dir=aux_library_dir,
        label_timeout=args.label_timeout,
        n_jobs=args.n_jobs,
        computed_facts=computed_facts,
    )


def _handle_ensemble_construct(args):
    """Perform model selection and generate the ILP predictions tensor."""
    from chebILP.evaluation.ensemble_eval import EnsembleConstructor, load_dl_preds

    labels = _load_label_stats(args.labels_file)
    print(f"Label stats: {len(labels)} labels")

    dl_val_preds = load_dl_preds(args.dl_val_preds_npy, args.dl_val_preds_meta)
    print(
        f"DL val predictions: {dl_val_preds.shape[0]} molecules x {dl_val_preds.shape[1]} classes"
    )

    ilp_val_run_dirs = {os.path.basename(p.rstrip("/\\")): p for p in args.ilp_val_runs}
    data_dir = os.path.join("data", f"chebi_v{args.chebi_version}")
    import pickle as _pickle

    with open(os.path.join(data_dir, "chebi_graph.pkl"), "rb") as _f:
        chebi_graph = _pickle.load(_f)

    constructor = EnsembleConstructor(
        ilp_val_runs=ilp_val_run_dirs,
        dl_val_preds=dl_val_preds,
        label_stats=labels,
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
    from chebILP.evaluation.ensemble_eval import (
        EnsembleAggregator,
        load_dl_preds,
        load_ilp_preds,
    )
    import numpy as np
    import json as _json
    import pandas as pd

    labels = _load_label_stats(args.labels_file)

    dl_preds = load_dl_preds(args.dl_preds_npy, args.dl_preds_meta)
    print(f"DL predictions: {dl_preds.shape[0]} molecules x {dl_preds.shape[1]} labels")

    ilp_preds = load_ilp_preds(args.ilp_preds_npy, args.ilp_preds_meta)
    print(
        f"ILP predictions: {ilp_preds.shape[0]} molecules x {ilp_preds.shape[1]} classes"
    )

    with open(args.trusted_models) as f:
        lines = [line.strip().split(",") for line in f if line.strip()]
    if lines[0] == ["chebi_id", "model"]:
        trusted_model = {
            line[0]: [
                model.replace("'", "").replace("[", "").replace("]", "").strip()
                for model in line[1:]
            ]
            for line in lines[1:]
        }
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

    if trusted_model is not None and any(len(m) > 1 for m in trusted_model.values()):
        raise ValueError("Not supported: multiple trusted models per class.")
    single_trusted_model = (
        {cls_id: models[0] for cls_id, models in trusted_model.items()}
        if trusted_model is not None
        else None
    )

    aggregator = EnsembleAggregator(
        dl_preds=dl_preds,
        ilp_preds=ilp_preds,
        label_stats=labels,
        chebi_graph=chebi_graph,
        trusted_model=single_trusted_model,
        model_weights=model_weights_dict,
    )

    mol_ids = dl_preds.index.tolist()
    print(f"Predicting on {len(mol_ids)} molecules...")

    predictions_df = aggregator.predict_set(mol_ids)

    arr = predictions_df.to_numpy().astype("float32")
    meta = {
        "mol_order": list(predictions_df.index),
        "class_labels": list(predictions_df.columns),
    }

    npy_path = args.output if args.output.endswith(".npy") else args.output + ".npy"
    meta_path = npy_path.replace(".npy", "_metadata.json")

    os.makedirs(os.path.dirname(npy_path) or ".", exist_ok=True)
    np.save(npy_path, arr)
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)

    n_pos = int(arr.sum())
    print(
        f"Saved predictions: {npy_path}  (shape {arr.shape}, {n_pos} positive assignments)"
    )
    print(f"Saved metadata:    {meta_path}")


def _handle_test(args):
    from chebILP.evaluation.test import test_chebi_classes

    # load config from the run to evaluate
    config = _read_run_config(args.run_to_evaluate)
    assert "problem_dir" in config, "Config file must contain problem_dir"
    predicate_dir = _library_from_config(config)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("data", f"results_{args.test_on}", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w+") as f:
        f.write("")

    log_path = os.path.join(results_dir, "run.log")

    with open(os.path.join(results_dir, "config.yml"), "w+") as f:
        f.write("args:\n")
        for arg in vars(args):
            f.write(f"  {arg}: {getattr(args, arg)}\n")
        f.write(f"predicate_dir: {predicate_dir}\n")

    print(f"Config for test run saved to {os.path.join(results_dir, 'config.yml')}")

    with tee_output(log_path):
        test_chebi_classes(
            args.run_to_evaluate,
            config["problem_dir"],
            results_dir,
            test_on=args.test_on,
            verbose=args.verbose,
            predicate_dir=predicate_dir,
        )


def _handle_predict(args):
    import json
    from chebILP.evaluation.test import predict_smiles
    from chebILP.preprocessing.registry import dataset_class

    smiles_list = args.smiles
    if smiles_list is None:
        with open(args.smiles_file, "r") as f:
            smiles_list = [line.strip() for line in f if line.strip()]

    target_predicates = args.target_predicates
    if target_predicates is None:
        with open(args.targets_file, "r") as f:
            target_predicates = [line.strip() for line in f if line.strip()]

    # Only the ChEBI molecule encoding is used, so no dataset files are loaded.
    dataset = dataset_class("chebi")(version="predict")
    results = predict_smiles(
        smiles_list=smiles_list,
        rules_file=args.rules_file,
        target_predicates=target_predicates,
        dataset=dataset,
        verbose=args.verbose,
        aux_library_dir=args.predicate_dir,
        computed_facts=getattr(args, "computed_facts", False),
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Predictions saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


# ── Argument parsing ─────────────────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by all subcommands that build an ILPProblemBuilder."""
    from chebILP.preprocessing.registry import add_dataset_arguments

    add_class_arguments(parser)
    add_dataset_arguments(parser)
    parser.add_argument(
        "--problem_dir",
        default=None,
        help="ILP problem tree holding each class's examples / BK "
        "(default: <dataset dir>/ilp_problems, e.g. data/chebi_v251/ilp_problems).",
    )
    parser.add_argument(
        "--no_aleph",
        action="store_true",
        help="Skip emitting the Aleph-format train files (.b/.f/.n).",
    )


def _handle_prepare_dataset(args):
    from chebILP.preprocessing.registry import dataset_class

    dataset_class(args.dataset.lower()).prepare(
        version=args.version,
        base_dir=args.base_dir,
        min_pos_samples=args.min_pos_samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ILP classification CLI for ChEBI classes using Popper or Aleph.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── prepare_dataset ──────────────────────────────────────────────────
    sp_pd = subparsers.add_parser(
        "prepare_dataset",
        help="Download dataset files and build artefacts (label selection, train/val/test splits).",
    )
    sp_pd.add_argument(
        "dataset",
        choices=["chebi", "rhea", "scope"],
        help="Which dataset to prepare.",
    )
    sp_pd.add_argument(
        "--version",
        "-v",
        type=str,
        required=True,
        help="Dataset version (e.g. 251 for ChEBI, 142 for Rhea, 2.08 for SCOPe).",
    )
    sp_pd.add_argument(
        "--base_dir",
        type=str,
        default="data",
        help="Root directory for the dataset (default: data).",
    )
    sp_pd.add_argument(
        "--min_pos_samples",
        type=int,
        default=25,
        help="Minimum descendant molecules per label class (default: 25).",
    )
    sp_pd.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of molecules held back for validation (default: 0.2).",
    )
    sp_pd.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of molecules held back for testing (default: 0.2).",
    )
    sp_pd.add_argument(
        "--seed", type=int, default=42, help="Random seed for splits (default: 42)."
    )
    sp_pd.set_defaults(func=_handle_prepare_dataset)

    # ── build_samples ────────────────────────────────────────────────────
    sp_samples = subparsers.add_parser(
        "build_samples",
        help="Build positive/negative example files (exs.pl) for the given ChEBI classes.",
    )
    _add_common_args(sp_samples)
    sp_samples.add_argument(
        "--max_pos_samples",
        type=int,
        default=200,
        help="Maximum positive training samples per class.",
    )
    sp_samples.add_argument(
        "--min_neg_samples",
        type=int,
        default=25,
        help="Minimum negative samples per class.",
    )
    sp_samples.add_argument(
        "--max_neg_samples",
        type=int,
        default=200,
        help="Maximum negative training samples per class.",
    )

    sp_samples.set_defaults(func=_handle_build_samples)

    # ── build_bk ─────────────────────────────────────────────────────────
    sp_bk = subparsers.add_parser(
        "build_bk",
        help="Build background knowledge (bk.pl) and bias template (bias.pl) for the given ChEBI classes.",
    )
    _add_common_args(sp_bk)
    sp_bk.add_argument(
        "--predicate_dir",
        type=str,
        default=None,
        help="LLM rule library whose class rules are ground into the BK (default: none, "
        "base encoding only). The BK is written to llm_generated_rules_<library name>/, "
        "so libraries can coexist.",
    )
    sp_bk.add_argument(
        "--computed_facts",
        action="store_true",
        help="Compute the dataset's numeric facts (e.g. mol_weight/ring_size) to ground "
        "the rules against (kept out of bk.pl; only aux_* extensions are saved).",
    )
    sp_bk.set_defaults(func=_handle_build_bk)

    # ── learn ────────────────────────────────────────────────────────────
    sp_learn = subparsers.add_parser(
        "learn",
        help="Run ILP learning (training + validation) for the given ChEBI classes.",
    )
    add_class_arguments(sp_learn)
    sp_learn.add_argument(
        "--problem_dir",
        type=str,
        required=True,
        help="Base directory of the ILP problems (as given to build_samples / build_bk).",
    )
    sp_learn.add_argument(
        "--tool",
        type=str,
        default="popper",
        choices=["popper", "aleph"],
        help="ILP engine: 'popper' (default) or 'aleph'.",
    )
    sp_learn.add_argument(
        "--timeout", type=int, default=20, help="Timeout for ILP solver in seconds."
    )
    sp_learn.add_argument(
        "--max_vars",
        type=int,
        default=MAX_VARS,
        help="Maximum number of variables in learned rules.",
    )
    sp_learn.add_argument(
        "--max_body",
        type=int,
        default=MAX_BODY,
        help="Maximum number of body literals in learned rules.",
    )
    sp_learn.add_argument(
        "--nuwls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Popper's NuWLS anytime MaxSAT solver (default: on). Needs the NuWLS-c binary on PATH; "
        "see the Popper README.",
    )
    _mdl_note = " Not actively maintained: upstream Popper silently ignores it."
    sp_learn.add_argument(
        "--mdl_weight_fn",
        type=int,
        default=1,
        help="Weight β for false negatives in MDL cost (default: 1)." + _mdl_note,
    )
    sp_learn.add_argument(
        "--mdl_weight_fp",
        type=int,
        default=1,
        help="Weight γ for false positives in MDL cost (default: 1)." + _mdl_note,
    )
    sp_learn.add_argument(
        "--mdl_weight_size",
        type=int,
        default=1,
        help="Weight α for program size in MDL cost (default: 1)." + _mdl_note,
    )
    _cost_note = " Replaces the cost of 1 per literal in Popper's MDL size term; needs the feature/predicate-costs Popper branch."
    sp_learn.add_argument(
        "--aux_pred_cost",
        type=int,
        default=1,
        help="Cost of an LLM-generated aux_* body literal (default: 1)." + _cost_note,
    )
    sp_learn.add_argument(
        "--seed_pred_cost",
        type=int,
        default=1,
        help="Cost of a seeded functional-group body literal (efg_, efga_, chembl_fg_; default: 1)."
        + _cost_note,
    )
    sp_learn.add_argument(
        "--base_pred_cost",
        type=int,
        default=1,
        help="Cost of any other body literal, i.e. the atom-level primitives (default: 1)."
        + _cost_note,
    )
    sp_learn.add_argument(
        "--predicate_dir",
        type=str,
        default=None,
        help="LLM rule library the BK was built with (default: none). Selects the BK folder "
        "build_bk wrote for that library, and is read by --seed_hypothesis / --heuristic_guidance.",
    )
    sp_learn.add_argument(
        "--seed_hypothesis",
        action="store_true",
        help="Seed Aleph's search with the LLM's saved class hypothesis, when it grounds "
        "and fits the bias (--tool aleph with --predicate_dir only).",
    )
    sp_learn.add_argument(
        "--heuristic_guidance",
        action="store_true",
        help="Steer generation toward the predicates in the LLM's class hypothesis via "
        "prefer_body_pred directives in the bias file (with --predicate_dir only)."
        + _mdl_note,
    )
    sp_learn.set_defaults(func=_handle_learn)

    # ── test ─────────────────────────────────────────────────────────────
    sp_test = subparsers.add_parser(
        "test",
        help="Evaluate learned programs on the test set using results from a previous run.",
    )
    sp_test.add_argument(
        "--run_to_evaluate",
        type=str,
        required=True,
        help="Path to a previous run directory (must contain results.json and config.yml).",
    )
    sp_test.add_argument(
        "--test_on",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Split to evaluate on: 'test' (default), 'validation', or 'train'.",
    )
    sp_test.add_argument(
        "--verbose",
        action="store_true",
        help="Log classification result for up to 10 positive and negative samples per class.",
    )
    sp_test.set_defaults(func=_handle_test)

    # ── predict ──────────────────────────────────────────────────────────
    sp_predict = subparsers.add_parser(
        "predict",
        help="Predict target predicates for a list of SMILES using a rule set, "
        "without depending on a previous run or split.",
    )
    smiles_group_pred = sp_predict.add_mutually_exclusive_group(required=True)
    smiles_group_pred.add_argument(
        "--smiles", type=str, nargs="+", help="One or more SMILES strings to classify."
    )
    smiles_group_pred.add_argument(
        "--smiles_file", type=str, help="File with one SMILES string per line."
    )
    sp_predict.add_argument(
        "--rules_file",
        type=str,
        required=True,
        help="File containing logic programs (rule clauses).",
    )
    targets_group_pred = sp_predict.add_mutually_exclusive_group(required=True)
    targets_group_pred.add_argument(
        "--target_predicates",
        type=str,
        nargs="+",
        help="Target head predicate names (e.g. chebi_35341).",
    )
    targets_group_pred.add_argument(
        "--targets_file", type=str, help="File with one target predicate name per line."
    )
    sp_predict.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save predictions as JSON.",
    )
    sp_predict.add_argument(
        "--verbose",
        action="store_true",
        help="Print satisfied predicates per molecule.",
    )
    sp_predict.add_argument(
        "--predicate_dir",
        type=str,
        default=None,
        help="LLM rule library the rules were learned with (default: none).",
    )
    sp_predict.add_argument(
        "--computed_facts",
        action="store_true",
        help="Recompute mol_weight/ring_size facts when grounding the library rules "
        "(must match how the BK was built).",
    )
    sp_predict.set_defaults(func=_handle_predict)

    # ── build_ilp_preds_for_ensemble ─────────────────────────────────────
    sp_bipe = subparsers.add_parser(
        "build_ilp_preds_for_ensemble",
        help="Build a full ILP predictions tensor for a split using programs from a run's results.json. "
        "Builds background knowledge from scratch (not from bk.pl files). "
        "Output is saved as full_val_preds.npy / full_test_preds.npy inside the run directory.",
    )
    sp_bipe.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory containing results.json (output of 'learn' or 'test').",
    )
    sp_bipe.add_argument(
        "--predict_on",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Split to build predictions for (default: validation).",
    )
    sp_bipe.add_argument(
        "--chebi_split",
        type=str,
        required=True,
        help="Path to the splits CSV (mol_id,split).",
    )
    from chebILP.preprocessing.registry import add_dataset_arguments

    add_dataset_arguments(sp_bipe)
    sp_bipe.add_argument(
        "--label_timeout",
        type=float,
        default=60.0,
        help="Per-label Clingo solving timeout in seconds (default: 60).",
    )
    sp_bipe.add_argument(
        "--predicate_dir",
        type=str,
        default=None,
        help="LLM rule library the programs were learned with. Default: read from the "
        "run's config.yml.",
    )
    sp_bipe.add_argument(
        "--computed_facts",
        action="store_true",
        help="For llm_generated_rules: recompute mol_weight/ring_size facts when "
        "grounding the rules. Default: read from the run's config.yml.",
    )
    sp_bipe.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Worker processes for evaluating molecules in parallel. "
        "1 (default) runs serially; <= 0 uses all CPU cores.",
    )
    sp_bipe.set_defaults(func=_handle_build_ilp_preds_for_ensemble)

    # ── ensemble_construct ───────────────────────────────────────────────
    sp_ec = subparsers.add_parser(
        "ensemble_construct",
        help="Perform model selection and generate the ILP predictions tensor.",
    )
    sp_ec.add_argument("--chebi_version", type=int, default=248)
    sp_ec.add_argument(
        "--chebi_split",
        type=str,
        required=True,
        help="Path to the splits CSV (mol_id,split); used to obtain the molecule list for the ILP tensor.",
    )
    sp_ec.add_argument(
        "--predict_on",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Which split to build the ILP predictions tensor for (default: validation).",
    )
    sp_ec.add_argument(
        "--dl_val_preds_npy",
        type=str,
        required=True,
        help="DL validation predictions .npy (mol × class float scores).",
    )
    sp_ec.add_argument(
        "--dl_val_preds_meta",
        type=str,
        required=True,
        help="Metadata JSON for --dl_val_preds_npy.",
    )
    sp_ec.add_argument(
        "--ilp_val_runs",
        type=str,
        nargs="+",
        default=[],
        help="Validation-run directories (output of 'test --test_on validation'). "
        "Each must contain full_val_preds.npy + full_val_preds_metadata.json "
        "and results.json (for ILP programs).",
    )
    sp_ec.add_argument(
        "--labels_file",
        type=str,
        default=os.path.join("data", "chebi_v248", "processed_25_3_STAR", "labels.txt"),
        help="Class statistics CSV (label list + has_negatives flag).",
    )
    sp_ec.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "ensemble_predictions", "ensemble_f1"),
        help="Base output path. Suffixes _trusted_models.csv, _ilp_preds.npy, _ilp_preds_metadata.json are appended.",
    )
    sp_ec.set_defaults(func=_handle_ensemble_construct)

    # ── ensemble_aggregate ───────────────────────────────────────────────
    sp_ea = subparsers.add_parser(
        "ensemble_aggregate",
        help="Aggregate pre-computed DL and ILP tensors into ensemble predictions.",
    )
    sp_ea.add_argument("--chebi_version", type=int, default=248)
    sp_ea.add_argument(
        "--dl_preds_npy",
        type=str,
        required=True,
        help="DL predictions .npy for the target split.",
    )
    sp_ea.add_argument(
        "--dl_preds_meta",
        type=str,
        required=True,
        help="Metadata JSON for --dl_preds_npy.",
    )
    sp_ea.add_argument(
        "--ilp_preds_npy",
        type=str,
        required=True,
        help="ILP predictions tensor .npy (from ensemble_construct).",
    )
    sp_ea.add_argument(
        "--ilp_preds_meta",
        type=str,
        required=True,
        help="Metadata JSON for --ilp_preds_npy.",
    )
    sp_ea.add_argument(
        "--trusted_models",
        type=str,
        required=True,
        help="Trusted models CSV (_trusted_models.csv) or model weights CSV (_model_weights.csv) from ensemble_construct.",
    )
    sp_ea.add_argument(
        "--labels_file",
        type=str,
        default=os.path.join("data", "chebi_v248", "processed_25_3_STAR", "labels.txt"),
        help="File containing the list of labels.",
    )
    sp_ea.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            "data", "ensemble_predictions", "ensemble_predictions.npy"
        ),
        help="Output .npy path; a matching _metadata.json is written alongside.",
    )
    sp_ea.set_defaults(func=_handle_ensemble_aggregate)

    # ── explain ──────────────────────────────────────────────────────────────
    sp_explain = subparsers.add_parser(
        "explain",
        help="Explain why a molecule satisfies a learned ILP rule using xclingo.",
    )
    smiles_group = sp_explain.add_mutually_exclusive_group(required=True)
    smiles_group.add_argument(
        "--smiles", type=str, help="SMILES string of the molecule to explain."
    )
    smiles_group.add_argument(
        "--smiles_file", type=str, help="File containing a single SMILES string."
    )
    rule_group = sp_explain.add_mutually_exclusive_group(required=True)
    rule_group.add_argument("--rule", type=str, help="ILP rule clause(s) as a string.")
    rule_group.add_argument(
        "--rule_file", type=str, help="File containing ILP rule clause(s)."
    )
    sp_explain.add_argument(
        "--chebi_graph_path",
        type=str,
        default=None,
        help="Path to chebi_graph.pkl for class name and parent lookup (optional).",
    )
    sp_explain.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the molecule visualization image (PNG).",
    )
    sp_explain.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the assembled xclingo program before running.",
    )
    sp_explain.set_defaults(func=_handle_explain)

    # ── rule_to_nl ───────────────────────────────────────────────────────────
    sp_rtnl = subparsers.add_parser(
        "rule_to_nl",
        help="Translate a learned ILP rule to a natural language description.",
    )
    rule_group_rtnl = sp_rtnl.add_mutually_exclusive_group(required=True)
    rule_group_rtnl.add_argument(
        "--rule", type=str, help="ILP rule clause(s) as a string."
    )
    rule_group_rtnl.add_argument(
        "--rule_file", type=str, help="File containing ILP rule clause(s)."
    )
    sp_rtnl.add_argument(
        "--chebi_graph_path",
        type=str,
        default=None,
        help="Path to chebi_graph.pkl for class name and parent lookup (optional).",
    )
    sp_rtnl.add_argument(
        "--aux_library_dir",
        type=str,
        default=None,
        help="Auxiliary library directory (llm_generated_rules/... ) whose aux_* predicate "
        "descriptions annotate the LLM-generated predicates in the rule (optional).",
    )
    sp_rtnl.set_defaults(func=_handle_rule_to_nl)

    return parser


def _handle_rule_to_nl(args):
    from chebILP.explainability.rule_to_nl import translate_rule

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
        print(
            f"Warning: chebi_graph_path '{args.chebi_graph_path}' does not exist. Proceeding without ChEBI graph."
        )

    print(
        translate_rule(
            rule, chebi_graph=chebi_graph, aux_library_dir=args.aux_library_dir
        )
    )


def _handle_explain(args):
    from chebILP.explainability.explain import explain_molecule

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
        print(
            f"Warning: chebi_graph_path '{args.chebi_graph_path}' does not exist. Proceeding without ChEBI graph."
        )

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
