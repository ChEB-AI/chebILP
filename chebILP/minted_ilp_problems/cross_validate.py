"""k-fold cross-validation of Popper or Aleph on the minted problems.

Folds group examples by their first argument, so all examples of one state are
held out together. Each fold is written to ``<results_dir>/<problem>/fold<i>/{train,validation}/``;
per-fold scores and the pooled validation confusion matrix go to ``<results_dir>/results.json``.

Usage (from repo root):
    wsl -e bash -lc '.wslvenv/bin/python -m chebILP.minted_ilp_problems.cross_validate --tool popper'
"""
import argparse
import json
import os
import random
import time

from chebILP.aleph_runner import run_ilp_training_aleph
from chebILP.evaluation.clingo_eval import run_ilp_validation_clingo
from chebILP.ilp_classifier import run_ilp_training_subprocess
from chebILP.ilp_problem import ILPProblem
from chebILP.minted_ilp_problems.build_minted_problems import MINTERS
from popper.util import MAX_BODY, MAX_VARS


def split_folds(problem: ILPProblem, k: int, fold_seed: int) -> list[tuple[ILPProblem, ILPProblem]]:
    """(train, validation) per fold. Both keep the full BK."""
    first = lambda s: s[0] if isinstance(s, tuple) else s
    entities = sorted({first(s) for s in problem.pos_samples + problem.neg_samples}, key=str)
    random.Random(fold_seed).shuffle(entities)
    fold_of = {e: i % k for i, e in enumerate(entities)}
    folds = []
    for i in range(k):
        held_out = lambda s: fold_of[first(s)] == i
        folds.append(tuple(
            ILPProblem(problem.head_predicate,
                       [s for s in problem.pos_samples if held_out(s) == is_val],
                       [s for s in problem.neg_samples if held_out(s) == is_val],
                       problem.bk_facts)
            for is_val in (False, True)))
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", choices=["popper", "aleph"], default="popper")
    ap.add_argument("--seed", type=int, default=20260907, help="minting seed")
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--nuwls", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--problem_type", default=None)
    args = ap.parse_args()

    results_dir = args.results_dir or os.path.join("data", "results", "minted_cv", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    settings = {"noisy": False, "nuwls": args.nuwls, "timeout": args.timeout}

    results = {}
    for name, (minter_cls, minter_kwargs) in MINTERS.items():
        if args.problem_type is not None and args.problem_type != minter_cls.problem_type:
            continue
        minter = minter_cls(**minter_kwargs)
        max_vars = max(minter.max_vars(), MAX_VARS)
        max_body = max(minter.max_body(), MAX_BODY)
        problem = minter.mint(args.seed)
        head, arity = problem.head_predicate
        fold_results = []
        pooled = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for i, (train, val) in enumerate(split_folds(problem, args.k_folds, args.fold_seed)):
            fold_dir = os.path.join(results_dir, name, f"fold{i}")
            train_dir, val_dir = os.path.join(fold_dir, "train"), os.path.join(fold_dir, "validation")
            train.write_popper(train_dir)
            val.write_popper(val_dir)
            bias_path = os.path.join(train_dir, "bias.pl")
            with open(bias_path) as f:
                bias = f.read()
            bias = bias.replace(f"%% max_vars(TODO).", f"max_vars({max_vars}).")
            bias = bias.replace(f"%% max_body(TODO).", f"max_body({max_body}).")
            with open(bias_path, "w") as f:
                f.write(bias)

            print(f"[{name}] fold {i}: {len(train.pos_samples)}+/{len(train.neg_samples)}- train, "
                  f"{len(val.pos_samples)}+/{len(val.neg_samples)}- validation")
            start = time.perf_counter()
            if args.tool == "aleph":
                train.write_aleph(train_dir)
                out = run_ilp_training_aleph(None, os.path.join(train_dir, head), bias_path, timeout=args.timeout,
                                             max_body=max_body, log_dir=fold_dir, head_name=head,
                                             head_arity=arity, run_stem_name=head)
            else:
                out = run_ilp_training_subprocess(os.path.join(train_dir, "exs.pl"), os.path.join(train_dir, "bk.pl"),
                                                  bias_path, settings, log_dir=fold_dir)
            elapsed = time.perf_counter() - start

            val_cm = None
            if out["prog_str"]:
                val_cm = run_ilp_validation_clingo(head, out["prog_str"], os.path.join(val_dir, "exs.pl"),
                                                   os.path.join(val_dir, "bk.pl"))
                for key in pooled:
                    pooled[key] += val_cm[key]
            score = out["score"]
            fold_results.append({
                "fold": i,
                "program": out["prog_str"],
                "train_score": dict(zip(("TP", "FN", "TN", "FP"), score)) if score else None,
                "validation_score": val_cm,
                "num_programs": out.get("num_programs"),
                "time_taken": elapsed,
            })
            print(f"    program:\n    {out['prog_str']}\n    validation: {val_cm}")
        results[name] = {"folds": fold_results, "pooled_validation_score": pooled}
        print(f"[{name}] pooled validation: {pooled}")

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
