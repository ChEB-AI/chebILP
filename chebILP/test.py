

import json
import os
import time
from chebILP.ilp_classifier import run_ilp_validation_subprocess
from chebILP.ilp_path_manager import get_bk_path, get_exs_path
from chebILP.mol2ilp import ILPProblemBuilder, tee_output
from chebILP.learn_fgs import FGILPProblemBuilder
from typing import Literal

def test_chebi_classes(run_to_evaluate, ilp_builder: ILPProblemBuilder, results_dir, **kwargs):
    
    with open(os.path.join(results_dir, "config.yml"), "a+") as f:
        f.write(f"problem_dir: {ilp_builder.problem_dir}\n")
        f.write(f"run_to_evaluate: {run_to_evaluate}\n")

    with open(os.path.join(run_to_evaluate, "results.json"), "r") as f:
        results = []
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Failed to parse line: {line}")
        classes_list = [row["chebi_id"] for row in results]
        
    for row in results:
        chebi_id = row["chebi_id"]
        prog_str = row["program"]
        conf_matrix = None
        start_time = time.perf_counter()
        # Run validation in subprocess (isolated Prolog session)
        print(f"Testing ChEBI:{chebi_id}...")
        try:
            from chebILP.clingo_eval import run_ilp_validation_clingo
            conf_matrix = run_ilp_validation_clingo(
                chebi_id, prog_str,
                exs_file=get_exs_path(chebi_id, split="test", base_dir=ilp_builder.problem_dir),
                bk_file=get_bk_path(chebi_id, predicate_set=ilp_builder.predicate_set, split="test", base_dir=ilp_builder.problem_dir, 
                                    selection_mode=ilp_builder.selection_mode, selection_k=ilp_builder.selection_k),
            )
        except Exception as e:
            print(f"Testing failed for ChEBI:{chebi_id} with error: {e}")
            conf_matrix = None
        
        with open(os.path.join(results_dir, "results.json"), "a+") as f:
            result_entry = {
                "chebi_id": chebi_id,
                "time_taken": time.perf_counter() - start_time,
                "program": prog_str,
                "test_score": conf_matrix,
            }
            f.write(json.dumps(result_entry) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ILP classification on ChEBI classes using Popper.")
    parser.add_argument("--run_to_evaluate", type=str, default=None, help="Path to run to evaluate. Should contain a results.json and a config.yml file.")
    #parser.add_argument("--chebi_version", type=int, default=244, help="ChEBI version to use.")
    #parser.add_argument("--chebi_splits_file", type=str, default=None, help="Path to the ChEBI splits CSV file.")
    #parser.add_argument("--fg_mode", action="store_true", help="Evaluate run with functi.")
    #parser.add_argument("--predicate_set", type=str, default="atoms", choices=["atoms", "chembl_fgs"], help="Whether to include CHEMBL FG predicates in the background knowledge.")
    args = parser.parse_args()

    # load config file from run to evaluate
    with open(os.path.join(args.run_to_evaluate, "config.yml"), "r") as f:
        config = {}
        for line in f:
            if ": " in line:
                key, value = line.strip().split(": ", 1)
                config[key] = value
    assert "chebi_version" in config and "chebi_splits_file" in config and "fg_mode" in config and "predicate_set" in config, "Config file must contain chebi_version, chebi_splits_file, fg_mode, and predicate_set"
    
    chebi_splits_file = config["chebi_splits_file"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("ilp", "results_test", f"run_fgs_{timestamp}" if config["fg_mode"] == "True" else f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.json"), "w+") as f:
        f.write("")  # create empty results file

    log_path = os.path.join(results_dir, "run.log")

    # write config file with settings used for this run
    with open(os.path.join(results_dir, "config.yml"), "w+") as f:
        f.write(f"args:\n")
        for arg in vars(args):
            f.write(f"  {arg}: {getattr(args, arg)}\n")


    with tee_output(log_path):
        if config["fg_mode"] == "True":
            ilp_builder = FGILPProblemBuilder(chebi_version=config["chebi_version"], chebi_split=chebi_splits_file, dataset_path=os.path.join("data", "chebi_fgs_dataset.pkl"), predicate_set=config["predicate_set"])
        else:
            ilp_builder = ILPProblemBuilder(chebi_version=config["chebi_version"], chebi_split=chebi_splits_file, muggleton=False, predicate_set=config["predicate_set"])
    test_chebi_classes(args.run_to_evaluate, ilp_builder, results_dir, predicate_set=config["predicate_set"])
