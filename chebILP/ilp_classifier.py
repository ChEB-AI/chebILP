import os
import subprocess
import sys
import json
from datetime import datetime
import time
import traceback
import tqdm
from typing import Literal, Optional

from chebILP.ilp_path_manager import get_bias_path, get_bk_path, get_exs_path
from chebILP.ilp_problem_builder import AVAILABLE_PREDICATE_SETS

def log_subprocess_output(log_dir, phase, result):
    """Write subprocess stdout/stderr to the run log with timestamp."""
    if not log_dir:
        return
    log_file = os.path.join(log_dir, "run.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(result, str):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{timestamp}] === {phase} ===\n")
            for line in result.splitlines():
                f.write(f"[{timestamp}] {line}\n")
        return
    if not result.stdout.strip() and not result.stderr.strip():
        return
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] === {phase} (Return code: {result.returncode}) ===\n")
        if result.stdout.strip():
            f.write("--- stdout ---\n")
            for line in result.stdout.splitlines():
                f.write(f"[{timestamp}] [stdout] {line}\n")
        if result.stderr.strip():
            f.write("--- stderr ---\n")
            for line in result.stderr.splitlines():
                f.write(f"[{timestamp}] [stderr] {line}\n")


def run_ilp_training_subprocess(exs_file, bk_file, bias_file, settings_parameters, log_dir=None):
    """Run Popper ILP learning in a separate subprocess for isolated Prolog session."""
    #print(f"Running ILP training subprocess with exs_file={exs_file}, bk_file={bk_file}, bias_file={bias_file}...")
    #print(f"Settings parameters: {settings_parameters}")
    script = f'''
import json
import pickle
import base64
from popper.loop import learn_solution
from popper.util import Settings, format_prog

settings = Settings(ex_file=r"{exs_file}", bk_file=r"{bk_file}", bias_file=r"{bias_file}", **{repr(settings_parameters)})
prog, score, stats = learn_solution(settings)
prog_str = format_prog(prog) if prog else None

result = {{"prog_str": prog_str, "score": list(score) if score else None}}
print(json.dumps(result))
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        start_new_session=True,  # Start in a new session to isolate from parent process
        cwd=os.getcwd(),
    )
    if log_dir:
        log_subprocess_output(log_dir, f"Training: {bias_file}", result)
    # Parse only the last line (JSON output), ignore earlier lines (warnings/progress)
    stdout_lines = result.stdout.strip().split('\n')
    try:
        output = json.loads(stdout_lines[-1])
    except json.decoder.JSONDecodeError:
        output = {"prog_str": None, "score": None}
        print(f"    Failed to parse JSON output. See logs for details.")
    
    return output


def build_bias(problem_dir, predicate_set, target_ids, selection_mode:Literal["claude", "random", "top_k"]|None=None, selection_k:int|None=None, max_vars=6, max_body=8, max_clauses=2):
    # use bias template generated in build_bk and create settings-specific bias files
    for target_id in tqdm.tqdm(target_ids, desc="Building bias files for ChEBI classes"):
        plain_bias_path = get_bias_path(target_id, split="train", base_dir=problem_dir, predicate_set=predicate_set, selection_mode=selection_mode, selection_k=selection_k) # template bias file created in build_bk
        if selection_mode is None:
            assert os.path.exists(plain_bias_path), f"Bias template file {plain_bias_path} does not exist. Please run build_bk first to create the bias template before running build_bias."
        else:
            assert os.path.exists(plain_bias_path), f"Bias template file {plain_bias_path} does not exist. Please run predicate selection with selection_mode={selection_mode} and top_k={selection_k} first."

        bias_path = get_bias_path(target_id, split="train", base_dir=problem_dir, predicate_set=predicate_set, selection_mode=selection_mode, selection_k=selection_k, max_vars=max_vars, max_body=max_body, max_clauses=max_clauses)
        # use bias.pl to generate settings-specific bias file
        with open(plain_bias_path, "r") as f:
            bias_content = f.read()
        bias_content = bias_content.replace("%% max_vars(TODO).", f"max_vars({max_vars}).")
        bias_content = bias_content.replace("%% max_body(TODO).", f"max_body({max_body}).")
        bias_content = bias_content.replace("%% max_clauses(TODO).", f"max_clauses({max_clauses}).") 
                
        with open(bias_path, "w+") as f:
            f.write(bias_content)


def learn_chebi_classes(classes_list, problem_dir:Optional[str], predicate_set:Literal[AVAILABLE_PREDICATE_SETS], results_dir, timeout=20, selection_mode:Literal["claude", "random", "top_k"]|None=None, selection_k:int|None=None, max_vars=6, max_body=8, max_clauses=2, mdl_weight_fn=1, mdl_weight_fp=1, mdl_weight_size=1):
    if problem_dir is None:
        problem_dir = os.path.join("data", "ilp_problems")
    # Build settings parameters for Popper
    settings_parameters = {
        "noisy": True,
        "anytime_solver": "nuwls",
        "timeout": timeout,
        "mdl_weight_fn": mdl_weight_fn,
        "mdl_weight_fp": mdl_weight_fp,
        "mdl_weight_size": mdl_weight_size,
        "max_vars": max_vars,
        "max_body": max_body,
        "max_clauses": max_clauses
    }

    with open(os.path.join(results_dir, "config.yml"), "a+") as f:
        f.write(f"problem_dir: {problem_dir}\n")
        f.write("popper_settings:\n")
        for key, value in settings_parameters.items():
            f.write(f"\t{key}: {value}\n")

    build_bias(problem_dir, predicate_set, classes_list, selection_mode=selection_mode, selection_k=selection_k, max_vars=max_vars, max_body=max_body, max_clauses=max_clauses)

    for chebi_id in classes_list:
        start_time = time.perf_counter()
        # Run training in subprocess (isolated Prolog session)
        print(f"Training ChEBI:{chebi_id}")
        exs_path = get_exs_path(chebi_id, split="train", base_dir=problem_dir)
        bk_path = get_bk_path(chebi_id, split="train", base_dir=problem_dir, predicate_set=predicate_set, selection_mode=selection_mode, selection_k=selection_k)
        bias_path = get_bias_path(chebi_id, split="train", base_dir=problem_dir, predicate_set=predicate_set, selection_mode=selection_mode, selection_k=selection_k, max_vars=max_vars, max_body=max_body, max_clauses=max_clauses)
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
                    exs_file=get_exs_path(chebi_id, split="validation", base_dir=problem_dir),
                    bk_file=get_bk_path(chebi_id, predicate_set=predicate_set, split="validation", base_dir=problem_dir),
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

