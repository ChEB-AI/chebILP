import os
import re
import itertools
import string
import subprocess
import sys
import json
from datetime import datetime
import time
import traceback
import tqdm
from typing import Literal, Optional

from chebILP.ilp_path_manager import get_bias_path, get_bk_path, get_exs_path
from chebILP.utils import AVAILABLE_PREDICATE_SETS

# The subsystem whose auxiliary predicates are ASP rules whose heads can be analysed for
# seeding and heuristic guidance. Both features only apply here.
RULE_PREDICATE_SET = "llm_generated_rules"
# Clingo heuristic level for prefer_body_pred hints. Any non-zero value below the size
# heuristic's ~1000 works; when every hint shares a level the value only sets relative
# priority, so a single constant is enough (see popper-sfluegel/search-guidance.md).
DEFAULT_HEURISTIC_LEVEL = 10

_BODY_PRED_RE = re.compile(r"body_pred\(\s*([a-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)")


def _parse_body_preds(bias_text: str) -> set[tuple[str, int]]:
    """The ``(name, arity)`` pairs declared as ``body_pred`` in a bias file's text."""
    return {(m.group(1), int(m.group(2))) for m in _BODY_PRED_RE.finditer(bias_text)}


def _var_stream():
    """Fresh variable names for seed atoms: B, C, ... then V1, V2, ... (A is the molecule)."""
    yield from string.ascii_uppercase[1:]
    for i in itertools.count(1):
        yield f"V{i}"


def build_seed_hypothesis(target_id, programs, body_preds):
    """A single rule ANDing a class's chosen auxiliary predicates, with its var/body counts.

    ``chebi_<target>(A)`` holds when every chosen predicate holds. A molecule-level predicate
    is applied to the molecule variable ``A`` directly (``aux_x(A)``); an atom-level one is
    tied to the molecule with a ``has_atom(A, Atom)`` link, exactly as the task specifies.
    A predicate with several atom arguments (e.g. a ring's atoms) binds its remaining atoms
    through the aux literal itself, so one ``has_atom`` per predicate suffices.

    A predicate whose extension was empty for every molecule is not written to ``bk.pl`` and so
    is not a declared ``body_pred``; it is skipped, since Popper would reject a rule naming it.

    Returns ``(seed_str, n_vars, n_body)``, or ``None`` when no chosen predicate survives. The
    counts let the caller raise ``max_vars`` / ``max_body`` to fit the seed rather than drop it —
    the conjunction can need more variables than the default bias allows (e.g. a ring predicate
    contributes one atom variable per ring atom).
    """
    from chebILP.predicate_generation.auxiliary_rules import predicate_arg_sorts

    mol_var = "A"
    fresh = _var_stream()
    literals: list[tuple[str, list[str]]] = []
    for prog in programs:
        sorts = predicate_arg_sorts(prog)
        if not sorts or (prog.name, len(sorts)) not in body_preds:
            continue
        args, atom_vars = [], []
        for sort in sorts:
            if sort == "mol":
                args.append(mol_var)
            else:
                v = next(fresh)
                args.append(v)
                atom_vars.append(v)
        if atom_vars:
            literals.append(("has_atom", [mol_var, atom_vars[0]]))
        literals.append((prog.name, args))

    if not literals:
        return None
    all_vars = {v for _, args in literals for v in args}
    body = ", ".join(f"{pred}({','.join(args)})" for pred, args in literals)
    return f"chebi_{target_id}(A) :- {body}.", len(all_vars), len(literals)


def build_pred_heuristic_lines(programs, body_preds, level=DEFAULT_HEURISTIC_LEVEL):
    """``prefer_body_pred`` directives steering generation toward the chosen predicates.

    One line per chosen predicate that is actually a declared ``body_pred`` — Popper
    error-exits on a ``prefer_body_pred`` naming an undeclared predicate, so a predicate with
    an empty extension is skipped.
    """
    names_in_bk = {pred for pred, _ in body_preds}
    lines, seen = [], set()
    for prog in programs:
        if prog.name in names_in_bk and prog.name not in seen:
            seen.add(prog.name)
            lines.append(f"prefer_body_pred({prog.name},{level}).")
    return lines

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
from popper.loop import popper
from popper.util import Settings, format_prog
from popper import stats as popper_stats

settings = Settings(ex_file=r"{exs_file}", bk_file=r"{bk_file}", bias_file=r"{bias_file}", **{repr(settings_parameters)})
prog, score = popper(settings)
prog_str = format_prog(prog) if prog else None

result = {{"prog_str": prog_str, "score": list(score) if score else None, "num_programs": popper_stats.stats.total_programs}}
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


def build_bias(problem_dir, predicate_set, target_ids, selection_mode:Literal["claude", "random", "top_k"]|None=None, selection_k:int|None=None, max_vars=6, max_body=8, max_clauses=2, heuristic_guidance=False, aux_library_dir=None, heuristic_level=DEFAULT_HEURISTIC_LEVEL):
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

        # Heuristic guidance: steer Popper's non-recursive generator toward the class's chosen
        # auxiliary predicates by appending prefer_body_pred/2 directives (only those actually
        # declared as body_pred in the template — Popper error-exits on an undeclared one).
        if heuristic_guidance and predicate_set == RULE_PREDICATE_SET:
            from chebILP.predicate_generation.auxiliary_rules import load_class_rules
            programs = load_class_rules(target_id, library_dir=aux_library_dir)
            hint_lines = build_pred_heuristic_lines(programs, _parse_body_preds(bias_content), level=heuristic_level)
            if hint_lines:
                bias_content = bias_content.rstrip("\n") + "\n\n%% heuristic guidance toward LLM-chosen predicates\n" + "\n".join(hint_lines) + "\n"

        with open(bias_path, "w+") as f:
            f.write(bias_content)


def learn_chebi_classes(classes_list, problem_dir:Optional[str], predicate_set:Literal[AVAILABLE_PREDICATE_SETS], results_dir, timeout=20, selection_mode:Literal["claude", "random", "top_k"]|None=None, selection_k:int|None=None, max_vars=6, max_body=8, max_clauses=2, mdl_weight_fn=1, mdl_weight_fp=1, mdl_weight_size=1, seed_hypothesis=False, heuristic_guidance=False, aux_library_dir=None, heuristic_level=DEFAULT_HEURISTIC_LEVEL):
    if problem_dir is None:
        problem_dir = os.path.join("data", "ilp_problems")
    # Build settings parameters for Popper
    settings_parameters = {
        "noisy": True,
        "nuwls": True,
        "timeout": timeout,
    }
    if mdl_weight_fn != 1 or mdl_weight_fp != 1 or mdl_weight_size != 1:
        settings_parameters["fn_weight"] = mdl_weight_fn
        settings_parameters["fp_weight"] = mdl_weight_fp
        settings_parameters["size_weight"] = mdl_weight_size

    with open(os.path.join(results_dir, "config.yml"), "a+") as f:
        f.write(f"problem_dir: {problem_dir}\n")
        f.write(f"predicate_set: {predicate_set}\n")
        f.write(f"selection_mode: {selection_mode}\n")
        if selection_mode is not None:
            f.write(f"selection_k: {selection_k}\n")
        f.write(f"max_vars: {max_vars}\n")
        f.write(f"max_body: {max_body}\n")
        f.write(f"max_clauses: {max_clauses}\n")
        f.write(f"seed_hypothesis: {seed_hypothesis}\n")
        f.write(f"heuristic_guidance: {heuristic_guidance}\n")
        if seed_hypothesis or heuristic_guidance:
            f.write(f"aux_library_dir: {aux_library_dir}\n")
        f.write("popper_settings:\n")
        for key, value in settings_parameters.items():
            f.write(f"\t{key}: {value}\n")

    build_bias(problem_dir, predicate_set, classes_list, selection_mode=selection_mode, selection_k=selection_k, max_vars=max_vars, max_body=max_body, max_clauses=max_clauses, heuristic_guidance=heuristic_guidance, aux_library_dir=aux_library_dir, heuristic_level=heuristic_level)

    if seed_hypothesis and predicate_set != RULE_PREDICATE_SET:
        print(f"Warning: --seed_hypothesis only applies to predicate_set='{RULE_PREDICATE_SET}'; ignoring for '{predicate_set}'.")
    if heuristic_guidance and predicate_set != RULE_PREDICATE_SET:
        print(f"Warning: --heuristic_guidance only applies to predicate_set='{RULE_PREDICATE_SET}'; ignoring for '{predicate_set}'.")

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

        # Seed the search with the conjunction of the class's chosen auxiliary predicates.
        # Per-class, so it goes into a copy of the shared settings rather than the shared dict.
        class_settings = settings_parameters
        if seed_hypothesis and predicate_set == RULE_PREDICATE_SET:
            from chebILP.predicate_generation.auxiliary_rules import load_class_rules
            with open(bias_path, "r") as f:
                body_preds = _parse_body_preds(f.read())
            programs = load_class_rules(chebi_id, library_dir=aux_library_dir)
            seed_info = build_seed_hypothesis(chebi_id, programs, body_preds)
            if seed_info:
                seed, n_vars, n_body = seed_info
                # The seed must satisfy the bias it is validated against, so raise max_vars /
                # max_body to fit it and regenerate this class's bias at the wider bounds. The
                # search then runs at those bounds too — the seed is only reachable there.
                eff_vars, eff_body = max(max_vars, n_vars), max(max_body, n_body)
                if eff_vars != max_vars or eff_body != max_body:
                    print(f"    Seed needs max_vars={n_vars}, max_body={n_body}; "
                          f"raising bounds to max_vars={eff_vars}, max_body={eff_body} for this class.")
                    build_bias(problem_dir, predicate_set, [chebi_id], selection_mode=selection_mode,
                               selection_k=selection_k, max_vars=eff_vars, max_body=eff_body,
                               max_clauses=max_clauses, heuristic_guidance=heuristic_guidance,
                               aux_library_dir=aux_library_dir, heuristic_level=heuristic_level)
                    bias_path = get_bias_path(chebi_id, split="train", base_dir=problem_dir,
                                              predicate_set=predicate_set, selection_mode=selection_mode,
                                              selection_k=selection_k, max_vars=eff_vars,
                                              max_body=eff_body, max_clauses=max_clauses)
                print(f"    Seeding search with: {seed}")
                class_settings = dict(settings_parameters, best_hypothesis=seed)
            else:
                print(f"    No seed hypothesis for ChEBI:{chebi_id} "
                      f"(no usable predicates in bk.pl); running unseeded.")

        train_result = run_ilp_training_subprocess(exs_path, bk_path, bias_path, class_settings, log_dir=results_dir)
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
                from chebILP.evaluation.clingo_eval import run_ilp_validation_clingo
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
                "num_programs": train_result.get("num_programs"),
                "program": prog_str,
                "validation_score": conf_matrix,
            }
            f.write(json.dumps(result_entry) + "\n")

