import os
import re
import shutil
import subprocess
import sys
import json
from datetime import datetime
import time
import traceback
import tqdm
from typing import Literal, Optional

from chebILP.ilp_path_manager import (
    class_head,
    get_bias_path,
    get_bk_path,
    get_exs_path,
    get_aleph_stem,
)
from chebILP.prolog_bk import check_loadable, prolog_bk_files
from popper.util import MAX_BODY, MAX_VARS

# Clingo heuristic level for prefer_body_pred hints. Any non-zero value below the size
# heuristic's ~1000 works; when every hint shares a level the value only sets relative
# priority, so a single constant is enough (see popper-sfluegel/search-guidance.md).
DEFAULT_HEURISTIC_LEVEL = 10
# Popper skips the exact combine solve after the search (popper-sfluegel's final_combine_solve), so a
# run ends shortly after its timeout. One still going POPPER_KILL_GRACE s after it is killed and
# counts as no solution.
POPPER_KILL_GRACE = 60

_BODY_PRED_RE = re.compile(r"body_pred\(\s*([a-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)")
_HEAD_PRED_RE = re.compile(r"head_pred\(\s*[a-z_][A-Za-z0-9_]*\s*,\s*(\d+)\s*\)")


def _parse_body_preds(bias_text: str) -> set[tuple[str, int]]:
    """The ``(name, arity)`` pairs declared as ``body_pred`` in a bias file's text."""
    return {(m.group(1), int(m.group(2))) for m in _BODY_PRED_RE.finditer(bias_text)}


def _parse_head_arity(bias_text: str) -> int:
    """Arity of the ``head_pred`` in a bias file's text (1 if none is declared)."""
    m = _HEAD_PRED_RE.search(bias_text)
    return int(m.group(1)) if m else 1


def seed_from_hypothesis(clause, target_id, body_preds, head_arity=1):
    """Validate an LLM class hypothesis for use as Aleph's seed clause.

    Returns ``(seed_str, n_vars, n_body)`` when the clause is a plain Datalog rule whose head is
    the class head with ``head_arity`` arguments and whose every body predicate is declared as a ``body_pred`` in the
    bias, else ``None``. An aggregate/negation/comparison (``datalog_ok`` False) or a predicate
    that never reached ``bk.pl`` (so is not a declared ``body_pred``) makes it unusable. The
    caller still checks it against ``max_vars``/``max_body``.
    """
    from chebILP.predicate_generation.auxiliary_rules import analyze_hypothesis

    info = analyze_hypothesis(clause)
    if info is None or not info["datalog_ok"] or not info["body_preds"]:
        return None
    if info["head"] != (class_head(target_id), head_arity):
        return None
    if not info["body_preds"] <= body_preds:
        return None
    return info["clause"], info["n_vars"], info["n_body"]


def build_hypothesis_heuristic_lines(clause, body_preds, level=DEFAULT_HEURISTIC_LEVEL):
    """``prefer_body_pred`` directives for the predicates appearing in an LLM hypothesis.

    Only names that are actually declared as ``body_pred`` in the bias are kept (Popper
    error-exits on an undeclared one), in the order they appear in the hypothesis body.
    """
    from chebILP.predicate_generation.auxiliary_rules import analyze_hypothesis

    info = analyze_hypothesis(clause)
    if info is None:
        return []
    names_in_bk = {pred for pred, _ in body_preds}
    return [
        f"prefer_body_pred({name},{level})."
        for name in info["body_pred_names"]
        if name in names_in_bk
    ]


def build_body_cost_lines(body_preds, aux_cost=1, seed_cost=1, base_cost=1):
    """``body_cost`` directives pricing each body predicate in Popper's MDL size term.

    Three tiers: LLM-generated ``aux_*`` predicates, seeded functional groups (``efg_``,
    ``efga_``, ``chembl_fg_``) and everything else (atom-level primitives). Popper's default
    cost is 1, so only predicates with a different cost get a line.
    """
    from chebILP.preprocessing.chebi.fg_matching import is_fg_seed_name

    lines = []
    for name in sorted({pred for pred, _ in body_preds}):
        if name.startswith("aux_"):
            cost = aux_cost
        elif is_fg_seed_name(name):
            cost = seed_cost
        else:
            cost = base_cost
        if cost != 1:
            lines.append(f"body_cost({name},{cost}).")
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


def run_ilp_training_subprocess(
    exs_file, bk_file, bias_file, settings_parameters, log_dir=None
):
    """Run Popper ILP learning in a separate subprocess for isolated Prolog session."""
    # print(f"Running ILP training subprocess with exs_file={exs_file}, bk_file={bk_file}, bias_file={bias_file}...")
    # print(f"Settings parameters: {settings_parameters}")
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
    hard_limit = settings_parameters.get("timeout", 0) + POPPER_KILL_GRACE
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            start_new_session=True,  # Start in a new session to isolate from parent process
            cwd=os.getcwd(),
            timeout=hard_limit,
        )
    except subprocess.TimeoutExpired as e:
        stderr = (
            e.stderr.decode(errors="replace")
            if isinstance(e.stderr, bytes)
            else (e.stderr or "")
        )
        log_subprocess_output(
            log_dir, f"Training: {bias_file} (killed after {hard_limit} s)", stderr
        )
        print(f"    Popper did not finish within {hard_limit} s; no solution.")
        return {
            "prog_str": None,
            "score": None,
            "timed_out": True,
            "time_to_best": None,
        }
    if log_dir:
        log_subprocess_output(log_dir, f"Training: {bias_file}", result)
    # Parse only the last line (JSON output), ignore earlier lines (warnings/progress)
    stdout_lines = result.stdout.strip().split("\n")
    try:
        output = json.loads(stdout_lines[-1])
    except json.decoder.JSONDecodeError:
        output = {"prog_str": None, "score": None}
        stderr_lines = result.stderr.strip().splitlines()
        reason = f": {stderr_lines[-1]}" if stderr_lines else ""
        print(f"    Popper produced no result{reason}. See run.log for details.")

    # Time-to-best hypothesis: Popper's anytime search prints "<seconds>s New best hypothesis:"
    # to stderr on each improvement; the last one's timestamp is the time to the final program.
    best_times = re.findall(r"([\d.]+)s New best hypothesis:", result.stderr or "")
    output["time_to_best"] = float(best_times[-1]) if best_times else None

    return output


def build_bias(
    problem_dir,
    target_ids,
    max_vars=MAX_VARS,
    max_body=MAX_BODY,
    heuristic_guidance=False,
    aux_library_dir=None,
    heuristic_level=DEFAULT_HEURISTIC_LEVEL,
    aux_pred_cost=1,
    seed_pred_cost=1,
    base_pred_cost=1,
):
    # use bias template generated in build_bk and create settings-specific bias files
    for target_id in tqdm.tqdm(target_ids, desc="Building bias files"):
        plain_bias_path = get_bias_path(
            target_id,
            split="train",
            base_dir=problem_dir,
            aux_library_dir=aux_library_dir,
        )  # template bias file created in build_bk
        assert os.path.exists(plain_bias_path), (
            f"Bias template file {plain_bias_path} does not exist. Please run build_bk first to create the bias template before running build_bias."
        )

        bias_path = get_bias_path(
            target_id,
            split="train",
            base_dir=problem_dir,
            aux_library_dir=aux_library_dir,
            max_vars=max_vars,
            max_body=max_body,
        )
        # use bias.pl to generate settings-specific bias file
        with open(plain_bias_path, "r") as f:
            bias_content = f.read()
        bias_content = bias_content.replace(
            "%% max_vars(TODO).", f"max_vars({max_vars})."
        )
        bias_content = bias_content.replace(
            "%% max_body(TODO).", f"max_body({max_body})."
        )

        # Heuristic guidance: steer Popper's non-recursive generator toward the class's chosen
        # auxiliary predicates by appending prefer_body_pred/2 directives (only those actually
        # declared as body_pred in the template — Popper error-exits on an undeclared one).
        if heuristic_guidance and aux_library_dir:
            from chebILP.predicate_generation.auxiliary_rules import (
                load_class_hypothesis,
            )

            # Steer toward the predicates in the LLM's class hypothesis.
            hypothesis = load_class_hypothesis(target_id, library_dir=aux_library_dir)
            hint_lines = []
            if hypothesis and hypothesis.get("hypothesis"):
                hint_lines = build_hypothesis_heuristic_lines(
                    hypothesis["hypothesis"],
                    _parse_body_preds(bias_content),
                    level=heuristic_level,
                )
            if hint_lines:
                bias_content = (
                    bias_content.rstrip("\n")
                    + "\n\n%% heuristic guidance toward LLM-chosen predicates\n"
                    + "\n".join(hint_lines)
                    + "\n"
                )

        cost_lines = build_body_cost_lines(
            _parse_body_preds(bias_content),
            aux_cost=aux_pred_cost,
            seed_cost=seed_pred_cost,
            base_cost=base_pred_cost,
        )
        if cost_lines:
            bias_content = (
                bias_content.rstrip("\n")
                + "\n\n%% per-predicate costs in the MDL size term\n"
                + "\n".join(cost_lines)
                + "\n"
            )

        with open(bias_path, "w+") as f:
            f.write(bias_content)


def learn_chebi_classes(
    classes_list,
    problem_dir: Optional[str],
    results_dir,
    timeout=20,
    max_vars=MAX_VARS,
    max_body=MAX_BODY,
    mdl_weight_fn=1,
    mdl_weight_fp=1,
    mdl_weight_size=1,
    seed_hypothesis=False,
    heuristic_guidance=False,
    aux_library_dir=None,
    heuristic_level=DEFAULT_HEURISTIC_LEVEL,
    tool: Literal["popper", "aleph"] = "popper",
    nuwls=True,
    aux_pred_cost=1,
    seed_pred_cost=1,
    base_pred_cost=1,
):
    if problem_dir is None:
        problem_dir = os.path.join("data", "ilp_problems")
    # Build settings parameters for Popper
    settings_parameters = {
        "noisy": True,
        "nuwls": nuwls,
        "timeout": timeout,
        "final_combine_solve": False,
    }
    if mdl_weight_fn != 1 or mdl_weight_fp != 1 or mdl_weight_size != 1:
        settings_parameters["fn_weight"] = mdl_weight_fn
        settings_parameters["fp_weight"] = mdl_weight_fp
        settings_parameters["size_weight"] = mdl_weight_size

    with open(os.path.join(results_dir, "config.yml"), "a+") as f:
        f.write(f"tool: {tool}\n")
        f.write(f"problem_dir: {problem_dir}\n")
        f.write(f"max_vars: {max_vars}\n")
        f.write(f"max_body: {max_body}\n")
        f.write(f"seed_hypothesis: {seed_hypothesis}\n")
        f.write(f"heuristic_guidance: {heuristic_guidance}\n")
        f.write(f"predicate_dir: {aux_library_dir}\n")
        f.write(
            f"pred_costs: {{aux: {aux_pred_cost}, seed: {seed_pred_cost}, base: {base_pred_cost}}}\n"
        )
        f.write("popper_settings:\n")
        for key, value in settings_parameters.items():
            f.write(f"\t{key}: {value}\n")

    if tool == "aleph" and (aux_pred_cost, seed_pred_cost, base_pred_cost) != (1, 1, 1):
        print(
            "Warning: predicate costs only apply to Popper; ignoring them for --tool aleph."
        )
        aux_pred_cost = seed_pred_cost = base_pred_cost = 1
    build_bias(
        problem_dir,
        classes_list,
        max_vars=max_vars,
        max_body=max_body,
        heuristic_guidance=heuristic_guidance,
        aux_library_dir=aux_library_dir,
        heuristic_level=heuristic_level,
        aux_pred_cost=aux_pred_cost,
        seed_pred_cost=seed_pred_cost,
        base_pred_cost=base_pred_cost,
    )

    if seed_hypothesis and tool != "aleph":
        print(
            "Warning: --seed_hypothesis only works with --tool aleph; ignoring it for Popper."
        )
        seed_hypothesis = False
    if seed_hypothesis and not aux_library_dir:
        print(
            "Warning: --seed_hypothesis needs an LLM rule library (--predicate_dir); ignoring it."
        )
    if tool == "popper" and nuwls and shutil.which("NuWLS-c") is None:
        print(
            "Warning: --nuwls is on but no NuWLS-c binary is on PATH, so Popper's anytime MaxSAT solve does nothing. "
            "Install it as described in the Popper README (https://github.com/logic-and-learning-lab/Popper), "
            "or pass --no-nuwls."
        )
    if heuristic_guidance and not aux_library_dir:
        print(
            "Warning: --heuristic_guidance needs an LLM rule library (--predicate_dir); ignoring it."
        )

    # Popper's Settings swallows unknown kwargs, so an older Popper would run the unbounded final solve
    if tool == "popper":
        import inspect
        from popper.util import Settings

        if "final_combine_solve" not in inspect.signature(Settings.__init__).parameters:
            raise RuntimeError(
                "The installed Popper has no final_combine_solve setting; install popper-sfluegel (see README)."
            )

    # fail before the first run rather than hours into it
    first_files = None
    for chebi_id in classes_list:
        if os.path.exists(get_bk_path(chebi_id, "train", problem_dir, aux_library_dir)):
            files = prolog_bk_files(
                chebi_id, problem_dir, aux_library_dir, verify=False
            )
            first_files = first_files or files
    if first_files:
        check_loadable(first_files, tool)

    for chebi_id in classes_list:
        try:
            prolog_bk = prolog_bk_files(chebi_id, problem_dir, aux_library_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"ChEBI:{chebi_id}: {e} - skipping.")
            continue
        start_time = time.perf_counter()
        # Run training in subprocess (isolated Prolog session)
        print(f"Training ChEBI:{chebi_id}")
        exs_path = get_exs_path(chebi_id, split="train", base_dir=problem_dir)
        bk_path = get_bk_path(
            chebi_id,
            split="train",
            base_dir=problem_dir,
            aux_library_dir=aux_library_dir,
        )
        bias_path = get_bias_path(
            chebi_id,
            split="train",
            base_dir=problem_dir,
            aux_library_dir=aux_library_dir,
            max_vars=max_vars,
            max_body=max_body,
        )
        if (
            not os.path.exists(exs_path)
            or not os.path.exists(bk_path)
            or not os.path.exists(bias_path)
        ):
            print(
                f"Missing files for ChEBI:{chebi_id} - skipping. exs_path: {exs_path}, bk_path: {bk_path}, bias_path: {bias_path}"
            )
            continue

        with open(bias_path, "r") as f:
            bias_text = f.read()
        head_arity = _parse_head_arity(bias_text)

        # Seed Aleph's search with the LLM's class hypothesis.
        seed_clause = None
        if seed_hypothesis and aux_library_dir:
            from chebILP.predicate_generation.auxiliary_rules import (
                load_class_hypothesis,
            )

            body_preds = _parse_body_preds(bias_text)
            hypothesis = load_class_hypothesis(chebi_id, library_dir=aux_library_dir)
            seed_info = None
            if (
                hypothesis
                and hypothesis.get("groundable")
                and hypothesis.get("hypothesis")
            ):
                seed_info = seed_from_hypothesis(
                    hypothesis["hypothesis"], chebi_id, body_preds, head_arity
                )
            if seed_info:
                seed, n_vars, n_body = seed_info
                # The seed must satisfy the bias it is validated against. Widening max_vars /
                # max_body to fit an oversized seed also widens the search the baseline runs at,
                # making the comparison unfair (and the 100-class run showed it losing whole
                # classes for no F1 gain). So keep the standard bounds and drop a seed that does
                # not fit — running unseeded at the same bias as every other config.
                if n_vars > max_vars or n_body > max_body:
                    print(
                        f"    Seed needs max_vars={n_vars}, max_body={n_body} > bounds "
                        f"(max_vars={max_vars}, max_body={max_body}); running unseeded."
                    )
                else:
                    print(f"    Seeding search with: {seed}")
                    seed_clause = seed
            else:
                print(
                    f"    No usable LLM hypothesis for ChEBI:{chebi_id} "
                    f"(missing, did not ground, or does not fit the bias); running unseeded."
                )

        if tool == "aleph":
            from chebILP.aleph_runner import run_ilp_training_aleph

            aleph_stem = get_aleph_stem(
                chebi_id,
                base_dir=problem_dir,
                aux_library_dir=aux_library_dir,
            )
            missing = [
                aleph_stem + ext
                for ext in (".b", ".f", ".n")
                if not os.path.exists(aleph_stem + ext)
            ]
            if missing:
                print(
                    f"Missing Aleph file(s) for ChEBI:{chebi_id}: {', '.join(missing)}; "
                    f"run build_samples + build_bk (Aleph output on) first - skipping."
                )
                continue
            train_result = run_ilp_training_aleph(
                chebi_id,
                aleph_stem,
                bias_path,
                timeout=timeout,
                max_body=max_body,
                seed_clause=seed_clause,
                log_dir=results_dir,
                head_name=class_head(chebi_id),
                head_arity=head_arity,
                run_stem_name=str(chebi_id),
                prolog_bk_files=prolog_bk,
            )
        else:
            class_parameters = {**settings_parameters, "prolog_bk_files": prolog_bk}
            train_result = run_ilp_training_subprocess(
                exs_path, bk_path, bias_path, class_parameters, log_dir=results_dir
            )
        prog_str = train_result["prog_str"]  # string representation for display/storage
        score = train_result["score"]
        if score:
            tp, fn, tn, fp = score[0], score[1], score[2], score[3]
            f1 = (2 * tp / (2 * tp + fn + fp)) if (tp + fn + fp) > 0 else 0.0
            print(
                f"ChEBI:{chebi_id} - F1: {f1:.2f} (TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn})"
            )
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
                    class_head(chebi_id),
                    prog_str,
                    exs_file=get_exs_path(
                        chebi_id, split="validation", base_dir=problem_dir
                    ),
                    bk_file=get_bk_path(
                        chebi_id,
                        split="validation",
                        base_dir=problem_dir,
                        aux_library_dir=aux_library_dir,
                    ),
                )
                f1 = (
                    (
                        2
                        * conf_matrix["TP"]
                        / (
                            2 * conf_matrix["TP"]
                            + conf_matrix["FP"]
                            + conf_matrix["FN"]
                        )
                    )
                    if (conf_matrix["TP"] + conf_matrix["FP"] + conf_matrix["FN"]) > 0
                    else 0.0
                )
                print(
                    f"    Validation F1: {f1:.2f} (TP: {conf_matrix['TP']}, FP: {conf_matrix['FP']}, TN: {conf_matrix['TN']}, FN: {conf_matrix['FN']})"
                )
            except Exception as e:
                print(f"Validation failed for ChEBI:{chebi_id} with error: {e}")
                traceback.print_exc()
                conf_matrix = None

        with open(os.path.join(results_dir, "results.json"), "a+") as f:
            result_entry = {
                "chebi_id": chebi_id,
                "train_score": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
                if score
                else None,
                "time_taken": time.perf_counter() - start_time,
                "time_to_best": train_result.get("time_to_best"),
                "num_programs": train_result.get("num_programs"),
                "timed_out": train_result.get("timed_out", False),
                "program": prog_str,
                "validation_score": conf_matrix,
            }
            f.write(json.dumps(result_entry) + "\n")
