import copy
import json
import os
import time
from chebILP.ilp_path_manager import (
    class_head,
    class_id_from_head,
    get_bk_path,
    get_exs_path,
)
from typing import Literal, Optional

import numpy as np
import pandas as pd


def _silent_clingo_logger(code, message):
    """Swallow Clingo's grounding messages."""
    pass


# Per-process configuration for the example workers. Populated once (in the main
# process for the serial path, or by each pool worker's initializer) so that the
# large, read-only bits (all programs, rule programs) are not re-shipped for every example.
_WORKER_STATE: dict = {}


def _collect_rule_programs(programs, rule_library_dir):
    """Rule programs of the LLM rule library needed to evaluate the learned programs.

    Returns ``(rule_programs, dependency_programs)``: the programs the classes chose, and
    the library programs those build on. ``class_map.json`` records only the former, so
    without the latter every layered rule grounds against an empty body and derives nothing.
    Kept apart because only the chosen programs' extensions belong in the background —
    the dependencies exist to make them derivable, exactly as in ``ILPProblemBuilder.build_bk``.

    Names are library-wide unique (``add_rule_to_library`` disambiguates colliding content
    with the class id), so the programs go into the shared background as they are. Rule
    programs carry only strings, so they are picklable and shipped to workers directly.

    A class's *whole* rule set is loaded, not just the heads its learned program uses: a head
    the program never mentions may still be the helper another head depends on.
    """
    import re
    from chebILP.predicate_generation.auxiliary_rules import (
        load_class_rules,
        resolve_rule_dependencies,
    )
    from chebILP.preprocessing.chebi.fg_matching import FG_SEED_PREFIXES

    library_name = re.compile(
        r"\b(?:%s)\w+" % "|".join(map(re.escape, ("aux_",) + FG_SEED_PREFIXES))
    )
    chosen = {}
    for cls_id, prog in programs.items():
        if not library_name.search(prog):
            continue
        for rp in load_class_rules(cls_id, library_dir=rule_library_dir):
            chosen.setdefault(rp.name, rp)

    rule_programs = list(chosen.values())
    dependencies = [
        dep
        for dep in resolve_rule_dependencies(rule_programs, rule_library_dir)
        if dep.name not in chosen
    ]
    return rule_programs, dependencies


def _without_data(dataset):
    """A copy of ``dataset`` without its example frame and hierarchy, cheap to ship to workers.

    Encoding an example (``build_background`` and friends) needs only the rows passed in.
    """
    light = copy.copy(dataset)
    for attr in (
        "molecules",
        "graph",
        "_closure",
        "_example_id_cache",
        "_sizes",
    ):
        light.__dict__.pop(attr, None)
    return light


def _worker_init(state):
    """Pool initializer: stash the shared config."""
    global _WORKER_STATE
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _WORKER_STATE = dict(state)


def _evaluate_example(payload):
    """Evaluate every program against a single example.

    ``payload`` is ``(example_id, row)``, one row of the dataset's example frame. Returns
    ``(example_id, fired_names, status)`` with ``status`` one of ``""`` (ok), ``"timeout"``
    (partial result still returned) or ``"error: ..."``.
    """
    import clingo
    from chebILP.preprocessing.base_problem_builder import background_with_rules

    example_id, row = payload
    st = _WORKER_STATE
    row_df = pd.DataFrame([row], index=[example_id])

    try:
        bg_facts = background_with_rules(
            st["dataset"],
            row_df,
            st.get("rule_programs"),
            st.get("rule_dependencies"),
            computed_facts=st.get("computed_facts", False),
        )
    except Exception as e:  # noqa: BLE001
        return example_id, [], f"error: {e}"

    col_of = st["col_of"]
    fired: set[str] = set()

    def _on_model(model):
        for atom in model.symbols(atoms=True):
            if atom.name in col_of:
                fired.add(atom.name)

    label_timeout = st["label_timeout"]
    try:
        ctl = clingo.Control(logger=_silent_clingo_logger)
        ctl.add("base", [], "\n".join(bg_facts) + "\n" + st["programs_str"])
        ctl.ground([("base", [])])
        if label_timeout and label_timeout > 0:
            with ctl.solve(on_model=_on_model, async_=True) as handle:
                if not handle.wait(label_timeout):
                    handle.cancel()
                    return example_id, sorted(fired), "timeout"
        else:
            ctl.solve(on_model=_on_model)
    except Exception as e:  # noqa: BLE001
        return example_id, [], f"error: {e}"
    return example_id, sorted(fired), ""


def build_ilp_preds_tensor(
    programs: dict[str, str],
    dataset,
    mol_order: list[str],
    output_npy_path: str,
    output_meta_path: str,
    aux_library_dir: str | None = None,
    label_timeout: float = 60.0,
    n_jobs: int = 1,
    computed_facts: bool = False,
) -> pd.DataFrame:
    """
    Evaluate ILP programs against every example in mol_order and save the
    result as a boolean numpy tensor.

    Builds background knowledge from scratch (not from existing bk.pl problem
    files), exactly as ILPProblemBuilder.build_bk does, so the programs are
    evaluated against the BK they were learned on. Intended for generating full
    split prediction tensors used by ensemble_construct.

    Args:
        programs:         class_id → Prolog program string (only classes with
                          a learned program; null programs must be excluded).
        dataset:          the dataset the programs were learned on; its example
                          frame supplies the rows and its encoding the BK.
        mol_order:        ordered list of example IDs to evaluate.
        output_npy_path:  destination .npy file for the boolean tensor.
        output_meta_path: destination JSON file for mol_order / class_labels.
        aux_library_dir:  the LLM rule library the run was learned with, or None.
        label_timeout:    Clingo solving timeout in seconds (default 60).
        n_jobs:           number of worker processes to evaluate examples in
                          parallel. 1 (default) runs serially in-process; <= 0
                          uses all available CPU cores. Examples are independent,
                          so this scales close to linearly with cores.

    Returns a boolean DataFrame (index=mol_order, columns=sorted class IDs).
    """
    import tqdm
    import clingo
    from chebILP.predicate_generation.auxiliary_rules import referenced_fact_predicates
    from chebILP.preprocessing.chebi.fg_matching import (
        FG_SEED_PREFIXES,
        is_fg_seed_name,
    )

    ilp_class_ids = sorted(programs.keys())
    n_mols = len(mol_order)
    n_classes = len(ilp_class_ids)
    print(
        f"Building ILP predictions tensor: {n_mols} examples × {n_classes} classes "
        f"(rule library: {aux_library_dir})"
    )

    # Examples we actually have; anything missing stays False in the tensor.
    examples = dataset.molecules
    present_rows = examples[examples.index.isin(set(mol_order))].dropna(subset=["mol"])
    present_ids = [str(i) for i in present_rows.index]
    missing = len(set(mol_order)) - len(set(present_ids))
    if missing:
        print(f"  {missing} examples missing from the dataset — predicted False")

    # Column index per target label, and row index per example id.
    col_of = {class_head(cls_id): j for j, cls_id in enumerate(ilp_class_ids)}
    mol_idx = {mol_id: i for i, mol_id in enumerate(mol_order)}
    tensor = np.zeros((n_mols, n_classes), dtype=bool)

    # Parse-check every program once and keep the valid ones (class_id -> program). Each
    # program's only derived head is its unique class_head target, so they never interfere.
    valid_programs: dict[str, str] = {}
    for cls_id in ilp_class_ids:
        prog = "\n".join(line for line in programs[cls_id].split("\n") if line.strip())
        try:
            clingo.Control(logger=_silent_clingo_logger).add(
                "t", [], prog
            )  # syntax check only
            valid_programs[cls_id] = prog
        except RuntimeError as e:
            print(
                f"  Skipping {dataset.class_label(cls_id)} — failed to parse program: {e}"
            )

    # Gather the classes' rule programs (plus the library programs they build on) and
    # recompute their extensions in the background. RuleProgram objects are picklable,
    # so they ride in worker_state directly.
    rule_programs = rule_dependencies = None
    if aux_library_dir:
        rule_programs, rule_dependencies = _collect_rule_programs(
            valid_programs, aux_library_dir
        )
        print(
            f"{len(rule_programs)} distinct auxiliary rule(s) referenced by programs"
            + (
                f" (+{len(rule_dependencies)} dependencies)"
                if rule_dependencies
                else ""
            )
        )
        dataset.load_seed_matches(
            {p.name for p in rule_programs if is_fg_seed_name(p.name)}
            | referenced_fact_predicates(
                rule_programs + rule_dependencies, FG_SEED_PREFIXES
            )
        )

    # Shared, read-only config handed to each example evaluation (main process for the
    # serial path, or every pool worker via the initializer).
    worker_state = dict(
        dataset=_without_data(dataset),
        programs_str="\n".join(valid_programs.values()),
        col_of=col_of,
        label_timeout=label_timeout,
        rule_programs=rule_programs,
        rule_dependencies=rule_dependencies,
        computed_facts=computed_facts,
    )
    payloads = [(str(i), row) for i, row in present_rows.iterrows()]

    n_timeout = 0
    n_error = 0

    def _apply(result):
        nonlocal n_timeout, n_error
        mol_id, fired, status = result
        if status == "timeout":
            n_timeout += 1
        elif status.startswith("error"):
            n_error += 1
            print(
                f"  Clingo evaluation failed for {mol_id}: {status[len('error: ') :]}"
            )
            return
        i = mol_idx.get(mol_id)
        if i is not None:
            for name in fired:
                tensor[i, col_of[name]] = True

    # Examples are independent, so each is evaluated against its own tiny Clingo
    # grounding (one example's background knowledge + all programs). With n_jobs != 1
    # these are farmed out to worker processes for near-linear speedup.
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1

    if n_jobs == 1:
        _worker_init(worker_state)
        for payload in tqdm.tqdm(payloads, desc="Examples"):
            _apply(_evaluate_example(payload))
    else:
        from concurrent.futures import ProcessPoolExecutor

        print(f"Evaluating {len(payloads)} examples across {n_jobs} worker process(es)")
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_worker_init,
            initargs=(worker_state,),
        ) as ex:
            for result in tqdm.tqdm(
                ex.map(_evaluate_example, payloads, chunksize=8),
                total=len(payloads),
                desc="Examples",
            ):
                _apply(result)

    if n_timeout:
        print(
            f"  {n_timeout} example(s) hit the {label_timeout}s solving timeout — those rows may be incomplete"
        )
    if n_error:
        print(f"  {n_error} example(s) failed to evaluate — predicted False")

    np.save(output_npy_path, tensor)
    with open(output_meta_path, "w") as f:
        json.dump({"mol_order": mol_order, "class_labels": ilp_class_ids}, f)
    print(f"Saved ILP predictions tensor: {output_npy_path} ({n_mols} × {n_classes})")

    return pd.DataFrame(tensor, index=mol_order, columns=ilp_class_ids)


def predict_smiles(
    smiles_list: list[str],
    rules_file: str,
    target_predicates: list[str],
    dataset,
    verbose: bool = False,
    aux_library_dir: str | None = None,
    computed_facts: bool = False,
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
        dataset:           A molecule dataset (ChEBI) whose encoding builds the BK.
        verbose:           If True, print the satisfied predicates for each molecule.
        aux_library_dir:   The LLM rule library the rules were learned with, or None.

    Returns:
        list of dicts, one per input SMILES, each with keys:
          - ``"SMILES"``: the input SMILES string.
          - ``"labels"``: list of target predicates the molecule satisfies (empty if none
            or if the SMILES is invalid).
          - ``"time"``: time (seconds) the Clingo evaluation took, or None for invalid SMILES.
    """
    from rdkit import Chem
    from chebILP.preprocessing.base_problem_builder import background_with_rules
    from chebILP.evaluation.clingo_eval import evaluate_with_clingo

    with open(rules_file, "r") as f:
        rules = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("%")
        ]

    # Gather the rule programs the target classes use (and the library programs they
    # build on) so the background recomputes their extensions.
    rule_programs = rule_dependencies = None
    if aux_library_dir:
        programs_by_class = {
            class_id_from_head(t): "\n".join(r for r in rules if t in r)
            for t in target_predicates
            if class_id_from_head(t) is not None
        }
        rule_programs, rule_dependencies = _collect_rule_programs(
            programs_by_class, aux_library_dir
        )

    results: list[dict] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES (skipped): {smiles!r}")
            results.append({"SMILES": smiles, "labels": [], "time": None})
            continue

        mol_id = "m0"
        mol_df = pd.DataFrame([{"mol": mol}], index=[mol_id])
        background_facts = background_with_rules(
            dataset,
            mol_df,
            rule_programs,
            rule_dependencies,
            computed_facts=computed_facts,
        )
        print(
            f"Evaluating {smiles!r} against {len(rules)} rules and {len(background_facts)} background facts..."
        )
        print(f"  Target predicates: {', '.join(target_predicates)}")
        print(f"  Background knowledge: {background_facts}")

        start_time = time.perf_counter()
        try:
            positives = evaluate_with_clingo(
                rules, background_facts, target_predicates, [mol_id]
            )
        except Exception as e:
            print(f"Clingo evaluation failed for {smiles!r}: {e}")
            positives = {}
        elapsed = time.perf_counter() - start_time

        labels = [
            target
            for target in target_predicates
            if mol_id in positives.get(target, [])
        ]
        results.append({"SMILES": smiles, "labels": labels, "time": elapsed})

    if verbose:
        for entry in results:
            labels = entry["labels"]
            time_str = f"{entry['time']:.3f}s" if entry["time"] is not None else "n/a"
            print(
                f"{entry['SMILES']} [{time_str}]: {', '.join(labels) if labels else '(none)'}"
            )

    return results


def test_chebi_classes(
    run_to_evaluate,
    problem_dir: str,
    results_dir,
    test_on: Literal["validation", "test"] = "test",
    verbose: bool = False,
    predicate_dir: Optional[str] = None,
    **kwargs,
):
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
        from chebILP.evaluation.clingo_eval import run_ilp_validation_clingo

        try:
            conf_matrix, details = run_ilp_validation_clingo(
                class_head(chebi_id),
                prog_str,
                exs_file=get_exs_path(chebi_id, split=test_on, base_dir=problem_dir),
                bk_file=get_bk_path(
                    chebi_id,
                    split=test_on,
                    base_dir=problem_dir,
                    aux_library_dir=predicate_dir,
                ),
                return_details=True,
            )
            if verbose:
                balanced_acc = 0.5 * (
                    conf_matrix["TP"] / (conf_matrix["TP"] + conf_matrix["FN"])
                    if (conf_matrix["TP"] + conf_matrix["FN"]) > 0
                    else 0
                ) + 0.5 * (
                    conf_matrix["TN"] / (conf_matrix["TN"] + conf_matrix["FP"])
                    if (conf_matrix["TN"] + conf_matrix["FP"]) > 0
                    else 0
                )
                f1_score = (
                    conf_matrix["TP"]
                    / (
                        conf_matrix["TP"]
                        + 0.5 * (conf_matrix["FP"] + conf_matrix["FN"])
                    )
                    if (conf_matrix["TP"] + conf_matrix["FP"] + conf_matrix["FN"]) > 0
                    else 0.0
                )
                print(
                    f"  F1: {f1_score:.2f}, BA: {balanced_acc:.2f} TPs: {conf_matrix['TP']}, FPs: {conf_matrix['FP']}, TNs: {conf_matrix['TN']}, FNs: {conf_matrix['FN']}"
                )
                for outcome in ("FP", "FN"):
                    subset = [d for d in details if d["outcome"] == outcome][:10]
                    print(
                        f"  {'False Positive' if outcome == 'FP' else 'False Negative'} samples (showing up to 10):"
                    )
                    for d in subset:
                        print(f"    {d['id']} (true label: {d['true_label']})")

            with open(os.path.join(results_dir, "results.json"), "a+") as f:
                result_entry = {
                    "chebi_id": chebi_id,
                    "time_taken": time.perf_counter() - start_time,
                    "program": prog_str,
                    f"{test_on}_score": conf_matrix,
                    f"{test_on}_details": details,
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
                    "error": str(e),
                }
                f.write(json.dumps(result_entry) + "\n")
