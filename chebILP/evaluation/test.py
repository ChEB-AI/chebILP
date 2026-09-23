

import json
import os
import time
from chebILP.ilp_path_manager import get_bk_path, get_exs_path
from typing import Literal, Optional

import numpy as np
import pandas as pd


def _silent_clingo_logger(code, message):
    """Swallow Clingo's grounding messages.
    """
    pass


# Per-process configuration for the molecule workers. Populated once (in the main
# process for the serial path, or by each pool worker's initializer) so that the
# large, read-only bits (all programs, auxiliary predicates) are not re-shipped or
# re-parsed for every molecule.
_WORKER_STATE: dict = {}


def _aux_predicate_specs(programs, aux_library_dir):
    """Which auxiliary predicates (llm_generated_fgs) the learned programs reference.

    An ``aux_`` name identifies one implementation library-wide: ``add_program_to_library``
    disambiguates a name that would back different code with the class id, so a shared
    background knowledge can hold every class's predicates side by side under their own
    names. Only the ones some program actually mentions are worth computing.

    Returns a list of ``(name, source_file)`` telling a worker which predicates to load.
    """
    import re
    from chebILP.predicate_generation.auxiliary_predicates import load_auxiliary_predicates

    specs: dict[str, str] = {}
    for cls_id, prog in programs.items():
        used = set(re.findall(r"\baux_\w+", prog))
        if not used:
            continue
        for p in load_auxiliary_predicates(cls_id, library_dir=aux_library_dir):
            if p.name in used:
                specs.setdefault(p.name, p.source_file)
    return list(specs.items())


def _collect_rule_programs(programs, rule_library_dir):
    """Rule programs (llm_generated_rules) needed to evaluate the learned programs.

    Returns ``(rule_programs, dependency_programs)``: the programs the classes chose, and
    the library programs those build on. ``class_map.json`` records only the former, so
    without the latter every layered rule grounds against an empty body and derives nothing.
    Kept apart because only the chosen programs' extensions belong in the background —
    the dependencies exist to make them derivable, exactly as in ``ILPProblemBuilder.build_bk``.

    Names are library-wide unique (``add_rule_to_library`` disambiguates colliding content
    with the class id), so the programs go into the shared background as they are. Rule
    programs carry only strings, so (unlike Python aux predicates) they are picklable and
    shipped to workers directly rather than reloaded from disk.

    A class's *whole* rule set is loaded, not just the heads its learned program uses: a head
    the program never mentions may still be the helper another head depends on.
    """
    import re
    from chebILP.predicate_generation.auxiliary_rules import load_class_rules, resolve_rule_dependencies

    chosen = {}
    for cls_id, prog in programs.items():
        if not re.search(r"\baux_\w+", prog):
            continue
        for rp in load_class_rules(cls_id, library_dir=rule_library_dir):
            chosen.setdefault(rp.name, rp)

    rule_programs = list(chosen.values())
    dependencies = [
        dep for dep in resolve_rule_dependencies(rule_programs, rule_library_dir)
        if dep.name not in chosen
    ]
    return rule_programs, dependencies


def _load_aux_predicates(specs):
    """Reload auxiliary predicates for a worker from ``(name, source_file)`` specs.

    The predicates' ``extension`` functions are ``exec``-compiled and therefore not
    picklable, so each worker rebuilds them from disk rather than receiving them.
    """
    from chebILP.predicate_generation.auxiliary_predicates import load_program_source

    preds = []
    for _, source_file in specs:
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                pred = load_program_source(f.read(), source_file=source_file)
        except OSError:
            pred = None
        if pred is not None:
            preds.append(pred)
    return preds


def _quiet_aux_logging():
    """Silence the noise the LLM-generated auxiliary programs produce.

    Their compile/timeout/crash paths log via chebILP.predicate_generation.auxiliary_predicates, and they
    routinely trip RDKit's own C++ warnings/errors (valence, parsing, ...). During a
    full-split tensor build these would flood the console, so raise both above the
    default level. Failures are still reported in aggregate at the end of the build.
    """
    import logging
    from rdkit import RDLogger

    logging.getLogger("chebILP.predicate_generation.auxiliary_predicates").setLevel(logging.ERROR)
    RDLogger.DisableLog("rdApp.*")


def _worker_init(state, aux_load_args):
    """Pool initializer: stash the shared config and (re)load auxiliary predicates.

    ``aux_load_args`` is the spec list from ``_aux_predicate_specs`` (``(name, source_file)``
    pairs) or ``None``. The predicates' ``extension`` functions are ``exec``-compiled and
    therefore not picklable, so each worker reloads them from disk rather than receiving them.
    """
    global _WORKER_STATE
    _quiet_aux_logging()
    _WORKER_STATE = dict(state)
    _WORKER_STATE["aux_predicates"] = (
        _load_aux_predicates(aux_load_args) if aux_load_args is not None else None
    )


def _evaluate_molecule(payload):
    """Evaluate every program against a single molecule.

    ``payload`` is ``(mol_id, mol_binary)`` where ``mol_binary`` is the RDKit
    ``Mol.ToBinary(AllProps)`` blob (kept picklable and property-complete across the
    process boundary). Returns ``(mol_id, fired_names, status, aux_failures)`` with
    ``status`` one of ``""`` (ok), ``"timeout"`` (partial result still returned) or
    ``"error: ..."``, and ``aux_failures`` a ``{name: "timeout"|"error"}`` map of the
    auxiliary predicates that could not be computed for this molecule.
    """
    import clingo
    import pandas as pd
    from rdkit import Chem
    from chebILP.ilp_problem_builder import build_full_background

    mol_id, mol_binary = payload
    st = _WORKER_STATE
    row_df = pd.DataFrame({"mol": [Chem.Mol(mol_binary)]}, index=[mol_id])

    aux_failures: dict[str, str] = {}
    try:
        bg_facts = build_full_background(
            row_df, predicate_set=st["predicate_set"],
            aux_predicates=st["aux_predicates"], aux_timeout=st["aux_timeout"],
            aux_failures=aux_failures, fowl_smarts=st.get("fowl_smarts"),
            rule_programs=st.get("rule_programs"), rule_dependencies=st.get("rule_dependencies"),
            computed_facts=st.get("computed_facts", False),
        )
    except Exception as e:  # noqa: BLE001
        return mol_id, [], f"error: {e}", aux_failures

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
                    return mol_id, sorted(fired), "timeout", aux_failures
        else:
            ctl.solve(on_model=_on_model)
    except Exception as e:  # noqa: BLE001
        return mol_id, [], f"error: {e}", aux_failures
    return mol_id, sorted(fired), "", aux_failures


def build_ilp_preds_tensor(
    programs: dict[str, str],
    molecules_df: pd.DataFrame,
    mol_order: list[str],
    output_npy_path: str,
    output_meta_path: str,
    predicate_set: str = "atoms",
    aux_timeout: float | None = None,
    aux_library_dir: str | None = None,
    label_timeout: float = 60.0,
    n_jobs: int = 1,
    computed_facts: bool = False,
) -> pd.DataFrame:
    """
    Evaluate ILP programs against every molecule in mol_order and save the
    result as a boolean numpy tensor.

    Builds background knowledge from scratch (not from existing bk.pl problem
    files), using the same predicate-set logic as ILPProblemBuilder.build_bk so
    the programs are evaluated against the BK they were learned on. Intended for
    generating full split prediction tensors used by ensemble_construct.

    Args:
        programs:         class_id → Prolog program string (only classes with
                          a learned program; null programs must be excluded).
        molecules_df:     DataFrame with 'mol' column; index = molecule IDs
                          (string ChEBI IDs). Built from the raw SDF via
                          extract_molecules().
        mol_order:        ordered list of molecule IDs to evaluate.
        output_npy_path:  destination .npy file for the boolean tensor.
        output_meta_path: destination JSON file for mol_order / class_labels.
        predicate_set:    which background-knowledge predicate set to build
                          (must match the run being predicted for).
        aux_timeout:      per-predicate timeout for LLM auxiliary predicates
                          (llm_generated_fgs only); None → library default.
        aux_library_dir:  shared auxiliary-predicate library directory (needed
                          to load predicates for llm_generated_fgs); None →
                          DEFAULT_AUX_LIBRARY_DIR.
        label_timeout:    Clingo solving timeout in seconds (default 60).
        n_jobs:           number of worker processes to evaluate molecules in
                          parallel. 1 (default) runs serially in-process; <= 0
                          uses all available CPU cores. Molecules are independent,
                          so this scales close to linearly with cores.

    Returns a boolean DataFrame (index=mol_order, columns=sorted class IDs).
    """
    import tqdm
    import clingo
    from chebILP.predicate_generation.auxiliary_predicates import DEFAULT_AUX_TIMEOUT

    if aux_timeout is None:
        aux_timeout = DEFAULT_AUX_TIMEOUT

    ilp_class_ids = sorted(programs.keys())
    n_mols = len(mol_order)
    n_classes = len(ilp_class_ids)
    print(f"Building ILP predictions tensor: {n_mols} molecules × {n_classes} classes "
          f"(predicate_set={predicate_set})")

    # Molecules we actually have; anything missing stays False in the tensor.
    present_rows = molecules_df[molecules_df.index.isin(set(mol_order))].dropna(subset=["mol"])
    present_mol_ids = [str(i) for i in present_rows.index]
    missing = len(set(mol_order)) - len(set(present_mol_ids))
    if missing:
        print(f"  {missing} molecules missing from molecules_df — predicted False")

    # Column index per target label, and row index per molecule id.
    col_of = {f"chebi_{cls_id}": j for j, cls_id in enumerate(ilp_class_ids)}
    mol_idx = {mol_id: i for i, mol_id in enumerate(mol_order)}
    tensor = np.zeros((n_mols, n_classes), dtype=bool)

    # Parse-check every program once and keep the valid ones (class_id -> program). Each
    # program's only derived head is its unique chebi_{id} target, so they never interfere.
    valid_programs: dict[str, str] = {}
    for cls_id in ilp_class_ids:
        prog = "\n".join(line for line in programs[cls_id].split("\n") if line.strip())
        try:
            clingo.Control(logger=_silent_clingo_logger).add("t", [], prog)  # syntax check only
            valid_programs[cls_id] = prog
        except RuntimeError as e:
            print(f"  Skipping ChEBI:{cls_id} — failed to parse program: {e}")

    # Auxiliary predicates (llm_generated_fgs) are gathered across all classes and computed
    # into the single shared background knowledge. They are (re)loaded once per worker (their
    # exec-compiled extension functions are not picklable); ``aux_load_args`` is the spec
    # list a worker needs to reload them.
    aux_load_args = None
    if predicate_set == "llm_generated_fgs":
        aux_load_args = _aux_predicate_specs(valid_programs, aux_library_dir)
        print(f"{len(aux_load_args)} distinct auxiliary predicate implementation(s) referenced "
              f"by programs")

    # llm_generated_rules: gather the classes' rule programs (plus the library programs they
    # build on) and recompute their extensions in the background. RuleProgram objects are
    # picklable, so they ride in worker_state directly.
    rule_programs = rule_dependencies = None
    if predicate_set == "llm_generated_rules":
        rule_programs, rule_dependencies = _collect_rule_programs(valid_programs, aux_library_dir)
        print(f"{len(rule_programs)} distinct auxiliary rule(s) referenced by programs"
              + (f" (+{len(rule_dependencies)} dependencies)" if rule_dependencies else ""))

    # fowl predicates are class-specific (fowl_<cls_id>) and derived from a shared
    # SMARTS table. Gather the patterns for the classes being predicted so the same
    # facts the programs were learned on are emitted into each molecule's background.
    fowl_smarts = None
    if predicate_set == "fowl":
        from chebILP.ilp_problem_builder import load_fowl_smarts
        all_fowl_smarts = load_fowl_smarts()
        fowl_smarts = {cls_id: all_fowl_smarts[cls_id] for cls_id in valid_programs if cls_id in all_fowl_smarts}
        print(f"{len(fowl_smarts)} fowl SMARTS pattern(s) referenced by programs")

    programs_str = "\n".join(valid_programs.values())

    # Shared, read-only config handed to each molecule evaluation (main process for the
    # serial path, or every pool worker via the initializer).
    worker_state = dict(
        programs_str=programs_str, col_of=col_of, predicate_set=predicate_set,
        aux_timeout=aux_timeout, label_timeout=label_timeout, fowl_smarts=fowl_smarts,
        rule_programs=rule_programs, rule_dependencies=rule_dependencies,
        computed_facts=computed_facts,
    )

    # Ship mols as RDKit binary with all properties so nothing (stereo/CIP perception,
    # cached descriptors) is lost when they cross the process boundary.
    from rdkit import Chem
    all_props = Chem.PropertyPickleOptions.AllProps
    payloads = [
        (str(row.Index), row.mol.ToBinary(all_props)) for row in present_rows.itertuples()
    ]

    n_timeout = 0
    n_error = 0
    # Auxiliary predicates that could not be computed anywhere, name -> "timeout"|"error"
    # (a "timeout" on any molecule wins, since it drops the predicate entirely).
    aux_failures: dict[str, str] = {}

    def _apply(result):
        nonlocal n_timeout, n_error
        mol_id, fired, status, mol_aux_failures = result
        for name, reason in mol_aux_failures.items():
            if reason == "timeout" or name not in aux_failures:
                aux_failures[name] = reason
        if status == "timeout":
            n_timeout += 1
        elif status.startswith("error"):
            n_error += 1
            print(f"  Clingo evaluation failed for molecule {mol_id}: {status[len('error: '):]}")
            return
        i = mol_idx.get(mol_id)
        if i is not None:
            for name in fired:
                tensor[i, col_of[name]] = True

    # Molecules are independent, so each is evaluated against its own tiny Clingo
    # grounding (one molecule's background knowledge + all programs). With n_jobs != 1
    # these are farmed out to worker processes for near-linear speedup.
    if n_jobs is None or n_jobs <= 0:
        n_jobs = os.cpu_count() or 1

    if n_jobs == 1:
        _worker_init(worker_state, aux_load_args)
        for payload in tqdm.tqdm(payloads, desc="Molecules"):
            _apply(_evaluate_molecule(payload))
    else:
        from concurrent.futures import ProcessPoolExecutor

        print(f"Evaluating {len(payloads)} molecules across {n_jobs} worker process(es)")
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_worker_init,
            initargs=(worker_state, aux_load_args),
        ) as ex:
            for result in tqdm.tqdm(
                ex.map(_evaluate_molecule, payloads, chunksize=8),
                total=len(payloads), desc="Molecules",
            ):
                _apply(result)

    if n_timeout:
        print(f"  {n_timeout} molecule(s) hit the {label_timeout}s solving timeout — those rows may be incomplete")
    if n_error:
        print(f"  {n_error} molecule(s) failed to evaluate — predicted False")
    if aux_failures:
        n_aux_timeout = sum(1 for r in aux_failures.values() if r == "timeout")
        n_aux_error = sum(1 for r in aux_failures.values() if r == "error")
        print(f"  {len(aux_failures)} auxiliary predicate(s) could not be calculated "
              f"({n_aux_timeout} timed out, {n_aux_error} errored) and were dropped from the "
              f"background knowledge")
        print(f"  Failed auxiliary predicates: {', '.join(sorted(aux_failures.keys()))}")

    np.save(output_npy_path, tensor)
    with open(output_meta_path, "w") as f:
        json.dump({"mol_order": mol_order, "class_labels": ilp_class_ids}, f)
    print(f"Saved ILP predictions tensor: {output_npy_path} ({n_mols} × {n_classes})")

    return pd.DataFrame(tensor, index=mol_order, columns=ilp_class_ids)


def predict_smiles(
    smiles_list: list[str],
    rules_file: str,
    target_predicates: list[str],
    verbose: bool = False,
    predicate_set: str = "atoms",
    aux_library_dir: str | None = None,
    aux_timeout: float | None = None,
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
        verbose:           If True, print the satisfied predicates for each molecule.
        predicate_set:     Which background-knowledge predicate set to build (must match
                           the set the rules were learned on), via build_full_background.
        aux_library_dir:   shared auxiliary-predicate library directory (only needed for the
                           llm_generated_fgs predicate set); None → DEFAULT_AUX_LIBRARY_DIR.
        aux_timeout:       Per-predicate timeout for LLM auxiliary predicates
                           (llm_generated_fgs only); None → library default.

    Returns:
        list of dicts, one per input SMILES, each with keys:
          - ``"SMILES"``: the input SMILES string.
          - ``"labels"``: list of target predicates the molecule satisfies (empty if none
            or if the SMILES is invalid).
          - ``"time"``: time (seconds) the Clingo evaluation took, or None for invalid SMILES.
    """
    from rdkit import Chem
    from chebILP.ilp_problem_builder import build_full_background
    from chebILP.evaluation.clingo_eval import evaluate_with_clingo
    from chebILP.predicate_generation.auxiliary_predicates import DEFAULT_AUX_TIMEOUT

    if aux_timeout is None:
        aux_timeout = DEFAULT_AUX_TIMEOUT

    with open(rules_file, "r") as f:
        rules = [line.strip() for line in f if line.strip() and not line.strip().startswith("%")]

    # Auxiliary predicates (llm_generated_fgs) are class-specific, so load the ones
    # referenced by the rules for the target classes (first definition of each aux_ name
    # wins). They are molecule-independent, so gather them once here.
    aux_predicates = None
    if predicate_set == "llm_generated_fgs":
        import re
        from chebILP.predicate_generation.auxiliary_predicates import load_auxiliary_predicates
        used_aux_names = set(re.findall(r"\baux_\w+", "\n".join(rules)))
        class_ids = [t[len("chebi_"):] for t in target_predicates if t.startswith("chebi_")]
        seen: dict[str, object] = {}
        for cls_id in class_ids:
            for pred in load_auxiliary_predicates(cls_id, library_dir=aux_library_dir):
                if pred.name in used_aux_names:
                    seen.setdefault(pred.name, pred)
        aux_predicates = list(seen.values())

    # llm_generated_rules: gather the rule programs the target classes use (and the library
    # programs they build on) so the background recomputes their extensions.
    rule_programs = rule_dependencies = None
    if predicate_set == "llm_generated_rules":
        programs_by_class = {
            t[len("chebi_"):]: "\n".join(r for r in rules if f"chebi_{t[len('chebi_'):]}" in r)
            for t in target_predicates if t.startswith("chebi_")
        }
        rule_programs, rule_dependencies = _collect_rule_programs(programs_by_class, aux_library_dir)

    # fowl predicates are class-specific: gather the SMARTS patterns for the target
    # classes so the background emits the fowl_<cls_id> facts the rules reference.
    fowl_smarts = None
    if predicate_set == "fowl":
        from chebILP.ilp_problem_builder import load_fowl_smarts
        all_fowl_smarts = load_fowl_smarts()
        class_ids = [t[len("chebi_"):] for t in target_predicates if t.startswith("chebi_")]
        fowl_smarts = {cls_id: all_fowl_smarts[cls_id] for cls_id in class_ids if cls_id in all_fowl_smarts}

    results: list[dict] = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES (skipped): {smiles!r}")
            results.append({"SMILES": smiles, "labels": [], "time": None})
            continue

        mol_id = "m0"
        mol_df = pd.DataFrame([{"mol": mol}], index=[mol_id])
        background_facts = build_full_background(
            mol_df, predicate_set=predicate_set,
            aux_predicates=aux_predicates, aux_timeout=aux_timeout,
            fowl_smarts=fowl_smarts, rule_programs=rule_programs,
            rule_dependencies=rule_dependencies, computed_facts=computed_facts,
        )
        print(f"Evaluating {smiles!r} against {len(rules)} rules and {len(background_facts)} background facts...")
        print(f"  Target predicates: {', '.join(target_predicates)}")
        print(f"  Background knowledge: {background_facts}")

        start_time = time.perf_counter()
        try:
            positives = evaluate_with_clingo(rules, background_facts, target_predicates, [mol_id])
        except Exception as e:
            print(f"Clingo evaluation failed for {smiles!r}: {e}")
            positives = {}
        elapsed = time.perf_counter() - start_time

        labels = [target for target in target_predicates if mol_id in positives.get(target, [])]
        results.append({"SMILES": smiles, "labels": labels, "time": elapsed})

    if verbose:
        for entry in results:
            labels = entry["labels"]
            time_str = f"{entry['time']:.3f}s" if entry["time"] is not None else "n/a"
            print(f"{entry['SMILES']} [{time_str}]: {', '.join(labels) if labels else '(none)'}")

    return results


def test_chebi_classes(run_to_evaluate, problem_dir: str, predicate_set: str, results_dir, selection_mode: Optional[str], selection_k: Optional[int], test_on: Literal["validation", "test"] = "test", verbose: bool = False, **kwargs):

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
                chebi_id, prog_str,
                exs_file=get_exs_path(chebi_id, split=test_on, base_dir=problem_dir),
                bk_file=get_bk_path(chebi_id, predicate_set=predicate_set, split=test_on, base_dir=problem_dir,
                                    selection_mode=selection_mode, selection_k=selection_k),
                return_details=True,
            )
            if verbose:
                balanced_acc = 0.5 * (conf_matrix["TP"] / (conf_matrix["TP"] + conf_matrix["FN"]) if (conf_matrix["TP"] + conf_matrix["FN"]) > 0 else 0) + 0.5 * (conf_matrix["TN"] / (conf_matrix["TN"] + conf_matrix["FP"]) if (conf_matrix["TN"] + conf_matrix["FP"]) > 0 else 0)
                f1_score = conf_matrix["TP"] / (conf_matrix["TP"] + 0.5 * (conf_matrix["FP"] + conf_matrix["FN"])) if (conf_matrix["TP"] + conf_matrix["FP"] + conf_matrix["FN"]) > 0 else 0.0
                print(f"  F1: {f1_score:.2f}, BA: {balanced_acc:.2f} TPs: {conf_matrix['TP']}, FPs: {conf_matrix['FP']}, TNs: {conf_matrix['TN']}, FNs: {conf_matrix['FN']}")
                for outcome in ("FP", "FN"):
                    subset = [d for d in details if d["outcome"] == outcome][:10]
                    print(f"  {'False Positive' if outcome == 'FP' else 'False Negative'} samples (showing up to 10):")
                    for d in subset:
                        print(f"    {d['id']} (true label: {d['true_label']})")
            
            with open(os.path.join(results_dir, "results.json"), "a+") as f:
                result_entry = {
                    "chebi_id": chebi_id,
                    "time_taken": time.perf_counter() - start_time,
                    "program": prog_str,
                    f"{test_on}_score": conf_matrix,
                    f"{test_on}_details": details
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
                    "error": str(e)
                }
                f.write(json.dumps(result_entry) + "\n")
