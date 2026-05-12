"""
EXP-006: Ensemble validation evaluation.

Computes validation metrics on the *correct* validation set for each class:
  - Positive: validation molecules that are descendants of the target class
  - Negative: validation molecules that are descendants of any direct parent of
              the target class, but NOT descendants of the target class

This is narrower than the existing exs.pl validation set, which uses
get_closest_negatives(direct_only=True) and may include unrelated classes.

Provides evaluation for both ILP programs (via Clingo) and DL predictions
(via pre-extracted numpy array from prepare_dl_preds.py).
"""

import json
import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from chebILP.clingo_eval import run_ilp_validation_clingo


# ---------------------------------------------------------------------------
# Correct validation set computation
# ---------------------------------------------------------------------------

def get_correct_val_set(
    target_id: str,
    chebi_graph,
    hierarchy_graph,
    molecules_df: pd.DataFrame,
    validation_ids: set,
) -> tuple[list[str], list[str]]:
    """
    Return the correct (restricted) validation set for a target class.

    Sample space = all validation molecules that are descendants of any direct
    parent of target_id.  Positives = descendants of target_id within that
    space; negatives = the rest.

    Returns:
        pos_ids: list of positive validation molecule IDs
        neg_ids: list of negative validation molecule IDs
                 (empty when target has no siblings in validation split)
    """
    mol_index = set(molecules_df.index)

    pos_ids = [
        str(d)
        for d in hierarchy_graph.predecessors(target_id)
        if str(d) in validation_ids and str(d) in mol_index
    ]
    pos_set = set(pos_ids)

    sample_space_by_parent = dict()
    for parent in chebi_graph.successors(target_id):
        sample_space_by_parent[parent] = set()
        for desc in hierarchy_graph.predecessors(parent):
            s = str(desc)
            if s in validation_ids and s in mol_index:
                sample_space_by_parent[parent].add(s)
    if len(sample_space_by_parent) == 0:
        return pos_ids, []
    sample_space = set.intersection(*sample_space_by_parent.values())
    neg_ids = list(sample_space - pos_set)
    return pos_ids, neg_ids


# ---------------------------------------------------------------------------
# ILP evaluation on correct validation set
# ---------------------------------------------------------------------------

def eval_ilp_on_correct_val(
    target_id: str,
    prog_str: str,
    pos_ids: list[str],
    neg_ids: list[str],
    molecules_df: pd.DataFrame,
    problem_dir: Optional[str] = None,
    predicate_set: str = "atoms",
) -> Optional[tuple[dict, list]]:
    """
    Evaluate an ILP program on the correct validation set.

    Builds exs.pl and bk.pl for the specific molecule subset and runs Clingo.
    When problem_dir is given the files are written to
    ``<problem_dir>/chebi_<target_id>/validation_correct/`` (persisted for
    later inspection / reuse); otherwise a temporary directory is used.

    Returns ``(conf_matrix, details)`` where conf_matrix is {TP,FP,TN,FN} and
    details is a list of per-sample dicts with keys id/true_label/outcome.
    Returns None when evaluation is not possible (no program or no positives).
    """
    if not prog_str or not pos_ids:
        return None

    all_ids = set(pos_ids) | set(neg_ids)
    relevant_mols = molecules_df[molecules_df.index.isin(all_ids)].dropna(subset=["mol"])

    valid_ids = set(relevant_mols.index)
    pos_ids = [i for i in pos_ids if i in valid_ids]
    neg_ids = [i for i in neg_ids if i in valid_ids]

    if not pos_ids:
        return None

    from chebILP.mol2ilp import build_background_chemlog  # lazy: avoids popper import at module level
    from chebILP.ilp_path_manager import get_exs_path, get_bk_path

    if problem_dir is not None:
        exs_path = get_exs_path(target_id, split="validation_correct", base_dir=problem_dir)
        bk_path = get_bk_path(target_id, split="validation_correct", predicate_set=predicate_set, base_dir=problem_dir)
        ctx = None  # no tempdir needed
    else:
        _tmpdir = tempfile.TemporaryDirectory()
        exs_path = os.path.join(_tmpdir.name, "exs.pl")
        bk_path = os.path.join(_tmpdir.name, "bk.pl")
        ctx = _tmpdir

    try:
        with open(exs_path, "w") as f:
            for mol_id in pos_ids:
                f.write(f"pos(chebi_{target_id}({mol_id})).\n")
            for mol_id in neg_ids:
                f.write(f"neg(chebi_{target_id}({mol_id})).\n")

        prolog_lines, _ = build_background_chemlog(relevant_mols)
        with open(bk_path, "w") as f:
            f.write("\n".join(prolog_lines) + "\n")

        result = run_ilp_validation_clingo(target_id, prog_str, exs_path, bk_path, return_details=True)
        if result is None:
            return None
        conf_matrix, details = result
        return conf_matrix, details
    except Exception as e:
        print(f"  Clingo eval failed for ChEBI:{target_id}: {e}")
        return None
    finally:
        if ctx is not None:
            ctx.cleanup()


# ---------------------------------------------------------------------------
# DL evaluation on correct validation set
# ---------------------------------------------------------------------------

def load_dl_preds(npy_path: str, meta_json_path: str) -> pd.DataFrame:
    """
    Load pre-extracted DL predictions (created by prepare_dl_preds.py).

    Returns a DataFrame with molecule_id as index and ChEBI class ID as
    columns, containing float prediction scores.
    """
    arr = np.load(npy_path)
    with open(meta_json_path, "r") as f:
        meta = json.load(f)
    mol_order = meta["mol_order"]
    class_labels = meta["class_labels"]
    return pd.DataFrame(arr, index=mol_order, columns=class_labels)


def eval_dl_on_correct_val(
    target_id: str,
    pos_ids: list[str],
    neg_ids: list[str],
    dl_preds: pd.DataFrame,
    threshold: float = 0.5,
) -> Optional[dict]:
    """
    Compute confusion matrix for the DL model on the correct validation set.

    Args:
        target_id:  ChEBI class ID (str)
        pos_ids:    positive molecule IDs
        neg_ids:    negative molecule IDs
        dl_preds:   DataFrame (mol_id → class_id → prediction score)
        threshold:  classification threshold
    """
    if target_id not in dl_preds.columns:
        return None
    if not pos_ids:
        return None

    col = dl_preds[target_id]

    # only evaluate molecules present in dl_preds index
    pos_ids = [i for i in pos_ids if i in col.index]
    neg_ids = [i for i in neg_ids if i in col.index]

    if not pos_ids:
        return None

    tp = int((col[pos_ids] >= threshold).sum())
    fn = len(pos_ids) - tp
    fp = int((col[neg_ids] >= threshold).sum()) if neg_ids else 0
    tn = len(neg_ids) - fp

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


# ---------------------------------------------------------------------------
# Loading ILP run results
# ---------------------------------------------------------------------------

def load_ilp_results(run_dir: str) -> dict[str, dict]:
    """Load results.json from a run directory → {chebi_id: result_dict}."""
    results = {}
    with open(os.path.join(run_dir, "results.json"), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                results[str(entry["chebi_id"])] = entry
            except json.JSONDecodeError:
                pass
    return results


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def balanced_accuracy(conf: dict) -> Optional[float]:
    tp, fp, tn, fn = conf["TP"], conf["FP"], conf["TN"], conf["FN"]
    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    if tpr is None or tnr is None:
        return None
    return 0.5 * (tpr + tnr)


def _flat_metrics(prefix: str, conf: Optional[dict]) -> dict:
    if conf is None:
        return {f"{prefix}_{k}": None for k in ("tp", "fp", "tn", "fn", "balanced_acc")}
    ba = balanced_accuracy(conf)
    return {
        f"{prefix}_tp": conf["TP"],
        f"{prefix}_fp": conf["FP"],
        f"{prefix}_tn": conf["TN"],
        f"{prefix}_fn": conf["FN"],
        f"{prefix}_balanced_acc": round(ba, 4) if ba is not None else None,
    }


# ---------------------------------------------------------------------------
# Main evaluation orchestrator
# ---------------------------------------------------------------------------

def run_correct_val_eval(
    target_ids: list[str],
    ilp_run_dirs: dict[str, str],
    dl_preds: pd.DataFrame,
    chebi_graph,
    hierarchy_graph,
    molecules_df: pd.DataFrame,
    validation_ids: set,
    output_path: str,
    problem_dir: Optional[str] = None,
    predicate_set: str = "atoms",
):
    """
    For each target class, evaluate all models on the correct validation set
    and write results to a CSV file.

    Args:
        target_ids:    list of ChEBI class IDs to evaluate
        ilp_run_dirs:  dict of run_name → results directory path
        dl_preds:      DataFrame from load_dl_preds()
        chebi_graph:   non-transitive ChEBI hierarchy (from ILPProblemBuilder)
        hierarchy_graph: transitive ChEBI hierarchy
        molecules_df:  molecules DataFrame with mol objects
        validation_ids: set of validation molecule IDs (strings)
        output_path:   where to write the summary CSV
        problem_dir:   ILP problems base dir; when set, exs.pl and bk.pl are
                       written to <problem_dir>/chebi_<id>/validation_correct/
        predicate_set: predicate set used for BK (default "atoms")

    Side effects per ILP run:
        Appends lines to ``<run_dir>/results_correct_val.json``, one JSON
        object per class with keys chebi_id, validation_correct_score,
        validation_correct_details (list of {id, true_label, outcome}).
    """
    # Load all ILP programs
    ilp_programs = {}
    for run_name, run_dir in ilp_run_dirs.items():
        ilp_programs[run_name] = load_ilp_results(run_dir)
        print(f"Loaded {len(ilp_programs[run_name])} results from '{run_name}'")

    # Load already-evaluated classes per run (for resumability)
    cached: dict[str, dict[str, dict]] = {}  # run_name → {chebi_id → stored entry}
    for run_name, run_dir in ilp_run_dirs.items():
        json_path = os.path.join(run_dir, "results_correct_val.json")
        cached[run_name] = {}
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        cached[run_name][str(entry["chebi_id"])] = entry
                    except json.JSONDecodeError:
                        pass
            n = len(cached[run_name])
            if n:
                print(f"  '{run_name}': {n} classes already evaluated, will skip those")

    rows = []
    for target_id in target_ids:
        # correct val set and DL are always (re-)computed — both are fast
        pos_ids, neg_ids = get_correct_val_set(
            target_id, chebi_graph, hierarchy_graph, molecules_df, validation_ids
        )
        has_negatives = len(neg_ids) > 0

        row = {
            "chebi_id": target_id,
            "pos_count": len(pos_ids),
            "neg_count": len(neg_ids),
            "has_negatives": has_negatives,
        }

        dl_conf = eval_dl_on_correct_val(target_id, pos_ids, neg_ids, dl_preds)
        row.update(_flat_metrics("dl", dl_conf))

        # ILP evaluations — skip runs that already have a result for this class
        for run_name, run_dir in ilp_run_dirs.items():
            if target_id in cached[run_name]:
                ilp_conf = cached[run_name][target_id].get("validation_correct_score")
                print(f"  ILP [{run_name}] ChEBI:{target_id}: cached")
            else:
                entry = ilp_programs[run_name].get(target_id, {})
                prog_str = entry.get("program")
                ilp_conf, details = None, None
                if prog_str:
                    print(f"  ILP [{run_name}] ChEBI:{target_id}: evaluating...")
                    result = eval_ilp_on_correct_val(
                        target_id, prog_str, pos_ids, neg_ids, molecules_df,
                        problem_dir=problem_dir, predicate_set=predicate_set,
                    )
                    if result is not None:
                        ilp_conf, details = result
                else:
                    print(f"  ILP [{run_name}] ChEBI:{target_id}: no program")

                with open(os.path.join(run_dir, "results_correct_val.json"), "a") as f:
                    f.write(json.dumps({
                        "chebi_id": target_id,
                        "validation_correct_score": ilp_conf,
                        "validation_correct_details": details,
                    }) + "\n")

            row.update(_flat_metrics(run_name, ilp_conf))

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved correct-val metrics to {output_path}")
    return df


# ---------------------------------------------------------------------------
# Ensemble predictor
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """
    Hierarchical ensemble combining DL and ILP models.

    At init:
    1. Builds a label-projected hierarchy: for each label, the closest ancestors
       that are also labels (BFS upward through non-label nodes).
    2. Selects the most trustworthy model per class (highest balanced accuracy on
       the correct validation set from run_correct_val_eval).
       - "always_positive" when a class has no negatives in its correct val set.
       - Falls back to DL for classes absent from the metrics CSV.

    Prediction for a molecule M (predict_sample):
    - Processes labels in topological order (parents before children).
    - Label C is predicted positive iff:
        (a) every label parent of C is already predicted positive, AND
        (b) the trusted model for C predicts M positive.
    - DL predictions use the pre-loaded DataFrame (O(1) lookup per class).
    - ILP predictions first consult the per-molecule cache loaded from
      results_correct_val.json (O(1) lookup).  Clingo is only called for
      molecules absent from that cache, and requires molecules_df with mol
      objects.
    """

    def __init__(
        self,
        metrics_csv: str,
        ilp_run_dirs: dict[str, str],
        dl_preds: pd.DataFrame,
        chebi_graph,
        label_set: list[str],
    ):
        import networkx as nx

        self.dl_preds = dl_preds
        self.label_set = set(str(l) for l in label_set)

        print("Loading ILP programs and validation cache...")
        self.ilp_programs: dict[str, dict[str, Optional[str]]] = {}
        # ilp_val_cache[run_name][chebi_id][mol_id] = True (predicted pos) / False (predicted neg)
        # populated from results_correct_val.json written by eval_ilp_on_correct_val
        self.ilp_val_cache: dict[str, dict[str, dict[str, bool]]] = {}
        for run_name, run_dir in ilp_run_dirs.items():
            results = load_ilp_results(run_dir)
            self.ilp_programs[run_name] = {cid: e.get("program") for cid, e in results.items()}
            n_prog = sum(1 for p in self.ilp_programs[run_name].values() if p)

            self.ilp_val_cache[run_name] = {}
            json_path = os.path.join(run_dir, "results_correct_val.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            cls_id = str(entry["chebi_id"])
                            details = entry.get("validation_correct_details") or []
                            # outcome in {TP, FP} means the rule fired (predicted positive)
                            self.ilp_val_cache[run_name][cls_id] = {
                                d["id"]: d["outcome"] in ("TP", "FP")
                                for d in details
                            }
                        except (json.JSONDecodeError, KeyError):
                            pass
            n_cached = sum(len(v) for v in self.ilp_val_cache[run_name].values())
            print(f"  '{run_name}': {n_prog} programs, {len(self.ilp_val_cache[run_name])} classes cached ({n_cached} molecule results)")

        print("Building label-projected hierarchy...")
        self.label_parents = self._build_label_parents(chebi_graph)
        label_graph = nx.DiGraph()
        for cls in self.label_set:
            label_graph.add_node(cls)
            for parent in self.label_parents[cls]:
                label_graph.add_edge(parent, cls)
        self.topo_order = list(nx.topological_sort(label_graph))

        print("Selecting trusted models from metrics CSV...")
        self.trusted_model = self._select_trusted_models(metrics_csv)
        counts: dict[str, int] = {}
        for m in self.trusted_model.values():
            counts[m] = counts.get(m, 0) + 1
        print("  Trusted model counts:", counts)

    # ── Hierarchy construction ────────────────────────────────────────────

    def _build_label_parents(self, chebi_graph) -> dict[str, set]:
        """
        For each label, find its closest ancestors that are also labels.

        BFS upward from each label; stops at any encountered label node so that
        only the nearest label ancestors are collected (non-label intermediaries
        are skipped transparently).
        """
        label_parents: dict[str, set] = {cls: set() for cls in self.label_set}

        for cls in self.label_set:
            frontier = {str(p) for p in chebi_graph.successors(cls)}
            visited: set[str] = set()

            while frontier:
                next_frontier: set[str] = set()
                for node in frontier:
                    if node in visited:
                        continue
                    visited.add(node)
                    if node in self.label_set:
                        label_parents[cls].add(node)
                        # do not traverse above a found label node
                    else:
                        next_frontier.update(str(p) for p in chebi_graph.successors(node))
                frontier = next_frontier - visited

        return label_parents

    # ── Model selection ───────────────────────────────────────────────────

    def _select_trusted_models(self, metrics_csv: str) -> dict[str, str]:
        """
        Pick the model with the highest balanced accuracy per class.
        Classes absent from the CSV default to "dl".
        """
        df = pd.read_csv(metrics_csv)
        ba_cols = {
            col.replace("_balanced_acc", ""): col
            for col in df.columns if col.endswith("_balanced_acc")
        }

        trusted: dict[str, str] = {}
        for _, row in df.iterrows():
            cls_id = str(row["chebi_id"])

            has_neg = row.get("has_negatives", True)
            if isinstance(has_neg, str):
                has_neg = has_neg.strip().lower() == "true"
            if not has_neg:
                trusted[cls_id] = "always_positive"
                continue

            best_model = "dl"
            best_ba = float(row.get("dl_balanced_acc") or 0)

            for model_name, ba_col in ba_cols.items():
                ba = row.get(ba_col)
                try:
                    ba = float(ba)
                except (TypeError, ValueError):
                    continue
                if ba > best_ba:
                    if model_name != "dl":
                        prog = self.ilp_programs.get(model_name, {}).get(cls_id)
                        if not prog:
                            continue
                    best_model = model_name
                    best_ba = ba

            trusted[cls_id] = best_model

        # default for labels absent from metrics CSV
        for cls in self.label_set:
            if cls not in trusted:
                trusted[cls] = "dl"

        return trusted

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_sample(self, mol_id: str, molecules_df=None) -> set[str]:
        """
        Predict all classes for a single molecule.

        Args:
            mol_id:       ChEBI molecule ID (string)
            molecules_df: DataFrame with mol objects — only needed when an ILP
                          model is trusted for at least one class

        Returns set of class IDs predicted positive.
        """
        positive: set[str] = set()

        for cls in self.topo_order:
            # hierarchical gate: all label parents must already be predicted positive
            if not all(p in positive for p in self.label_parents.get(cls, set())):
                continue

            if self._eval_model_for_mol(cls, mol_id, molecules_df):
                positive.add(cls)

        return positive

    def _eval_model_for_mol(self, cls: str, mol_id: str, molecules_df) -> bool:
        model = self.trusted_model.get(cls, "dl")

        if model == "always_positive":
            return True

        if model == "dl":
            if cls not in self.dl_preds.columns or mol_id not in self.dl_preds.index:
                return False
            return float(self.dl_preds.at[mol_id, cls]) >= 0.5

        # ILP model
        prog = self.ilp_programs.get(model, {}).get(cls)
        if not prog:
            return False
        return self._run_ilp_for_mol(cls, mol_id, prog, run_name=model, molecules_df=molecules_df)

    def _run_ilp_for_mol(
        self, cls: str, mol_id: str, prog_str: str, run_name: str, molecules_df
    ) -> bool:
        """
        Evaluate an ILP rule for a single molecule.

        Checks ilp_val_cache[run_name][cls][mol_id] first (populated from
        results_correct_val.json).  Falls back to Clingo only when the molecule
        is not in the cache, which requires molecules_df with mol objects.
        """
        mol_cache = self.ilp_val_cache.get(run_name, {}).get(cls)
        if mol_cache is not None and mol_id in mol_cache:
            return mol_cache[mol_id]

        # Clingo fallback for molecules not covered by the validation cache
        if molecules_df is None:
            return False

        from chebILP.mol2ilp import build_background_chemlog

        mol_row = molecules_df[molecules_df.index == mol_id].dropna(subset=["mol"])
        if mol_row.empty:
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            exs_path = os.path.join(tmpdir, "exs.pl")
            bk_path = os.path.join(tmpdir, "bk.pl")

            with open(exs_path, "w") as f:
                f.write(f"pos(chebi_{cls}({mol_id})).\n")

            prolog_lines, _ = build_background_chemlog(mol_row)
            with open(bk_path, "w") as f:
                f.write("\n".join(prolog_lines) + "\n")

            try:
                result = run_ilp_validation_clingo(cls, prog_str, exs_path, bk_path)
                return result.get("TP", 0) > 0
            except Exception as e:
                print(f"  ILP eval failed for {cls}/{mol_id}: {e}")
                return False

    def predict_validation_set(
        self,
        validation_mol_ids: list[str],
        molecules_df=None,
    ) -> pd.DataFrame:
        """
        Run ensemble predictions for all validation molecules.

        Returns a boolean DataFrame: rows = mol_ids, columns = labels (topo order).
        """
        import tqdm

        records = []
        for mol_id in tqdm.tqdm(validation_mol_ids, desc="Ensemble predictions"):
            pos = self.predict_sample(mol_id, molecules_df)
            records.append({cls: cls in pos for cls in self.topo_order})

        return pd.DataFrame(records, index=validation_mol_ids, columns=self.topo_order)
