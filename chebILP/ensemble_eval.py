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
from typing import Literal, Optional

import numpy as np
import pandas as pd

from chebILP.clingo_eval import run_ilp_validation_clingo
from chebILP.mol2ilp import get_direct_neighbors



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


def eval_dl_on_ids(
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


# ---------------------------------------------------------------------------
# Ensemble predictor
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """
    Hierarchical ensemble combining DL and ILP models.

    ilp_val_runs: directories produced by test_chebi_classes(test_on="validation"),
    used only for model selection. Each results.json entry needs validation_score
    and validation_details. DL val scores are derived from the same pos/neg IDs.

    ilp_runs: ILP learning/target runs providing the actual programs for prediction.
    Their validation_details (if present) also populate the prediction cache.
    Run names must match between ilp_val_runs and ilp_runs for model selection
    to resolve to the correct programs.

    For validation-set predictions: dl_preds covers the validation molecules.
    For test-set predictions: pass dl_preds with test-molecule scores and
    dl_preds_val with validation-molecule scores for model selection.

    Prediction for a molecule M (predict_sample):
    - Processes labels in topological order (parents before children).
    - Label C is predicted positive iff:
        (a) every label parent of C is already predicted positive, AND
        (b) the trusted model for C predicts M positive.
    - DL predictions use self.dl_preds (O(1) lookup).
    - ILP predictions use ilp_val_cache (from validation_details) for val
      molecules; Clingo fallback for uncached molecules (requires molecules_df).
    """

    def __init__(
        self,
        ilp_val_runs: dict[str, str],
        ilp_runs: dict[str, str],
        dl_preds: pd.DataFrame,
        chebi_graph,
        label_set: list[str],
        classifier_chain_mode=True,
        model_selection_metric: Literal["balanced_acc", "f1", "weighted_f1"] = "f1",
        dl_preds_val: Optional[pd.DataFrame] = None,
    ):
        import networkx as nx

        self.dl_preds = dl_preds
        self.label_set = set(str(l) for l in label_set)
        self.classifier_chain_mode = classifier_chain_mode
        self.model_selection_metric = model_selection_metric
        _dl_preds_val = dl_preds_val if dl_preds_val is not None else dl_preds

        # ilp_val_cache[run_name][chebi_id][mol_id] = True/False (predicted positive)
        self.ilp_val_cache: dict[str, dict[str, dict[str, bool]]] = {}
        # ilp_val_scores[run_name][cls_id] = conf matrix (for model selection)
        self.ilp_val_scores: dict[str, dict[str, Optional[dict]]] = {}
        # dl_val_scores[cls_id] = conf matrix computed from validation_details + dl_preds_val
        self.dl_val_scores: dict[str, Optional[dict]] = {}

        print("Loading validation runs (for model selection)...")
        for run_name, run_dir in ilp_val_runs.items():
            results = load_ilp_results(run_dir)
            self.ilp_val_scores[run_name] = {}
            self.ilp_val_cache.setdefault(run_name, {})

            for cls_id, entry in results.items():
                self.ilp_val_scores[run_name][cls_id] = entry.get("validation_score")
                details = entry.get("validation_details") or []
                if details:
                    self.ilp_val_cache[run_name][cls_id] = {
                        d["id"]: d["outcome"] in ("TP", "FP")
                        for d in details
                    }
                    if cls_id not in self.dl_val_scores:
                        pos_ids = [d["id"] for d in details if d["true_label"] == "pos"]
                        neg_ids = [d["id"] for d in details if d["true_label"] == "neg"]
                        self.dl_val_scores[cls_id] = eval_dl_on_ids(cls_id, pos_ids, neg_ids, _dl_preds_val)

            n_cached = sum(len(v) for v in self.ilp_val_cache[run_name].values())
            print(f"  '{run_name}': {len(results)} classes, {n_cached} cached molecule results")

        print("Loading ILP runs (programs + cache)...")
        self.ilp_programs: dict[str, dict[str, Optional[str]]] = {}
        for run_name, run_dir in ilp_runs.items():
            results = load_ilp_results(run_dir)
            self.ilp_programs[run_name] = {cid: e.get("program") for cid, e in results.items()}
            cache = self.ilp_val_cache.setdefault(run_name, {})
            for cls_id, entry in results.items():
                if cls_id not in cache:
                    details = entry.get("validation_details") or []
                    if details:
                        cache[cls_id] = {
                            d["id"]: d["outcome"] in ("TP", "FP")
                            for d in details
                        }
            n_prog = sum(1 for p in self.ilp_programs[run_name].values() if p)
            print(f"  '{run_name}': {n_prog} programs")

        print("Building label-projected hierarchy...")
        self.label_parents = self._build_label_parents(chebi_graph)
        label_graph = nx.DiGraph()
        for cls in self.label_set:
            label_graph.add_node(cls)
            for parent in self.label_parents[cls]:
                label_graph.add_edge(parent, cls)
        self.topo_order = list(nx.topological_sort(label_graph))

        print("Selecting models...")
        if model_selection_metric == "weighted_f1":
            self.model_weights = self._compute_model_weights()
            self.trusted_model = {}
            counts: dict[str, int] = {}
            for w in self.model_weights.values():
                for name in w:
                    counts[name] = counts.get(name, 0) + 1
            print("  Weighted model participation counts:", counts)
        else:
            self.model_weights = {}
            self.trusted_model = self._select_trusted_models()
            counts = {}
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
                    else:
                        next_frontier.update(str(p) for p in chebi_graph.successors(node))
                frontier = next_frontier - visited

        return label_parents

    # ── Model selection ───────────────────────────────────────────────────

    @staticmethod
    def _f1_from_conf(conf: Optional[dict]) -> Optional[float]:
        if conf is None:
            return None
        tp, fp, fn = conf.get("TP", 0), conf.get("FP", 0), conf.get("FN", 0)
        if tp + fp + fn == 0:
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision == 0 and recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _compute_score_from_conf(self, conf: Optional[dict]) -> Optional[float]:
        if conf is None:
            return None
        if self.model_selection_metric == "balanced_acc":
            return balanced_accuracy(conf)
        return self._f1_from_conf(conf)

    def _has_negatives(self, cls_id: str) -> bool:
        dl_conf = self.dl_val_scores.get(cls_id)
        if dl_conf is not None and (dl_conf.get("TN", 0) + dl_conf.get("FP", 0)) > 0:
            return True
        for run_scores in self.ilp_val_scores.values():
            sc = run_scores.get(cls_id)
            if sc is not None and (sc.get("TN", 0) + sc.get("FP", 0)) > 0:
                return True
        return False

    def _select_trusted_models(self) -> dict[str, str]:
        """
        Pick the single best model per class by the selected metric.
        Classes absent from all validation runs default to "dl".
        """
        trusted: dict[str, str] = {}

        all_cls_ids: set[str] = set()
        for run_scores in self.ilp_val_scores.values():
            all_cls_ids.update(run_scores.keys())

        for cls_id in all_cls_ids:
            if not self._has_negatives(cls_id):
                trusted[cls_id] = "always_positive"
                continue

            dl_score = self._compute_score_from_conf(self.dl_val_scores.get(cls_id))
            best_model = "dl"
            best_score = dl_score if dl_score is not None else float("-inf")

            for run_name, run_scores in self.ilp_val_scores.items():
                ilp_score = self._compute_score_from_conf(run_scores.get(cls_id))
                if ilp_score is not None and ilp_score > best_score:
                    if self.ilp_programs.get(run_name, {}).get(cls_id):
                        best_model = run_name
                        best_score = ilp_score

            trusted[cls_id] = best_model

        for cls in self.label_set:
            if cls not in trusted:
                trusted[cls] = "dl"

        return trusted

    def _compute_model_weights(self) -> dict[str, dict[str, float]]:
        """
        Compute per-class model weights as own_F1 / sum(all_F1s).
        Only models with programs are included alongside DL.
        Classes with no negatives get weight 1.0 on "always_positive".
        Classes absent from validation runs default to {"dl": 1.0}.
        """
        weights: dict[str, dict[str, float]] = {}

        all_cls_ids: set[str] = set()
        for run_scores in self.ilp_val_scores.values():
            all_cls_ids.update(run_scores.keys())

        for cls_id in all_cls_ids:
            if not self._has_negatives(cls_id):
                weights[cls_id] = {"always_positive": 1.0}
                continue

            model_f1s: dict[str, float] = {}

            dl_f1 = self._f1_from_conf(self.dl_val_scores.get(cls_id))
            if dl_f1 is not None:
                model_f1s["dl"] = max(dl_f1, 0.0)

            for run_name, run_scores in self.ilp_val_scores.items():
                f1 = self._f1_from_conf(run_scores.get(cls_id))
                if f1 is not None and self.ilp_programs.get(run_name, {}).get(cls_id):
                    model_f1s[run_name] = max(f1, 0.0)

            total = sum(model_f1s.values())
            if total == 0:
                weights[cls_id] = {"dl": 1.0}
            else:
                weights[cls_id] = {name: f1 / total for name, f1 in model_f1s.items()}

        for cls in self.label_set:
            if cls not in weights:
                weights[cls] = {"dl": 1.0}

        return weights

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_sample(self, mol_id: str, molecules_df=None) -> set[str]:
        """
        Predict all classes for a single molecule.

        Args:
            mol_id:       ChEBI molecule ID (string)
            molecules_df: DataFrame with mol objects — only needed when an ILP
                          model is trusted for at least one class and the molecule
                          is not in the validation cache (e.g. test-set molecules)

        Returns set of class IDs predicted positive.
        """
        positive: set[str] = set()

        for cls in self.topo_order:
            always_pos = (
                self.trusted_model.get(cls) == "always_positive"
                or self.model_weights.get(cls, {}) == {"always_positive": 1.0}
            )
            if (self.classifier_chain_mode or always_pos) and not all(p in positive for p in self.label_parents.get(cls, set())):
                continue

            if self._eval_model_for_mol(cls, mol_id, molecules_df):
                positive.add(cls)

        return positive

    def _eval_model_for_mol(self, cls: str, mol_id: str, molecules_df) -> bool:
        if self.model_weights:
            weights = self.model_weights.get(cls, {"dl": 1.0})
            if weights == {"always_positive": 1.0}:
                return True
            return self._compute_weighted_prediction(cls, mol_id, weights, molecules_df)

        model = self.trusted_model.get(cls, "dl")
        if model == "always_positive":
            return True
        if model == "dl":
            if cls not in self.dl_preds.columns or mol_id not in self.dl_preds.index:
                return False
            return float(self.dl_preds.at[mol_id, cls]) >= 0.5
        prog = self.ilp_programs.get(model, {}).get(cls)
        if not prog:
            return False
        return self._run_ilp_for_mol(cls, mol_id, prog, run_name=model, molecules_df=molecules_df)

    def _compute_weighted_prediction(
        self, cls: str, mol_id: str, weights: dict[str, float], molecules_df
    ) -> bool:
        total = 0.0
        for model_name, weight in weights.items():
            if weight <= 0:
                continue
            if model_name == "dl":
                if cls in self.dl_preds.columns and mol_id in self.dl_preds.index:
                    total += weight * float(self.dl_preds.at[mol_id, cls])
            else:
                prog = self.ilp_programs.get(model_name, {}).get(cls)
                if prog:
                    pred = self._run_ilp_for_mol(cls, mol_id, prog, run_name=model_name, molecules_df=molecules_df)
                    total += weight * (1.0 if pred else 0.0)
        return total >= 0.5

    def _run_ilp_for_mol(
        self, cls: str, mol_id: str, prog_str: str, run_name: str, molecules_df
    ) -> bool:
        """
        Check ilp_val_cache (from validation_details) first; fall back to Clingo
        for molecules not in the cache (e.g. test-set molecules).
        Clingo fallback requires molecules_df with mol objects.
        """
        mol_cache = self.ilp_val_cache.get(run_name, {}).get(cls)
        if mol_cache is not None and mol_id in mol_cache:
            return mol_cache[mol_id]

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

    def predict_set(
        self,
        mol_ids: list[str],
        molecules_df=None,
    ) -> pd.DataFrame:
        """
        Run ensemble predictions for a list of molecules.

        Returns a boolean DataFrame: rows = mol_ids, columns = labels (topo order).
        """
        import tqdm

        records = []
        for mol_id in tqdm.tqdm(mol_ids, desc="Ensemble predictions"):
            pos = self.predict_sample(mol_id, molecules_df)
            records.append({cls: cls in pos for cls in self.topo_order})

        return pd.DataFrame(records, index=mol_ids, columns=self.topo_order)
