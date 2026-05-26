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
from typing import Literal, Optional
import networkx as nx

import numpy as np
import pandas as pd


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


def get_dl_direct_neighbor_scores(dl_preds: pd.DataFrame, chebi_graph: nx.DiGraph, hierarchy_graph, molecules_df, output_path: str):
    """
    Compute DL confusion matrices for all classes.

    Returns a dict: {class_id: conf_matrix or None if not computable}.
    """
    from chebILP.mol2ilp import get_direct_neighbors

    scores = {}
    for cls_id in dl_preds.columns:
        pos_ids, neg_ids = get_direct_neighbors(cls_id, chebi_graph, hierarchy_graph, molecules_df)
        scores[cls_id] = eval_dl_on_ids(cls_id, pos_ids, neg_ids, dl_preds)

    # save to csv
    with open(output_path, "w") as f:
        f.write("class_id,TP,FP,TN,FN\n")
        for cls_id, conf in scores.items():
            if conf is not None:
                f.write(f"{cls_id},{conf['TP']},{conf['FP']},{conf['TN']},{conf['FN']}\n")
            else:
                f.write(f"{cls_id},,,,\n")


# ---------------------------------------------------------------------------
# ILP predictions tensor
# ---------------------------------------------------------------------------

def load_ilp_preds(npy_path: str, meta_json_path: str) -> pd.DataFrame:
    """
    Load pre-computed ILP predictions tensor (created by build_ilp_preds_tensor).

    Returns a boolean DataFrame with molecule_id as index and ChEBI class ID as
    columns (only classes where ILP is the trusted model).
    """
    arr = np.load(npy_path)
    with open(meta_json_path, "r") as f:
        meta = json.load(f)
    return pd.DataFrame(arr, index=meta["mol_order"], columns=meta["class_labels"])


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


def compute_ground_truth(
    mol_order: list[str],
    class_ids: list[str],
    chebi_graph: nx.DiGraph,
) -> pd.DataFrame:
    """
    Build a boolean ground-truth DataFrame (mol_order × class_ids).
    Molecule M is positive for class C iff M is a descendant of C in the
    ChEBI hierarchy. Descendants are found via BFS over chebi_graph.predecessors.
    """
    mol_set = set(mol_order)
    gt: dict[str, list[bool]] = {}
    for cls_id in class_ids:
        descendants: set[str] = set()
        stack = [cls_id]
        while stack:
            node = stack.pop()
            for pred in chebi_graph.predecessors(node):
                s = str(pred)
                if s not in descendants:
                    descendants.add(s)
                    stack.append(s)
        gt[cls_id] = [mol_id in descendants for mol_id in mol_order]
    return pd.DataFrame(gt, index=mol_order, dtype=bool)


def compute_scores_from_preds(
    preds: pd.DataFrame,
    gt: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """
    Compute per-class confusion matrices from prediction and ground-truth DataFrames.
    Operates on the intersection of their molecule indices and class columns.
    """
    common_mols = preds.index.intersection(gt.index)
    common_cls = preds.columns.intersection(gt.columns)
    if common_mols.empty or common_cls.empty:
        return {}
    p = preds.loc[common_mols, common_cls] >= threshold
    g = gt.loc[common_mols, common_cls]
    tp = (p & g).sum()
    fp = (p & ~g).sum()
    fn = (~p & g).sum()
    tn = (~p & ~g).sum()
    return {
        str(cls_id): {"TP": int(tp[cls_id]), "FP": int(fp[cls_id]),
                      "TN": int(tn[cls_id]), "FN": int(fn[cls_id])}
        for cls_id in common_cls
    }


def build_label_parents(label_set, chebi_graph) -> dict[str, set]:
    label_parents: dict[str, set] = {cls: set() for cls in label_set}
    for cls in label_set:
        frontier = {str(p) for p in chebi_graph.successors(cls)}
        visited: set[str] = set()
        while frontier:
            next_frontier: set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                if node in label_set:
                    label_parents[cls].add(node)
                else:
                    next_frontier.update(str(p) for p in chebi_graph.successors(node))
            frontier = next_frontier - visited
    return label_parents


# ---------------------------------------------------------------------------
# Ensemble construction: model selection + ILP tensor generation
# ---------------------------------------------------------------------------

class EnsembleConstructor:
    """
    Performs model selection from ILP validation runs and builds the ILP
    predictions tensor for all molecules in a target split.

    Requires:
      dl_val_preds:  Full DL validation predictions (mol × class, float scores).
      ilp_val_runs:  Directories from test_chebi_classes(test_on="validation").
                     Each directory must contain a 'full_val_preds.npy' and
                     'full_val_preds_metadata.json' (same format as dl_val_preds)
                     in addition to 'results.json' for the ILP programs.

    Validation scores for both DL and ILP are computed directly from the
    prediction tensors against ground truth derived from the ChEBI hierarchy.
    """

    def __init__(
        self,
        ilp_val_runs: dict[str, str],
        dl_val_preds: pd.DataFrame,
        label_stats: dict[str, dict]|list[str],
        chebi_graph: nx.DiGraph,
        model_selection_metric: Literal["balanced_acc", "f1", "weighted_f1", "f1_bottom_ilp", "f1_bottom_ilp_best", "f1_bottom_ilp_multiple"] = "f1_bottom_ilp",
        predict_on: str = "validation",
    ):
        self.label_set = set(str(l) for l in label_stats.keys()) if isinstance(label_stats, dict) else set(label_stats)
        self.label_stats = label_stats
        self.dl_val_preds = dl_val_preds
        self.model_selection_metric = model_selection_metric
        self.chebi_graph = chebi_graph

        self.label_parents = build_label_parents(self.label_set, chebi_graph)
        label_graph = nx.DiGraph()
        for cls in self.label_set:
            label_graph.add_node(cls)
            for parent in self.label_parents[cls]:
                label_graph.add_edge(parent, cls)
        self.topo_order = list(nx.topological_sort(label_graph))

        # Ground truth: positive iff molecule is a ChEBI descendant of the class
        print("Computing ground truth from ChEBI hierarchy...")
        gt_class_ids = [c for c in self.label_set if c in dl_val_preds.columns]
        self._gt = compute_ground_truth(dl_val_preds.index.tolist(), gt_class_ids, chebi_graph)

        # DL validation scores derived from the predictions tensor
        self.dl_val_scores: dict[str, dict] = compute_scores_from_preds(dl_val_preds, self._gt)
        print(f"  DL val scores computed for {len(self.dl_val_scores)} classes")

        # Load ILP programs and validation predictions per run
        self.ilp_programs: dict[str, dict[str, Optional[str]]] = {}
        self.ilp_val_preds: dict[str, Optional[pd.DataFrame]] = {}
        self.ilp_val_scores: dict[str, dict[str, dict]] = {}

        print("Loading ILP validation runs...")
        for run_name, run_dir in ilp_val_runs.items():
            results = load_ilp_results(run_dir)
            self.ilp_programs[run_name] = {cid: e.get("program") for cid, e in results.items()}

            npy_path = os.path.join(run_dir, "full_val_preds.npy")
            meta_path = os.path.join(run_dir, "full_val_preds_metadata.json")
            if os.path.exists(npy_path) and os.path.exists(meta_path):
                ilp_preds = load_ilp_preds(npy_path, meta_path)
                self.ilp_val_preds[run_name] = ilp_preds
                self.ilp_val_scores[run_name] = compute_scores_from_preds(
                    ilp_preds.astype(float), self._gt
                )
                n_progs = sum(1 for p in self.ilp_programs[run_name].values() if p)
                print(f"  '{run_name}': {n_progs} programs, "
                      f"{ilp_preds.shape[0]} mol × {ilp_preds.shape[1]} classes in val tensor")
            else:
                self.ilp_val_preds[run_name] = None
                self.ilp_val_scores[run_name] = {}
                print(f"  Warning: no val tensor for '{run_name}' "
                      f"(expected {npy_path}); skipping run in model selection")

        # Load predictions for the target split (for slicing output tensor)
        # Always distinct from val preds when predict_on != "validation"
        slice_prefix = "val" if predict_on == "validation" else predict_on
        if predict_on == "validation":
            self.ilp_slice_preds = self.ilp_val_preds
        else:
            self.ilp_slice_preds: dict[str, Optional[pd.DataFrame]] = {}
            print(f"Loading ILP {predict_on} tensors for slicing...")
            for run_name, run_dir in ilp_val_runs.items():
                npy_path = os.path.join(run_dir, f"full_{slice_prefix}_preds.npy")
                meta_path = os.path.join(run_dir, f"full_{slice_prefix}_preds_metadata.json")
                if os.path.exists(npy_path) and os.path.exists(meta_path):
                    self.ilp_slice_preds[run_name] = load_ilp_preds(npy_path, meta_path)
                    print(f"  '{run_name}': loaded {slice_prefix} tensor {self.ilp_slice_preds[run_name].shape}")
                else:
                    self.ilp_slice_preds[run_name] = None
                    print(f"  Warning: no {predict_on} tensor for '{run_name}' "
                          f"(expected {npy_path}). Run build_ilp_preds_for_ensemble first.")

        print("Selecting models...")
        if model_selection_metric == "weighted_f1":
            self.model_weights: dict[str, dict[str, float]] = self._compute_model_weights()
            self.trusted_model: dict[str, str] = {}
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
                if isinstance(m, list):
                    for run_name in m:
                        counts[run_name] = counts.get(run_name, 0) + 1
                else:
                    counts[m] = counts.get(m, 0) + 1
            print("  Trusted model counts:", counts)

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
        if cls_id not in self.label_stats:
            print(f"Warning: no label stats for {cls_id}, assuming has_negatives=True")
        return self.label_stats.get(cls_id, {}).get("has_negatives", True)

    def _compute_bottom_ilp_f1(self, cls_id: str, run_name: str) -> Optional[float]:
        """
        Exact ensemble F1 for a leaf class: ensemble predicts positive iff
        ILP(cls) AND DL(parent) >= 0.5 for every label-set parent.
        Uses tensor ops on the full validation prediction matrices.
        """
        ilp_preds = self.ilp_val_preds.get(run_name)
        if ilp_preds is None or cls_id not in ilp_preds.columns:
            return None
        if cls_id not in self._gt.columns:
            print(f"  Warning: class {cls_id} not in ground truth, cannot compute bottom ILP F1")
            return None

        parents = [p for p in self.label_parents.get(cls_id, set())
                   if p in self.dl_val_preds.columns]
        common_mols = ilp_preds.index.intersection(self._gt.index)
        if len(common_mols) != len(self._gt.index):
            raise ValueError(f"ILP preds and GT have different molecule sets for class {cls_id} in run {run_name}: {len(ilp_preds.index)} != {len(self._gt.index)}")

        ilp_pred = ilp_preds.loc[common_mols, cls_id].astype(bool)
        if parents:
            dl_parents_pos = (self.dl_val_preds.loc[common_mols, parents] >= 0.5).all(axis=1)
            ensemble_pred = ilp_pred & dl_parents_pos
        else:
            ensemble_pred = ilp_pred

        gt = self._gt.loc[common_mols, cls_id]
        return self._f1_from_conf({
            "TP": int((ensemble_pred & gt).sum()),
            "FP": int((ensemble_pred & ~gt).sum()),
            "FN": int((~ensemble_pred & gt).sum()),
            "TN": int((~ensemble_pred & ~gt).sum()),
        })

    def _select_trusted_models(self) -> dict[str, str|list[str]]:
        trusted: dict[str, str|list[str]] = {}
        trusted_scores: dict[str, float] = {}
        label_children: dict[str, set] = {}
        if self.model_selection_metric.startswith("f1_bottom_ilp"):
            label_children = {cls: set() for cls in self.label_set}
            for cls, parents in self.label_parents.items():
                for p in parents:
                    label_children[p].add(cls)

        for cls_id in self.topo_order:
            dl_score = self._compute_score_from_conf(self.dl_val_scores.get(cls_id))
            best_model, best_score = "dl", dl_score if dl_score is not None else 0
            scores_log = {"dl": dl_score}

            # Only consider ILP for leaf classes in f1_bottom_ilp mode
            if not self.model_selection_metric.startswith("f1_bottom_ilp") or (
                cls_id in label_children and not label_children[cls_id]
            ):
                for run_name in self.ilp_val_scores:
                    if self.model_selection_metric.startswith("f1_bottom_ilp"):
                        ilp_score = self._compute_bottom_ilp_f1(cls_id, run_name)
                    else:
                        ilp_score = self._compute_score_from_conf(
                            self.ilp_val_scores[run_name].get(cls_id)
                        )

                    if ilp_score is None:
                        continue
                    scores_log[run_name] = ilp_score
                    # 3 options: take 1 ILP rule (if there is one that matches / beats DL), take all ILP rules that match the best score, take all ILP rules that beat DL
                    if self.model_selection_metric == "f1_bottom_ilp":
                        if ilp_score >= best_score and self.ilp_programs.get(run_name, {}).get(cls_id):
                            best_model, best_score = run_name, ilp_score
                    if self.model_selection_metric == "f1_bottom_ilp_best":
                        if ilp_score > best_score and self.ilp_programs.get(run_name, {}).get(cls_id):
                            best_model, best_score = [run_name], ilp_score
                        elif ilp_score == best_score and self.ilp_programs.get(run_name, {}).get(cls_id):
                            if best_model == "dl":
                                best_model = [run_name]
                            else:
                                best_model.append(run_name)
                    if self.model_selection_metric == "f1_bottom_ilp_multiple":
                        if ilp_score >= best_score and self.ilp_programs.get(run_name, {}).get(cls_id):
                            if best_model == "dl":
                                best_model = [run_name]
                            elif isinstance(best_model, list):
                                best_model.append(run_name)

            trusted[cls_id] = best_model
            trusted_scores[cls_id] = best_score if best_model != "dl" else dl_score or 0
        return trusted

    def _compute_model_weights(self) -> dict[str, dict[str, float]]:
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
            weights[cls_id] = {"dl": 1.0} if total == 0 else {n: f / total for n, f in model_f1s.items()}
        for cls in self.label_set:
            if cls not in weights:
                weights[cls] = {"dl": 1.0}
        return weights

    # ── ILP tensor generation ─────────────────────────────────────────────

    def slice_ilp_preds(
        self,
        mol_order: list[str],
        output_npy_path: str,
        output_meta_path: str,
    ) -> pd.DataFrame:
        """
        Assemble the ILP predictions tensor for mol_order by slicing columns
        from the already-loaded self.ilp_val_preds tensors.

        For each class where an ILP run is trusted, the prediction column from
        that run's validation tensor is reindexed to mol_order (molecules absent
        from the tensor are filled with False).  No Clingo evaluation is performed.
        """
        # Determine trusted class → run mapping
        if self.trusted_model:
            cls_to_run = {
                cls: models
                for cls, models in self.trusted_model.items()
                if all(model not in ("dl", "always_positive")
                and self.ilp_slice_preds.get(model) is not None for model in (models if isinstance(models, list) else [models]))
            }
        else:
            cls_to_run = {}
            for cls, weights in self.model_weights.items():
                for model_name, weight in weights.items():
                    if model_name not in ("dl", "always_positive") and weight > 0 \
                            and self.ilp_slice_preds.get(model_name) is not None:
                        cls_to_run[cls] = model_name
                        break

        ilp_class_ids = sorted(cls_to_run)
        cols = {}
        for cls_id in ilp_class_ids:
            run_names = cls_to_run[cls_id]
            if isinstance(run_names, str):
                cols[cls_id] = self.ilp_slice_preds.get(run_names)[cls_id].reindex(mol_order, fill_value=False).astype(bool)
                continue
            preds = np.stack([self.ilp_slice_preds.get(run_name)[cls_id].reindex(mol_order, fill_value=False).astype(bool) for run_name in run_names if self.ilp_slice_preds.get(run_name) is not None and cls_id in self.ilp_slice_preds.get(run_name).columns], axis=0)
            #if preds.shape[0] > 1:
            #    print(f"Combining ILP predictions for class {cls_id} from runs {run_names} with OR (avg sum: {preds.sum(axis=1).mean():.2f}, after aggregation: {np.any(preds, axis=0).sum()})")
            cols[cls_id] = np.mean(preds, axis=0) >= 0.5

        result = pd.DataFrame(cols, index=mol_order)
        arr = result.to_numpy().astype(bool)
        np.save(output_npy_path, arr)
        with open(output_meta_path, "w") as f:
            json.dump({"mol_order": mol_order, "class_labels": ilp_class_ids}, f)
        print(f"Saved ILP predictions tensor: {output_npy_path} "
              f"({len(mol_order)} × {len(ilp_class_ids)})")
        return result


# ---------------------------------------------------------------------------
# Ensemble aggregation: combine DL + ILP tensors into predictions
# ---------------------------------------------------------------------------

class EnsembleAggregator:
    """
    Aggregates pre-computed DL and ILP prediction tensors into hierarchical
    ensemble predictions using a classifier-chain approach.

    dl_preds:     float DataFrame (mol_id × class_id → score in [0, 1]).
    ilp_preds:    bool DataFrame (mol_id × class_id) built by
                  EnsembleConstructor.build_ilp_preds(); must cover every
                  molecule that will be passed to predict_sample / predict_set.
    trusted_model / model_weights: loaded from the CSV saved by
                  EnsembleConstructor (use one or the other, not both).
    """

    def __init__(
        self,
        dl_preds: pd.DataFrame,
        ilp_preds: pd.DataFrame,
        label_stats: dict[str, dict]|list[str],
        chebi_graph,
        trusted_model: Optional[dict[str, str|list[str]]] = None,
        model_weights: Optional[dict[str, dict[str, float]]] = None,
        classifier_chain_mode: bool = True,
    ):
        import networkx as nx

        self.dl_preds = dl_preds
        self.ilp_preds = ilp_preds
        self.label_set = set(str(l) for l in label_stats.keys()) if isinstance(label_stats, dict) else set(label_stats)
        self.classifier_chain_mode = classifier_chain_mode
        self.trusted_model = trusted_model or {}
        self.model_weights = model_weights or {}

        print("Building label-projected hierarchy...")
        self.label_parents = build_label_parents(self.label_set, chebi_graph)
        label_graph = nx.DiGraph()
        for cls in self.label_set:
            label_graph.add_node(cls)
            for parent in self.label_parents[cls]:
                label_graph.add_edge(parent, cls)
        self.topo_order = list(nx.topological_sort(label_graph))

        counts: dict[str, int] = {}
        if self.model_weights:
            for w in self.model_weights.values():
                for name in w:
                    counts[name] = counts.get(name, 0) + 1
            print("  Weighted model participation counts:", counts)
        else:
            for m in self.trusted_model.values():
                if isinstance(m, list):
                    for model in m:
                        counts[model] = counts.get(model, 0) + 1
                else:
                    counts[m] = counts.get(m, 0) + 1
            print("  Trusted model counts:", counts)


    # ── Prediction ────────────────────────────────────────────────────────

    def predict_sample(self, mol_id: str) -> set[str]:
        """Predict all positive classes for a single molecule."""
        positive: set[str] = set()
        for cls in self.topo_order:
            always_pos = (
                self.trusted_model.get(cls) == "always_positive"
                or self.model_weights.get(cls, {}) == {"always_positive": 1.0}
            )
            # native mode = DL model can predict classes that have not-predicted parents (ILP / always-positive still follow the chain)
            trusted_model_is_dl = self.trusted_model.get(cls) == "dl" or (isinstance(self.trusted_model.get(cls), list) and "dl" in self.trusted_model.get(cls))
            if (self.classifier_chain_mode or always_pos or not trusted_model_is_dl) and not all(
                p in positive for p in self.label_parents.get(cls, set())
            ):
                continue
            if self._eval_model_for_mol(cls, mol_id):
                positive.add(cls)
        return positive

    def _eval_model_for_mol(self, cls: str, mol_id: str) -> bool:
        if self.model_weights:
            weights = self.model_weights.get(cls, {"dl": 1.0})
            if weights == {"always_positive": 1.0}:
                return True
            return self._compute_weighted_prediction(cls, mol_id, weights)

        model = self.trusted_model.get(cls, "dl")
        if model == "always_positive":
            return True
        if model == "dl" or isinstance(model, list) and "dl" in model:
            if cls not in self.dl_preds.columns:
                raise ValueError(f"DL predictions for cls={cls} not found in dl_preds. dl_preds has {len(self.dl_preds.columns)} classes: {self.dl_preds.columns.tolist()[:10]}...")
            if mol_id not in self.dl_preds.index:
                raise ValueError(f"DL predictions for mol_id={mol_id} not found in dl_preds. dl_preds has {len(self.dl_preds.index)} molecules: {self.dl_preds.index.tolist()[:10]}...")
            return float(self.dl_preds.at[mol_id, cls]) >= 0.5
        # ILP model: look up in pre-computed tensor; False if not covered
        if cls not in self.ilp_preds.columns or mol_id not in self.ilp_preds.index:
            raise ValueError(f"ILP predictions for cls={cls} or mol_id={mol_id} not found in ilp_preds. ilp_preds has {len(self.ilp_preds.columns)} classes: {self.ilp_preds.columns.tolist()[:10]} and {len(self.ilp_preds.index)} molecules: {self.ilp_preds.index.tolist()[:10]}...")    
        return bool(self.ilp_preds.at[mol_id, cls])

    def _compute_weighted_prediction(self, cls: str, mol_id: str, weights: dict[str, float]) -> bool:
        total = 0.0
        for model_name, weight in weights.items():
            if weight <= 0:
                continue
            if model_name == "dl":
                if cls in self.dl_preds.columns and mol_id in self.dl_preds.index:
                    total += weight * float(self.dl_preds.at[mol_id, cls])
            else:
                if cls in self.ilp_preds.columns and mol_id in self.ilp_preds.index:
                    total += weight * (1.0 if bool(self.ilp_preds.at[mol_id, cls]) else 0.0)
        return total >= 0.5

    def predict_set(self, mol_ids: list[str]) -> pd.DataFrame:
        """Run ensemble predictions for a list of molecules.

        Returns a boolean DataFrame: rows = mol_ids, columns = labels (topo order).
        """
        import tqdm

        records = []
        for mol_id in tqdm.tqdm(mol_ids, desc="Ensemble predictions"):
            pos = self.predict_sample(mol_id)
            records.append({cls: cls in pos for cls in self.topo_order})

        return pd.DataFrame(records, index=mol_ids, columns=self.topo_order)


def ensemble_construct_demo():
    from chebi_utils import build_chebi_graph, extract_molecules, get_hierarchy_subgraph
    obo_path = os.path.join("data", "chebi_v248", "raw", "chebi.obo")
    sdf_path = os.path.join("data", "chebi_v248", "raw", "chebi.sdf.gz")
    chebi_graph = get_hierarchy_subgraph(build_chebi_graph(obo_path))

    dl_val_preds = load_dl_preds(
        npy_path=os.path.join("data", "preds", "val_preds_chebi25.npy"),
        meta_json_path=os.path.join("data", "preds", "val_preds_chebi25_metadata.json"),
    )
    print(f"Loaded DL val preds: {dl_val_preds.shape}")

    with open(os.path.join("data", "chebi_v248", "ChEBI25_3_STAR", "processed", "class_stats.csv")) as _f:
        _lines = [l.strip().split(",") for l in _f if l.strip()][1:]
        label_stats = {l[0]: {"has_negatives": l[3].lower() == "true"} for l in _lines}

    ilp_val_runs = {
        "server_20260511_163024_fp60": os.path.join("data", "results_val", "server_20260511_163024_fp60"),
        "server_20260511_162944_fn60": os.path.join("data", "results_val", "server_20260511_162944_fn60"),
        "server_20260511_153741_standard60": os.path.join("data", "results_val", "server_20260511_153741_standard60"),
    }

    constructor = EnsembleConstructor(
        ilp_val_runs=ilp_val_runs,
        dl_val_preds=dl_val_preds,
        label_stats=label_stats,
        chebi_graph=chebi_graph,
    )

    samples = dl_val_preds.index.tolist()
    print(f"Slicing ILP tensor for {len(samples)} validation molecules...")
    constructor.slice_ilp_preds(
        mol_order=samples,
        output_npy_path=os.path.join("data", "ensemble_predictions", "demo_ilp_preds.npy"),
        output_meta_path=os.path.join("data", "ensemble_predictions", "demo_ilp_preds_metadata.json"),
    )

if __name__ == "__main__":
    ensemble_construct_demo()