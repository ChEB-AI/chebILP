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


def build_ilp_preds_tensor(
    programs: dict[str, str],
    molecules_df,
    mol_order: list[str],
    output_npy_path: str,
    output_meta_path: str,
) -> pd.DataFrame:
    """
    Evaluate ILP programs against every molecule in mol_order and save the
    result as a boolean numpy tensor.

    Args:
        programs:         class_id → Prolog program string for each ILP-trusted
                          class (already filtered; excludes 'dl'/'always_positive').
        molecules_df:     DataFrame with 'mol' column; index = molecule IDs.
        mol_order:        ordered list of molecule IDs to evaluate.
        output_npy_path:  where to write the boolean .npy tensor.
        output_meta_path: where to write the JSON metadata file.

    All programs are submitted to Clingo in a single batched call: background
    knowledge is built once for all molecules, and all head predicates are
    evaluated simultaneously.

    Returns a boolean DataFrame (index=mol_order, columns=sorted class IDs).
    """
    import tqdm
    from chebILP.clingo_eval import evaluate_with_clingo
    from chebILP.mol2ilp import build_background_chemlog

    ilp_class_ids = sorted(programs.keys())
    n_mols = len(mol_order)
    n_classes = len(ilp_class_ids)
    print(f"Building ILP predictions tensor: {n_mols} molecules × {n_classes} classes")

    print("Building background facts for all molecules...")
    all_bg_facts: list[str] = []
    present_mol_ids: list[str] = []
    missing: int = 0
    for mol_id in tqdm.tqdm(mol_order, desc="Background knowledge"):
        mol_row = molecules_df[molecules_df.index == mol_id].dropna(subset=["mol"])
        if mol_row.empty:
            missing += 1
        else:
            lines, _ = build_background_chemlog(mol_row)
            all_bg_facts.extend(lines)
            present_mol_ids.append(mol_id)
    if missing:
        print(f"  {missing} molecules missing from molecules_df — predicted False")
    all_rules = [line for prog in programs.values() for line in prog.split("\n") if line.strip()]
    all_target_labels = [f"chebi_{cls_id}" for cls_id in ilp_class_ids]

    print("Running Clingo for all ILP classes...")
    try:
        positives = evaluate_with_clingo(all_rules, all_bg_facts, all_target_labels, present_mol_ids)
    except Exception as e:
        print(f"Clingo evaluation failed: {e}")
        positives = {}

    mol_idx = {mol_id: i for i, mol_id in enumerate(mol_order)}
    tensor = np.zeros((n_mols, n_classes), dtype=bool)
    for col_idx, cls_id in enumerate(ilp_class_ids):
        label = f"chebi_{cls_id}"
        for mol_id in positives.get(label, []):
            if mol_id in mol_idx:
                tensor[mol_idx[mol_id], col_idx] = True

    np.save(output_npy_path, tensor)
    with open(output_meta_path, "w") as f:
        json.dump({"mol_order": mol_order, "class_labels": ilp_class_ids}, f)
    print(f"Saved ILP predictions tensor to {output_npy_path} ({n_mols} × {n_classes})")

    return pd.DataFrame(tensor, index=mol_order, columns=ilp_class_ids)

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

    ilp_val_runs: directories produced by test_chebi_classes(test_on="validation").
    Each results.json entry must contain a 'program' and a 'validation_score'.
    Both validation scores (for model selection) and programs (for ILP tensor
    generation) are extracted from the same set of runs — no separate ilp_runs
    directory is needed.
    """

    def __init__(
        self,
        ilp_val_runs: dict[str, str],
        dl_val_scores: dict[str, dict[str, int]],
        label_stats: dict[str, dict]|list[str],
        chebi_graph: nx.DiGraph,
        model_selection_metric: Literal["balanced_acc", "f1", "weighted_f1", "f1_bottom_ilp"] = "f1",
    ):
        self.label_set = set(str(l) for l in label_stats.keys()) if isinstance(label_stats, dict) else set(label_stats)
        self.label_stats = label_stats
        self.dl_val_scores = dl_val_scores
        self.model_selection_metric = model_selection_metric
        self.chebi_graph = chebi_graph

        self.label_parents = build_label_parents(self.label_set, chebi_graph)
        label_graph = nx.DiGraph()
        for cls in self.label_set:
            label_graph.add_node(cls)
            for parent in self.label_parents[cls]:
                label_graph.add_edge(parent, cls)
        self.topo_order = list(nx.topological_sort(label_graph))

        self.ilp_programs: dict[str, dict[str, Optional[str]]] = {}
        self.ilp_val_scores: dict[str, dict[str, Optional[dict]]] = {}

        print("Loading ILP validation runs (scores + programs)...")
        for run_name, run_dir in ilp_val_runs.items():
            results = load_ilp_results(run_dir)
            self.ilp_programs[run_name] = {cid: e.get("program") for cid, e in results.items()}
            self.ilp_val_scores[run_name] = {
                cid: e.get("validation_score") for cid, e in results.items()
            }
            n_progs = sum(1 for p in self.ilp_programs[run_name].values() if p)
            print(f"  '{run_name}': {n_progs} programs, {len(self.ilp_val_scores[run_name])} val scores")

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

    def _select_trusted_models(self) -> dict[str, str]:
        trusted: dict[str, str] = {}
        trusted_scores: dict[str, float] = {}
        if self.model_selection_metric == "f1_bottom_ilp":
            label_parents = build_label_parents(self.label_set, self.chebi_graph)
            label_children = {cls: set() for cls in self.label_set}
            for cls, parents in label_parents.items():
                for p in parents:
                    label_children[p].add(cls)
            print(f"Got label children for {len(label_children)} classes, {len([c for c in label_children.values() if not c])} are leaves")
            
        for cls_id in self.topo_order:
            dl_score = self._compute_score_from_conf(self.dl_val_scores.get(cls_id))
            best_model, best_score = "dl", dl_score if dl_score is not None else 0

            # only consider ILP models for classes that are at the bottom of the label hierarchy
            if self.model_selection_metric != "f1_bottom_ilp" or (cls_id in label_children and len(label_children[cls_id]) == 0):
                #if not self._has_negatives(cls_id):
                #    trusted[cls_id] = "always_positive"
                #    trusted_scores[cls_id] = 1
                #    continue
                for run_name, run_scores in self.ilp_val_scores.items():
                    ilp_score = self._compute_score_from_conf(run_scores.get(cls_id))
                    if ilp_score is None:
                        continue
                    # ilp score is the relative to superclass - multiply with parent scores
                    parent_scores = [trusted_scores.get(p, 1.0) for p in self.label_parents.get(cls_id, [])]
                    for p_score in parent_scores:
                        ilp_score = ilp_score * p_score
                    if ilp_score >= best_score:
                        if self.ilp_programs.get(run_name, {}).get(cls_id):
                            best_model, best_score = run_name, ilp_score
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

    def build_ilp_preds(
        self,
        molecules_df,
        mol_order: list[str],
        output_npy_path: str,
        output_meta_path: str,
    ) -> pd.DataFrame:
        """
        Evaluate trusted ILP programs against every molecule in mol_order and
        save the result as a boolean numpy tensor.

        Programs come from self.ilp_programs as selected by self.trusted_model /
        self.model_weights. All programs are evaluated in a single Clingo call.
        """
        programs: dict[str, str] = {}
        if self.trusted_model:
            for cls_id, model in self.trusted_model.items():
                if model not in ("dl", "always_positive"):
                    prog = self.ilp_programs.get(model, {}).get(cls_id)
                    if prog:
                        programs[cls_id] = prog
        else:
            for cls_id, weights in self.model_weights.items():
                for model_name, weight in weights.items():
                    if model_name not in ("dl", "always_positive") and weight > 0:
                        prog = self.ilp_programs.get(model_name, {}).get(cls_id)
                        if prog:
                            programs[cls_id] = prog
                            break
        return build_ilp_preds_tensor(programs, molecules_df, mol_order, output_npy_path, output_meta_path)


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
        trusted_model: Optional[dict[str, str]] = None,
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
            if (self.classifier_chain_mode or always_pos or self.trusted_model.get(cls, "dl") != "dl") and not all(
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
        if model == "dl":
            if cls not in self.dl_preds.columns:
                raise ValueError(f"DL predictions for cls={cls} not found in dl_preds. dl_preds has {len(self.dl_preds.columns)} classes: {self.dl_preds.columns.tolist()[:10]}...")
            if mol_id not in self.dl_preds.index:
                raise ValueError(f"DL predictions for mol_id={mol_id} not found in dl_preds. dl_preds has {len(self.dl_preds.index)} molecules: {self.dl_preds.index.tolist()[:10]}...")
            return float(self.dl_preds.at[mol_id, cls]) >= 0.5
        # ILP model: look up in pre-computed tensor; False if not covered
        if cls not in self.ilp_preds.columns or mol_id not in self.ilp_preds.index:
            return False
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
    from chebILP.mol2ilp import get_direct_neighbors
    import networkx as nx
    obo_path = os.path.join("data", "chebi_v248", "raw", "chebi.obo")
    sdf_path = os.path.join("data", "chebi_v248", "raw", "chebi.sdf.gz")
    chebi_graph = get_hierarchy_subgraph(build_chebi_graph(obo_path))
    molecules_df = extract_molecules(sdf_path)
    hierarchy_graph = nx.transitive_closure_dag(chebi_graph)
    molecules_df = extract_molecules(sdf_path)
    molecules_df.index = molecules_df["chebi_id"].astype(str)
    molecules_df.index.name = None

    df_preds = load_dl_preds(
        npy_path=os.path.join("data", "preds", "val_preds_chebi25.npy"),
        meta_json_path=os.path.join("data", "preds", "val_preds_chebi25_metadata.json")
    )
    samples = df_preds.index.tolist()
    print(f"Loaded {len(samples)} samples")

    with open(os.path.join("data", "chebi_v248", "ChEBI25_3_STAR", "processed", "classes.txt"), "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    ilp_val_runs = {
        "server_20260511_163024_fp60": os.path.join("data", "results_val", "server_20260511_163024_fp60"),
        "server_20260511_162944_fn60": os.path.join("data", "results_val", "server_20260511_162944_fn60"),
        "server_20260511_153741_standard60": os.path.join("data", "results_val", "server_20260511_153741_standard60"),
    }
    dl_val_scores_csv = os.path.join("data", "preds", "dl_val_direct_neighbor_scores.csv")
    with open(dl_val_scores_csv) as _f:
        _lines = [l.strip().split(",") for l in _f if l.strip()][1:]
        dl_val_scores = {l[0]: {"TP": int(l[1] or 0), "FP": int(l[2] or 0), "TN": int(l[3] or 0), "FN": int(l[4] or 0)} for l in _lines}
    with open(os.path.join("data", "chebi_v248", "ChEBI25_3_STAR", "processed", "class_stats.csv")) as _f:
        _lines = [l.strip().split(",") for l in _f if l.strip()][1:]
        label_stats = {l[0]: {"has_negatives": l[3].lower() == "true"} for l in _lines}

    constructor = EnsembleConstructor(ilp_val_runs, dl_val_scores, label_stats, chebi_graph=chebi_graph)
    constructor.build_ilp_preds(
        molecules_df=molecules_df,
        mol_order=["145501"],
        output_npy_path=os.path.join("data", "ensemble_predictions", "DEMO145501_ilp_preds.npy"),
        output_meta_path=os.path.join("data", "ensemble_predictions", "DEMO145501_ilp_preds_metadata.json"),
    )

if __name__ == "__main__":
    ensemble_construct_demo()