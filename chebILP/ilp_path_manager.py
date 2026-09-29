import os
from typing import Literal

Split = Literal["train", "validation", "test"]


def get_problem_dir(chebi_id, split: Split, base_dir=None):
    if base_dir is None:
        base_dir = os.path.join("data", "ilp_problems")
    problem_dir = os.path.join(base_dir, f"chebi_{chebi_id}", split)
    os.makedirs(problem_dir, exist_ok=True)
    return problem_dir

def get_exs_path(chebi_id, split: Split, base_dir=None):
    problem_dir = get_problem_dir(chebi_id, split, base_dir)
    return os.path.join(problem_dir, "exs.pl")


def get_aux_programs_dir(base_dir):
    """Directory holding one canonical ``<aux_name>.py`` per unique predicate."""
    programs_dir = os.path.join(base_dir, "programs")
    os.makedirs(programs_dir, exist_ok=True)
    return programs_dir


def get_aux_class_map_path(base_dir):
    """JSON file mapping ``chebi_id -> [aux_name, ...]`` (predicates used by each class)."""
    return os.path.join(base_dir, "class_map.json")


def get_aux_hypotheses_path(base_dir):
    """JSON file mapping ``chebi_id -> {hypothesis, train_f1, ...}`` (the LLM's class rule)."""
    return os.path.join(base_dir, "hypotheses.json")


def get_aux_generation_log_path(chebi_id, base_dir):
    """Per-class LLM exchange log, now kept inside the library."""
    logs_dir = os.path.join(base_dir, "generation_logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, f"chebi_{chebi_id}.md")

def get_bk_path(chebi_id, split: Split, predicate_set, base_dir=None, selection_mode: Literal["claude", "random", "top_k"] | None = None, selection_k: int | None = None):
    problem_dir = get_problem_dir(chebi_id, split, base_dir)
    bk_dir = os.path.join(problem_dir, predicate_set)
    os.makedirs(bk_dir, exist_ok=True)
    if selection_mode is not None and selection_k is not None:
        bk_dir = os.path.join(bk_dir, f"selection_{selection_mode}_k={selection_k}")
        os.makedirs(bk_dir, exist_ok=True)
    return os.path.join(bk_dir, "bk.pl")

def get_bias_path(chebi_id, split: Split, base_dir=None, predicate_set="atoms", selection_mode: Literal["claude", "random", "top_k"] | None = None, selection_k: int | None = None, max_vars=None, max_body=None, max_clauses=None):
    problem_dir = get_problem_dir(chebi_id, split, base_dir)
    bk_dir = os.path.join(problem_dir, predicate_set)
    os.makedirs(bk_dir, exist_ok=True)
    if selection_mode is not None and selection_k is not None:
        bk_dir = os.path.join(bk_dir, f"selection_{selection_mode}_k={selection_k}")
    bias_file = "bias"
    if max_vars is not None:
        bias_file += f"_max_vars={max_vars}"
    if max_body is not None:
        bias_file += f"_max_body={max_body}"
    if max_clauses is not None:
        bias_file += f"_max_clauses={max_clauses}"
    bias_file += ".pl"
    return os.path.join(bk_dir, bias_file)


def get_aleph_stem(chebi_id, predicate_set, base_dir=None, selection_mode: Literal["claude", "random", "top_k"] | None = None, selection_k: int | None = None):
    """Aleph file stem for a class (train split only): callers append ``.b``/``.f``/``.n``.

    The stem basename is the bare ChEBI id, so files are e.g.
    ``.../train/atoms/13248.b``."""
    problem_dir = get_problem_dir(chebi_id, "train", base_dir)
    bk_dir = os.path.join(problem_dir, predicate_set)
    os.makedirs(bk_dir, exist_ok=True)
    if selection_mode is not None and selection_k is not None:
        bk_dir = os.path.join(bk_dir, f"selection_{selection_mode}_k={selection_k}")
        os.makedirs(bk_dir, exist_ok=True)
    return os.path.join(bk_dir, str(chebi_id))