import os
from typing import Literal

def get_problem_dir(chebi_id, split:Literal["train", "validation", "test"], base_dir=None):
    if base_dir is None:
        base_dir = os.path.join("ilp", "chebi_v248")
    problem_dir = os.path.join(base_dir, f"chebi_{chebi_id}", split)
    os.makedirs(problem_dir, exist_ok=True)
    return problem_dir

def get_exs_path(chebi_id, split:Literal["train", "validation", "test"], base_dir=None):
    problem_dir = get_problem_dir(chebi_id, split, base_dir)
    filename = "exs.pl"
    return os.path.join(problem_dir, filename)

def get_bk_path(chebi_id, split:Literal["train", "validation", "test"], predicate_set, base_dir=None, selection_mode:Literal["claude", "random", "top_k"]|None=None, selection_k:int|None=None):
    problem_dir = get_problem_dir(chebi_id, split, base_dir)
    bk_dir = os.path.join(problem_dir, predicate_set)
    os.makedirs(bk_dir, exist_ok=True)
    if selection_mode:
        bk_dir = os.path.join(bk_dir, f"selection_{selection_mode}_k={selection_k}")
        os.makedirs(bk_dir, exist_ok=True)
    filename = "bk.pl"
    return os.path.join(bk_dir, filename)

def get_bias_path(chebi_id, split:Literal["train", "validation", "test"], base_dir=None, predicate_set="atoms", selection_mode:Literal["claude", "random", "top_k"]|None=None, selection_k:int|None=None, max_vars=None, max_body=None, max_clauses=None):
    problem_dir = get_problem_dir(chebi_id, split, base_dir)
    bk_dir = os.path.join(problem_dir, predicate_set)
    os.makedirs(bk_dir, exist_ok=True)
    if selection_mode:
        bk_dir = os.path.join(bk_dir, f"selection_{selection_mode}_k={selection_k}")
    bias_file = f"bias"
    if max_vars is not None:
        bias_file += f"_max_vars={max_vars}"
    if max_body is not None:
        bias_file += f"_max_body={max_body}"
    if max_clauses is not None:
        bias_file += f"_max_clauses={max_clauses}"
    bias_file += ".pl"
    return os.path.join(bk_dir, bias_file)