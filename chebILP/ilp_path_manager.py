import os
from typing import Literal

Split = Literal["train", "validation", "test"]


# Class ids that are already valid head names: EC classes (``ec_3_1_1``), SCOPe folds (``scop_c_1``),
# GDL targets (``gdl_tictactoe_goal``).
PASSTHROUGH_PREFIXES = ("ec_", "scop_", "gdl_")


def class_head(class_id) -> str:
    """Target predicate and problem folder name: ``chebi_<id>``, or the id itself for an EC class, SCOPe node or GDL target."""
    class_id = str(class_id)
    return (
        class_id if class_id.startswith(PASSTHROUGH_PREFIXES) else f"chebi_{class_id}"
    )


def class_id_from_head(head: str) -> str | None:
    """Inverse of :func:`class_head`; ``None`` if ``head`` names no class target."""
    if head.startswith("chebi_"):
        return head[len("chebi_") :]
    return head if head.startswith(PASSTHROUGH_PREFIXES) else None


def get_problem_dir(chebi_id, split: Split, base_dir=None):
    if base_dir is None:
        base_dir = os.path.join("data", "ilp_problems")
    problem_dir = os.path.join(base_dir, class_head(chebi_id), split)
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
    return os.path.join(logs_dir, f"{class_head(chebi_id)}.md")


def bk_dirname(aux_library_dir=None) -> str:
    """BK folder name: ``atoms`` for the base encoding, ``llm_generated_rules_<library>`` with a rule library."""
    if aux_library_dir and aux_library_dir != "None":
        return (
            f"llm_generated_rules_{os.path.basename(os.path.normpath(aux_library_dir))}"
        )
    return "atoms"


def get_bk_dir(class_id, split: Split, base_dir=None, aux_library_dir=None):
    bk_dir = os.path.join(
        get_problem_dir(class_id, split, base_dir), bk_dirname(aux_library_dir)
    )
    os.makedirs(bk_dir, exist_ok=True)
    return bk_dir


def get_bk_path(class_id, split: Split, base_dir=None, aux_library_dir=None):
    return os.path.join(get_bk_dir(class_id, split, base_dir, aux_library_dir), "bk.pl")


def get_bias_path(
    class_id,
    split: Split,
    base_dir=None,
    aux_library_dir=None,
    max_vars=None,
    max_body=None,
):
    bias_file = "bias"
    if max_vars is not None:
        bias_file += f"_max_vars={max_vars}"
    if max_body is not None:
        bias_file += f"_max_body={max_body}"
    bias_file += ".pl"
    return os.path.join(
        get_bk_dir(class_id, split, base_dir, aux_library_dir), bias_file
    )


def get_aleph_stem(class_id, base_dir=None, aux_library_dir=None):
    """Aleph file stem for a class (train split only): callers append ``.b``/``.f``/``.n``.

    The stem basename is the bare class id, so files are e.g. ``.../train/atoms/13248.b``."""
    return os.path.join(
        get_bk_dir(class_id, "train", base_dir, aux_library_dir), str(class_id)
    )
