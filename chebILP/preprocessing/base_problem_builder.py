"""
Turns a dataset into ILP problems: one per target class, with train/validation/test splits.

Everything dataset-specific (which examples are positive/negative, how a capped train split is
sampled, the BK encoding, computed and seeded facts) comes from the ``BaseDataset``. With an
``aux_library_dir``, each class's LLM rules from that library are ground over the BK and their
extensions added as facts.

Layout: ``<problem_dir>/<class_head>/<split>/exs.pl`` and ``.../<split>/<bk_dirname>/bk.pl``
(plus ``bias.pl``, and Aleph files for train), see ``ilp_path_manager``.
"""

import os
from typing import Optional

import pandas as pd
import tqdm

from chebILP.ilp_path_manager import (
    Split,
    class_head,
    get_aleph_stem,
    get_bk_dir,
    get_bk_path,
    get_exs_path,
)
from chebILP.ilp_problem import ILPProblem
from chebILP.prolog_bk import QlfCompiler, write_aux
from chebILP.predicate_generation.auxiliary_rules import (
    SHARED_OWNER,
    derive_rule_extensions,
    load_class_rules,
    referenced_fact_predicates,
    resolve_rule_dependencies,
)
from chebILP.preprocessing.base_dataset import BaseDataset
from chebILP.preprocessing.chebi.fg_matching import FG_SEED_PREFIXES, is_fg_seed_name

SPLITS: tuple[Split, ...] = ("train", "validation", "test")


def read_examples(
    class_id, problem_dir: str, split: Split
) -> tuple[list[str], list[str]]:
    """Positive and negative example ids of a class's ``exs.pl``, in file order.

    An id is the head's argument list without spaces: ``pos(h(e3, x, 1)).`` gives ``e3,x,1``.
    """
    pos, neg = [], []
    with open(get_exs_path(class_id, split, base_dir=problem_dir), "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            atom = line[len("pos(") :].rstrip(".").removesuffix(")")
            args = atom[atom.index("(") + 1 : -1]
            example = ",".join(a.strip() for a in args.split(","))
            (pos if line.startswith("pos(") else neg).append(example)
    return pos, neg


def _facts_by_predicate(lines) -> dict[str, list[tuple | str]]:
    """Ground fact lines (``p(a,b).``) grouped by predicate, as ``ILPProblem.bk_facts``."""
    facts = {}
    for line in lines:
        pred, args = line.rstrip(".").rstrip(")").split("(", 1)
        args = tuple(args.split(","))
        facts.setdefault(pred, []).append(args if len(args) > 1 else args[0])
    return facts


def build_split_backgrounds(
    dataset: BaseDataset,
    rows_by_split: dict[str, pd.DataFrame],
    programs=None,
    dependencies=None,
    aux_library_dir: Optional[str] = None,
    computed_facts: bool = True,
) -> tuple[dict[str, list[str]], bool, dict[str, int]]:
    """BK fact lines per split, with the extensions of ``programs`` if given.

    The rule programs are ground once over all splits, with the dataset's computed facts and the
    seeded FG facts they reference (both used for grounding only), and each derived fact goes to
    the split of its example, or to every split if it mentions none. Seeded FG predicates among ``programs`` are emitted from their RDKit
    match. ``dependencies`` defaults to the library programs the rules build on; they are ground
    but emit nothing. Returns ``(lines_by_split, grounding_failed, n_base)``: each split's lines start
    with its ``n_base[split]`` base-encoding facts, followed by the library facts.
    """
    lines_by_split = {
        split: list(dataset.build_background(rows)[0])
        for split, rows in rows_by_split.items()
    }
    n_base = {split: len(lines) for split, lines in lines_by_split.items()}
    if not programs:
        return lines_by_split, False, n_base

    seed_names = {p.name for p in programs if is_fg_seed_name(p.name)}
    asp_programs = [p for p in programs if not is_fg_seed_name(p.name)]
    if dependencies is None:
        dependencies = resolve_rule_dependencies(asp_programs, aux_library_dir)

    failed = False
    if asp_programs:
        all_rows = pd.concat(list(rows_by_split.values()))
        all_ids = [str(i) for i in all_rows.index]
        eval_facts = [line for lines in lines_by_split.values() for line in lines]
        if computed_facts:
            eval_facts += dataset.build_computed_facts(all_rows)
        referenced = referenced_fact_predicates(
            asp_programs + dependencies, FG_SEED_PREFIXES
        )
        if referenced:
            eval_facts += dataset.seed_fact_lines(referenced, all_rows)
        try:
            extensions = derive_rule_extensions(
                asp_programs + dependencies, eval_facts, all_ids
            )
        except (RuntimeError, MemoryError) as e:
            tqdm.tqdm.write(
                f"  Grounding failed ({e}); continuing without the auxiliary extensions of "
                f"{', '.join(p.name for p in asp_programs)}"
            )
            failed, extensions = True, {}
        split_of = {
            str(i): split for split, rows in rows_by_split.items() for i in rows.index
        }
        for prog in asp_programs:
            emitted = set()
            for example, arg_tuples in extensions.get(prog.name, {}).items():
                if example is SHARED_OWNER:
                    splits = list(rows_by_split)
                elif str(example) in split_of:
                    splits = [split_of[str(example)]]
                else:
                    continue
                for split in splits:
                    for args in arg_tuples:
                        line = f"{prog.name}({','.join(args)})."
                        if (split, line) not in emitted:
                            emitted.add((split, line))
                            lines_by_split[split].append(line)

    if seed_names:
        for split, rows in rows_by_split.items():
            lines_by_split[split] += dataset.seed_fact_lines(seed_names, rows)
    return lines_by_split, failed, n_base


def background_with_rules(
    dataset: BaseDataset,
    rows: pd.DataFrame,
    programs=None,
    dependencies=None,
    aux_library_dir: Optional[str] = None,
    computed_facts: bool = True,
) -> list[str]:
    """One flat BK fact list for ``rows``, exactly as :meth:`ILPProblemBuilder.build_bk` builds it."""
    lines_by_split, _, _ = build_split_backgrounds(
        dataset,
        {"all": rows},
        programs,
        dependencies,
        aux_library_dir=aux_library_dir,
        computed_facts=computed_facts,
    )
    return lines_by_split["all"]


class ILPProblemBuilder:
    """Turns a dataset into ILP problems."""

    def __init__(
        self,
        dataset: BaseDataset,
        problem_dir: Optional[str] = None,
        aux_library_dir: Optional[str] = None,
        computed_facts: bool = True,
        write_aleph: bool = True,
    ):
        self.dataset = dataset
        self.problem_dir = problem_dir or os.path.join(dataset.data_dir, "ilp_problems")
        os.makedirs(self.problem_dir, exist_ok=True)
        self.aux_library_dir = aux_library_dir
        # Computed facts (e.g. molecular weight) feed rule grounding only, never bk.pl.
        self.computed_facts = computed_facts
        self.write_aleph = write_aleph

        splits = self.dataset.load_splits_from_csv()
        self.split_ids = {
            split: set(splits.loc[splits["split"] == split, "id"].astype(str))
            for split in SPLITS
        }

    def build_examples(
        self,
        target_ids: list[str],
        max_pos_train: int = 200,
        max_neg_train: int = 200,
    ):
        """Write ``exs.pl`` per split for the given targets.

        The examples come from the dataset's ``split_examples``: by default positives are the
        examples at or below the target, negatives its direct neighbours in the hierarchy. Over a
        cap, a split keeps the examples the dataset picks. Validation and test are capped at the
        dataset's ``eval_cap`` (uncapped unless it sets one).
        """
        eval_cap = self.dataset.eval_cap
        caps = {"train": (max_pos_train, max_neg_train)}
        caps |= {split: (eval_cap, eval_cap) for split in SPLITS if split != "train"}
        counts = {}
        for target_id in tqdm.tqdm(target_ids, desc="Building ILP examples"):
            examples = self.dataset.split_examples(target_id, self.split_ids, caps)
            n_pos = sum(len(pos) for pos, _ in examples.values())
            n_neg = sum(len(neg) for _, neg in examples.values())

            head = class_head(target_id)
            for split in SPLITS:
                pos, neg = examples[split]
                with open(
                    get_exs_path(target_id, split, base_dir=self.problem_dir), "w"
                ) as f:
                    f.writelines(f"pos({head}({i})).\n" for i in pos)
                    f.writelines(f"neg({head}({i})).\n" for i in neg)
            counts[target_id] = (n_pos, n_neg)

        if counts:
            min_pos_id = min(counts, key=lambda c: counts[c][0])
            min_neg_id = min(counts, key=lambda c: counts[c][1])
            print(
                f"Fewest positives: {self.dataset.class_label(min_pos_id)} with "
                f"{counts[min_pos_id][0]} across all splits"
            )
            print(
                f"Fewest negatives: {self.dataset.class_label(min_neg_id)} with "
                f"{counts[min_neg_id][1]} across all splits"
            )

    def build_bk(self, target_ids: list[str], compile_workers: int = 4):
        """Write ``bk.pl``/``bias.pl`` per split (and Aleph files for train) for the given targets, and
        the train BK for Prolog (``chebILP.prolog_bk``): ``bk.qlf`` for atoms, compiled by
        ``compile_workers`` swipl processes alongside, and with a library its ``bk_aux.pl``. The library
        BK needs the class's atoms BK to be built already."""
        if self.aux_library_dir:
            missing = [
                t
                for t in target_ids
                if not os.path.exists(
                    get_bk_path(t, "train", base_dir=self.problem_dir)
                )
            ]
            if missing:
                raise FileNotFoundError(
                    f"Build the atoms BK first: no train atoms bk.pl for {len(missing)} class(es) "
                    f"({', '.join(map(str, missing[:10]))})."
                )
        failed_classes = []
        compiler = QlfCompiler(compile_workers)
        pbar = tqdm.tqdm(target_ids, desc="Building background knowledge")
        for target_id in pbar:
            pbar.set_description(
                f"Building background knowledge for {self.dataset.class_label(target_id)}"
            )
            examples = {
                split: read_examples(target_id, self.problem_dir, split)
                for split in SPLITS
            }
            rows_by_split = {
                split: self.dataset.molecules.loc[self.dataset.example_rows(pos + neg)]
                for split, (pos, neg) in examples.items()
            }
            programs = (
                load_class_rules(target_id, library_dir=self.aux_library_dir)
                if self.aux_library_dir
                else None
            )
            lines_by_split, failed, n_base = build_split_backgrounds(
                self.dataset,
                rows_by_split,
                programs,
                aux_library_dir=self.aux_library_dir,
                computed_facts=self.computed_facts,
            )
            if failed:
                failed_classes.append(target_id)

            for split, (pos, neg) in examples.items():
                problem = ILPProblem(
                    (class_head(target_id), self.dataset.head_arity(target_id)),
                    pos,
                    neg,
                    _facts_by_predicate(lines_by_split[split]),
                )
                problem.write_popper(
                    get_bk_dir(
                        target_id,
                        split,
                        base_dir=self.problem_dir,
                        aux_library_dir=self.aux_library_dir,
                    ),
                    exs_folder=os.path.dirname(
                        get_exs_path(target_id, split, base_dir=self.problem_dir)
                    ),
                )
                if split == "train" and self.write_aleph:
                    stem = get_aleph_stem(
                        target_id,
                        base_dir=self.problem_dir,
                        aux_library_dir=self.aux_library_dir,
                    )
                    problem.write_aleph(
                        os.path.dirname(stem),
                        stem=os.path.basename(stem),
                        with_bk=False,
                    )
            if self.aux_library_dir:
                train_lines = lines_by_split["train"]
                write_aux(
                    target_id,
                    self.problem_dir,
                    self.aux_library_dir,
                    train_lines[n_base["train"] :],
                )
                compiler.submit(target_id, self.problem_dir)
            else:
                compiler.submit(target_id, self.problem_dir, force=True)

        failed_compiles = compiler.finish()
        if failed_classes:
            print(
                f"\n{len(failed_classes)} class(es) built without their auxiliary rule "
                f"extensions because grounding failed: {', '.join(failed_classes)}"
            )
        if failed_compiles:
            raise RuntimeError(
                f"qcompile failed for {len(failed_compiles)} class(es); learn cannot run them "
                f"until compile_bk succeeds: {failed_compiles[:3]}"
            )
