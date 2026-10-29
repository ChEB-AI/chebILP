"""
ILP problems for SCOPe folds, built by the ChEBI problem builder over the fold graph.

Example usage:
    python -m chebILP.protein_processing.scope_problem_builder --classes scop_a_1 scop_b_1 scop_c_1

Writes ``<problem_dir>/scop_<cl>_<cf>/<split>/{exs.pl,<predicate_set>/bk.pl,...}``. Positives are
the domains of the fold, negatives those of the other folds in its class. Only the ``atoms``
predicate set (here: the SSE facts) and the LLM rule sets apply.
"""

import argparse
import os

import pandas as pd

from chebILP.ilp_problem_builder import ILPProblemBuilder
from chebILP.protein_processing.protein_to_fol import (
    build_background_protein,
    build_computed_facts_protein,
)
from chebILP.protein_processing.scope_data import ScopeDataset
from chebILP.utils import add_class_arguments, resolve_classes

DEFAULT_PROBLEM_DIR = os.path.join("data", "scope_208", "ilp_problems")


class ScopeILPProblemBuilder(ILPProblemBuilder):
    """``ILPProblemBuilder`` whose examples are protein domains, sized by their number of SSEs."""

    def __init__(
        self,
        scope_version: str = "2.08",
        base_dir: str = "data",
        min_pos_samples: int = 25,
        problem_dir: str | None = None,
        **kwargs,
    ):
        dataset = ScopeDataset(
            scope_version=scope_version,
            base_dir=base_dir,
            min_pos_samples=min_pos_samples,
        )
        super().__init__(
            chebi_version=None,
            base_dir=base_dir,
            min_pos_samples=min_pos_samples,
            problem_dir=problem_dir or DEFAULT_PROBLEM_DIR,
            dataset=dataset,
            **kwargs,
        )

    def _example_sizes(self) -> pd.Series:
        return self.dataset.molecules["mol"].map(lambda s: len(s["sses"]))

    def _take_smallest(self, df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
        """A seeded random ``max_samples`` of ``df``: fold correlates with domain size, so the smallest would bias train."""
        return (
            df
            if len(df) <= max_samples
            else df.sample(max_samples, random_state=0).sort_index()
        )

    def _atom_background(self, rows, **kwargs):
        return build_background_protein(rows)

    def _computed_facts(self, rows) -> list[str]:
        return build_computed_facts_protein(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_class_arguments(parser)
    parser.add_argument("--scope_version", default="2.08")
    parser.add_argument("--base_dir", default="data")
    parser.add_argument("--problem_dir", default=None)
    parser.add_argument("--min_pos_samples", type=int, default=25)
    parser.add_argument("--max_pos_samples", type=int, default=200)
    parser.add_argument("--max_neg_samples", type=int, default=200)
    parser.add_argument("--predicate_set", default="atoms")
    parser.add_argument("--predicate_dir", default=None)
    args = parser.parse_args()

    builder = ScopeILPProblemBuilder(
        args.scope_version,
        args.base_dir,
        args.min_pos_samples,
        args.problem_dir,
        predicate_set=args.predicate_set,
        aux_library_dir=args.predicate_dir,
    )
    classes = resolve_classes(args)
    builder.build_examples(
        classes,
        min_pos_samples=args.min_pos_samples,
        max_pos_samples=args.max_pos_samples,
        max_neg_samples=args.max_neg_samples,
    )
    builder.build_bk(classes)


if __name__ == "__main__":
    main()
