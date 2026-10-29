"""
ILP problems for Rhea EC classes, built by the ChEBI problem builder over the EC graph.

Example usage:
    python -m chebILP.reaction_processing.rhea_problem_builder --classes ec_3_1_1 ec_2_7_1 ec_4_1_1

Writes ``<problem_dir>/ec_<a>_<b>_<c>/<split>/{exs.pl,<predicate_set>/bk.pl,...}``. Positives are
the reactions under the class, negatives those under its sibling sub-subclasses.
"""

import argparse
import os

from chebILP.ilp_problem_builder import ILPProblemBuilder
from chebILP.preprocessing.reaction_processing.reaction_to_fol import (
    build_background_reaction,
    build_computed_facts_reaction,
)
from chebILP.preprocessing.reaction_processing.rhea_data import RheaDataset
from chebILP.utils import add_class_arguments, resolve_classes

DEFAULT_PROBLEM_DIR = os.path.join("data", "rhea_v142", "ilp_problems")


class RheaILPProblemBuilder(ILPProblemBuilder):
    """``ILPProblemBuilder`` whose examples are reactions; ``_atom_counts`` sums over all participants."""

    def __init__(
        self,
        rhea_version: int = 142,
        base_dir: str = "data",
        min_pos_samples: int = 25,
        problem_dir: str | None = None,
        **kwargs,
    ):
        dataset = RheaDataset(
            rhea_version=rhea_version,
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

    def _atom_background(self, rows, **kwargs):
        return build_background_reaction(rows, **kwargs)

    def _computed_facts(self, rows) -> list[str]:
        return build_computed_facts_reaction(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_class_arguments(parser)
    parser.add_argument("--rhea_version", type=int, default=142)
    parser.add_argument("--base_dir", default="data")
    parser.add_argument("--problem_dir", default=None)
    parser.add_argument("--min_pos_samples", type=int, default=25)
    parser.add_argument("--max_pos_samples", type=int, default=200)
    parser.add_argument("--max_neg_samples", type=int, default=200)
    parser.add_argument("--predicate_set", default="atoms")
    parser.add_argument("--predicate_dir", default=None)
    args = parser.parse_args()

    builder = RheaILPProblemBuilder(
        args.rhea_version,
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
