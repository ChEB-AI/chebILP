"""
Hand-written ASP definitions of SCOPe folds in the SSE vocabulary, scored against the built problems.

Each program turns the fold's SCOPe description into the ``bk.pl`` facts and derives
``scop_<cl>_<cf>(D)``. They check that the encoding can express a fold at all, and give a
ceiling to compare learned programs with.

Example usage:
    python -m chebILP.protein_processing.reference_rules --problem_dir data/scope_208/ilp_problems
"""

import argparse
import os

from chebILP.reaction_processing.reference_rules import derive, read_examples, score
from chebILP.utils import add_class_arguments, resolve_classes

_COMMON = """
before(A,B) :- next_sse(A,B).
before(A,C) :- before(A,B), next_sse(B,C).
strand_between(A,B) :- strand(A), strand(B), before(A,X), before(X,B), strand(X).
next_strand(A,B) :- strand(A), strand(B), before(A,B), not strand_between(A,B).
n_strands(D,N) :- has_sse(D,_), N = #count{ S : has_sse(D,S), strand(S) }.
n_helices(D,N) :- has_sse(D,_), N = #count{ S : has_sse(D,S), helix(S) }.
"""

REFERENCE_PROGRAMS = {
    # DNA/RNA-binding 3-helical bundle — core: 3 helices; bundle, up-and-down.
    "scop_a_4": _COMMON
    + """
three_helix(H1) :- helix(H1), next_sse(H1,H2), helix(H2), next_sse(H2,H3), helix(H3), contact(H1,H3).
scop_a_4(D) :- has_sse(D,H), three_helix(H), n_helices(D,N), N <= 5, n_strands(D,M), M <= 3.
""",
    # Immunoglobulin-like beta-sandwich — 7 strands in 2 sheets; greek-key.
    "scop_b_1": _COMMON
    + """
big_sheet(S) :- strand(S), 2 <= #count{ T : same_sheet(S,T) }.
scop_b_1(D) :- has_sse(D,S1), has_sse(D,S2), big_sheet(S1), big_sheet(S2), S1 != S2, not same_sheet(S1,S2),
               contact(S1,S2), antiparallel(S1,_), antiparallel(S2,_),
               n_strands(D,N), 6 <= N, N <= 11, n_helices(D,K), K <= 2.
""",
    # TIM beta/alpha-barrel — parallel beta-sheet barrel, closed; n=8. DSSP often misses one
    # ladder of the barrel, so closure is a parallel pair spanning 5+ strands, not "no edge strand".
    "scop_c_1": _COMMON
    + """
par(S) :- parallel(S,_).
closure(S) :- parallel(S,T), before(S,T), 5 <= #count{ X : strand(X), before(S,X), before(X,T) }.
scop_c_1(D) :- has_sse(D,S), closure(S), 7 <= #count{ T : same_sheet(S,T), par(T) }.
""",
    # Flavodoxin-like — 3 layers a/b/a; parallel beta-sheet of 5 strands, order 21345.
    "scop_c_23": _COMMON
    + """
scop_c_23(D) :- has_sse(D,S1), next_strand(S1,S2), next_strand(S2,S3), next_strand(S3,S4), next_strand(S4,S5),
                parallel(S1,S2), parallel(S1,S3), parallel(S3,S4), parallel(S4,S5).
""",
    # Ferredoxin-like — alpha+beta sandwich with antiparallel beta-sheet; (beta-alpha-beta)x2, order 4132.
    "scop_d_58": _COMMON
    + """
bab(S1,S2) :- next_strand(S1,S2), before(S1,H), before(H,S2), helix(H).
scop_d_58(D) :- has_sse(D,S1), bab(S1,S2), next_strand(S2,S3), bab(S3,S4),
                antiparallel(S1,S3), antiparallel(S2,S3), antiparallel(S1,S4).
""",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--problem_dir", default=os.path.join("data", "scope_208", "ilp_problems")
    )
    add_class_arguments(parser, required=False)
    parser.add_argument("--predicate_set", default="atoms")
    parser.add_argument(
        "--show_errors", type=int, default=0, help="List this many FPs/FNs per split"
    )
    args = parser.parse_args()

    classes = (
        resolve_classes(args)
        if (args.classes or args.labels_file)
        else list(REFERENCE_PROGRAMS)
    )
    for cls in classes:
        for split in ("train", "validation", "test"):
            split_dir = os.path.join(args.problem_dir, cls, split)
            pos, neg = read_examples(os.path.join(split_dir, "exs.pl"))
            derived = derive(
                REFERENCE_PROGRAMS[cls],
                os.path.join(split_dir, args.predicate_set, "bk.pl"),
                cls,
            )
            s = score(pos, neg, derived)
            print(
                f"{cls:10s} {split:10s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
                f"(tp={s['tp']} fp={s['fp']} fn={s['fn']} tn={s['tn']})"
            )
            if args.show_errors:
                print(f"    FP: {sorted(neg & derived)[: args.show_errors]}")
                print(f"    FN: {sorted(pos - derived)[: args.show_errors]}")


if __name__ == "__main__":
    main()
