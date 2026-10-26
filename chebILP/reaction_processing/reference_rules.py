"""
Hand-written ASP definitions of EC classes in the reaction vocabulary, scored against the built problems.

They check that the encoding can express a class at all, and give a ceiling to compare learned
programs with. Each program derives ``ec_<a>_<b>_<c>(R)`` from the atom-level ``bk.pl``.

Example usage:
    python -m chebILP.reaction_processing.reference_rules --problem_dir data/rhea_v142/ilp_problems
"""

import argparse
import os

import clingo

from chebILP.utils import add_class_arguments, resolve_classes

_COMMON = """
bonded(A) :- has_bond_to(A,_).
degree(A,N) :- has_atom(_,A), N = #count{ B : has_bond_to(A,B) }.
carbonyl_c(C) :- c(C), bDOUBLE(C,O), o(O).
water(M) :- has_atom(M,O), o(O), not bonded(O), has_2_hs(O).
"""

REFERENCE_PROGRAMS = {
    # Carboxylic ester + H2O -> carboxylate + alcohol.
    "ec_3_1_1": _COMMON + """
carboxylic_ester(M) :- has_atom(M,C), carbonyl_c(C), bSINGLE(C,O), o(O), bSINGLE(O,C2), c(C2), C2 != C.
carboxylic_acid(M) :- has_atom(M,C), carbonyl_c(C), bSINGLE(C,O), o(O), degree(O,1).
ec_3_1_1(R) :- has_reactant(R,W), water(W), has_reactant(R,E), carboxylic_ester(E),
               has_product(R,P), carboxylic_acid(P).
""",
    # A phosphate monoester on a non-carbonyl carbon is formed: more of them on the right than on the left.
    "ec_2_7_1": _COMMON + """
terminal_o(O) :- o(O), degree(O,1).
monoester_p(P) :- p(P), 3 = #count{ O : has_bond_to(P,O), terminal_o(O) }.
alkyl_phosphate(M,P) :- has_atom(M,P), monoester_p(P), bSINGLE(P,O), o(O), bSINGLE(O,C), c(C), not carbonyl_c(C).
n_left(R,N) :- has_reactant(R,_), N = #count{ M,P : has_reactant(R,M), alkyl_phosphate(M,P) }.
n_right(R,N) :- has_product(R,_), N = #count{ M,P : has_product(R,M), alkyl_phosphate(M,P) }.
ec_2_7_1(R) :- n_left(R,L), n_right(R,N), N > L.
""",
    # CO2 or hydrogencarbonate is released.
    "ec_4_1_1": _COMMON + """
co2(M) :- has_atom(M,C), c(C), 2 = #count{ O : bDOUBLE(C,O), o(O) }, degree(C,2).
co2(M) :- has_atom(M,C), c(C), 3 = #count{ O : has_bond_to(C,O), o(O) }, 4 = #count{ A : has_atom(M,A) }.
ec_4_1_1(R) :- has_product(R,M), co2(M).
""",
}


def read_examples(exs_path: str) -> tuple[set[str], set[str]]:
    pos, neg = set(), set()
    with open(exs_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(("pos(", "neg(")):
                (pos if line.startswith("pos(") else neg).add(line.split("(")[-1].split(")")[0])
    return pos, neg


def derive(program: str, bk_path: str, head: str) -> set[str]:
    ctl = clingo.Control(["--warn=none"])
    ctl.load(bk_path)
    ctl.add("base", [], program + f"\n#show {head}/1.")
    ctl.ground([("base", [])])
    derived = set()
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            derived = {str(sym.arguments[0]) for sym in model.symbols(shown=True)}
    return derived


def score(pos: set[str], neg: set[str], derived: set[str]) -> dict:
    tp, fp = len(pos & derived), len(neg & derived)
    fn = len(pos - derived)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": len(neg - derived), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem_dir", default=os.path.join("data", "rhea_v142", "ilp_problems"))
    add_class_arguments(parser, required=False)
    parser.add_argument("--predicate_set", default="atoms")
    parser.add_argument("--show_errors", type=int, default=0, help="List this many FPs/FNs per split")
    args = parser.parse_args()

    classes = resolve_classes(args) if (args.classes or args.labels_file) else list(REFERENCE_PROGRAMS)
    for cls in classes:
        for split in ("train", "validation", "test"):
            split_dir = os.path.join(args.problem_dir, cls, split)
            pos, neg = read_examples(os.path.join(split_dir, "exs.pl"))
            derived = derive(REFERENCE_PROGRAMS[cls], os.path.join(split_dir, args.predicate_set, "bk.pl"), cls)
            s = score(pos, neg, derived)
            print(f"{cls:10s} {split:10s} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
                  f"(tp={s['tp']} fp={s['fp']} fn={s['fn']} tn={s['tn']})")
            if args.show_errors:
                print(f"    FP: {sorted(neg & derived)[:args.show_errors]}")
                print(f"    FN: {sorted(pos - derived)[:args.show_errors]}")


if __name__ == "__main__":
    main()
