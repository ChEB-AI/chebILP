"""
LLM auxiliary ASP rules for Rhea EC classes: ``RuleGenerator`` with a reaction prompt and reaction facts.

Example usage (makes one paid LLM call per class):
    python -m chebILP.reaction_processing.generate_rhea_rules \
        --classes ec_3_1_1 ec_2_7_1 ec_4_1_1 --model claude-opus-5-5 \
        --predicate_dir data/rhea_v142/llm_rules_opus

Then build the BK with ``rhea_problem_builder --predicate_set llm_generated_rules --predicate_dir <same>``.
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from chebILP.predicate_generation.auxiliary_generation import OUTPUT_CONTRACT, format_candidates
from chebILP.predicate_generation.generate_auxiliary_rules import RuleGenerator, _load_train_samples
from chebILP.reaction_processing.reaction_to_fol import build_background_reaction, build_computed_facts_reaction
from chebILP.reaction_processing.rhea_data import RheaDataset
from chebILP.reaction_processing.rhea_problem_builder import DEFAULT_PROBLEM_DIR
from chebILP.utils import add_class_arguments, resolve_classes

DEFAULT_LIBRARY_DIR = os.path.join("data", "rhea_v142", "llm_rules")

_EXISTING_PREDICATES = """\
- has_reactant(R, M) / has_product(R, M): molecule M is on the left / right side of reaction R
  (Rhea's reaction direction; stoichiometry is not given, each molecule appears once)
- has_atom(M, A): molecule M contains atom A
- c(A), n(A), o(A), p(A), s(A), h(A), ...: atom element (lowercase). Generic R-group atoms
  (`*` in SMILES) have has_atom only, no element.
- charge_p/charge_n/charge0/charge1/charge_m1(A)/...: charge of atom A (p = positive, n = negative, 0 = neutral, m1 = -1, etc.)
- has_X_hs(A) / has_at_least_X_hs(A): attached-hydrogen counts
- cip_code_R(A), cip_code_S(A): R/S CIP stereochemistry
- bSINGLE/bDOUBLE/bTRIPLE/bAROMATIC(A1, A2): bond type (symmetric)
- bSTEREOCIS/bSTEREOTRANS(A1, A2): cis/trans bond stereochemistry
- has_bond_to(A1, A2): any bond between two atoms (symmetric)
- in_ring(A), in_ringN(A): A is in a ring / in an N-membered ring
- steroid_1..steroid_17(A): atom at a steroid-nucleus position
- net_charge_positive/negative/neutral(M), aromatic/aliphatic(M): molecule-level\
"""

_COMPUTED_PREDICATES = """\
- mol_weight(M, W): integer molecular weight of molecule M (rounded)
- ring_size(M, S): M has a ring of size S.\
"""

_SYSTEM_PROMPT = """\
You are an expert in biochemistry, cheminformatics, Answer Set Programming (clingo/ASP), and
Inductive Logic Programming (ILP). Your task is to write AUXILIARY PREDICATES that help an ILP
system decide whether an enzyme-catalysed reaction belongs to an EC (Enzyme Commission) class,
as opposed to reactions of its sibling EC classes.

Each example is a REACTION R. Its molecules hang off R through has_reactant(R,M) /
has_product(R,M), and their atoms through has_atom(M,A). Molecules are the major microspecies
at pH 7.3 (carboxylates and phosphates are charged), hydrogens are implicit except that a
released proton is its own one-atom molecule ([H+]). Reactions are written in the direction of
catalysis. There is no atom mapping: you cannot follow an atom from a reactant to a product, so
express a bond change as "this group is on the left, that group is on the right".

Each predicate has three parts. The program needs to be parsed by clingo.

    name:        aux_snake_case_name
    description: <one short line describing what the predicate captures>
    program:     aux_snake_case_name(...) :- <body over the allowed predicates>.
                 % ... optional helper clauses (give helpers aux_-prefixed, specific names)

Contract:
- "program" holds ONLY clauses. Do not repeat the name or description inside it as comments.
- You MAY use clingo aggregates (#count, #sum), comparisons (=, !=, <, <=, >, >=) and
  negation-as-failure (not ...). You MAY define recursive helper predicates.
- Don't use `a ; b` as an `or`. That is a syntax error.
- A variable that appears ONLY inside an aggregate is LOCAL to it and does NOT bind the head.
  Bind it in the body outside the aggregate first.
- `not p(X)` is valid, `not (p(X), q(X))` is NOT. To negate a conjunction, define a helper.
- Every variable must be joined to the reaction or molecule it belongs to. Relate a reactant
  atom to a product atom ONLY through their shared reaction R
  (has_reactant(R,M1), has_atom(M1,A), has_product(R,M2), has_atom(M2,B)); anything else
  ranges over all reactions at once and is rejected.

There is a SHARED LIBRARY of auxiliary rules already written for other classes. REUSE them where possible.

Examples of useful kinds of predicates:

An atom-level group, reusable on either side:
    name:        aux_carboxyl_carbon
    description: carbon of a carboxyl or carboxylate group
    program:     aux_carboxyl_carbon(C) :- c(C), bDOUBLE(C,O1), o(O1), bSINGLE(C,O2), o(O2), has_1_hs(O2).
                 aux_carboxyl_carbon(C) :- c(C), bDOUBLE(C,O1), o(O1), bSINGLE(C,O2), o(O2), charge_m1(O2).

A specific co-substrate or by-product, recognised by structure:
    name:        aux_water
    description: the molecule is water
    program:     aux_bonded(A) :- has_bond_to(A,_).
                 aux_water(M) :- has_atom(M,O), o(O), has_2_hs(O), not aux_bonded(O).

A reaction-level feature: which side a group is on:
    name:        aux_consumes_water
    description: water is a reactant
    program:     aux_consumes_water(R) :- has_reactant(R,M), aux_water(M).

A change across the reaction, by counting both sides:
    name:        aux_ring_atoms_lost
    description: fewer ring atoms on the product side than on the reactant side (a ring is opened)
    program:     aux_ring_atoms_left(R,N) :- has_reactant(R,_), N = #count{ M,A : has_reactant(R,M), has_atom(M,A), in_ring(A) }.
                 aux_ring_atoms_right(R,N) :- has_product(R,_), N = #count{ M,A : has_product(R,M), has_atom(M,A), in_ring(A) }.
                 aux_ring_atoms_lost(R) :- aux_ring_atoms_left(R,L), aux_ring_atoms_right(R,K), K < L.

Guidance:
- Prefer small predicates for one chemical feature; the ILP system combines them. Favour
  predicates that can be reused across EC classes (cofactors such as NAD(P)+, ATP, CoA,
  S-adenosyl-L-methionine; groups such as esters, amides, phosphates).
- Siblings in the EC hierarchy often differ only in WHICH bond is formed or broken, or in the
  acceptor/cofactor. Look for that contrast.
- Prefer predicates TRUE for many reactions of the target class and FALSE for sibling reactions.

Finally, write a HYPOTHESIS: a single rule that defines the target class by ANDing the
predicates you chose. It becomes the ILP system's starting guess, so it must be a plain
conjunction — no aggregates, negation or arithmetic (put those inside the aux_ predicates).

    hypothesis: ec_<a>_<b>_<c>(A) :- <literal>, <literal>, ... .

Rules for the hypothesis:
- The head is exactly the target predicate given in the prompt, with A the reaction variable.
- Use ONLY the background predicates and the auxiliary predicates you reused or wrote above.
  Do NOT use mol_weight/ring_size or #count/not/comparisons here.
- A reaction-level predicate applies to A directly: aux_x(A). A molecule-level one goes through
  a side: has_reactant(A,M), aux_y(M). An atom-level one goes through a side and has_atom.
- Aim for the conjunction that best separates the target from its siblings.\
"""


def ec_label(node_id: str) -> str:
    """``ec_3_1_1`` -> ``EC 3.1.1``."""
    return "EC " + node_id[3:].replace("_", ".")


class RheaRuleGenerator(RuleGenerator):
    """``RuleGenerator`` whose examples are reactions; ``molecules`` is the Rhea reaction frame."""

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def class_context(self, chebi_id) -> dict:
        pos_rows, neg_rows = _load_train_samples(chebi_id, self.problem_dir, self.molecules)
        val_rows = pd.concat([pos_rows, neg_rows])
        val_facts, _ = build_background_reaction(val_rows)
        if self.computed_facts:
            val_facts += build_computed_facts_reaction(val_rows)
        return {
            "pos_smiles": list(pos_rows["smiles"][:self.prompt_samples]),
            "neg_smiles": list(neg_rows["smiles"][:self.prompt_samples]),
            "smiles_by_id": dict(val_rows["smiles"]),
            "val_facts": val_facts,
            "val_ids": [str(i) for i in val_rows.index],
            "val_rows": val_rows,
            "pos_ids": {str(i) for i in pos_rows.index},
        }

    def build_user_prompt(self, chebi_id, info, candidates, ctx) -> str:
        parents = "".join(f"  - {ec_label(p['id'])}: {p['name']}\n" for p in info["parents"])
        siblings = "".join(f"  - {ec_label(s['id'])}: {s['name']}\n" for s in info["siblings"])
        examples = ["Positive example reactions (reaction SMILES, reactants>>products):"]
        examples += [f"  + {s}" for s in ctx["pos_smiles"]]
        examples += ["Negative reactions (from the sibling classes):"]
        examples += [f"  - {s}" for s in ctx["neg_smiles"]]
        computed = f"\nExtra facts available in this mode:\n{_COMPUTED_PREDICATES}\n" if self.computed_facts else ""
        return f"""\
Target class: {ec_label(chebi_id)} {info['name']} (target predicate {chebi_id})
Definition: {info['definition']}
Superclass:
{parents}Sibling classes (the negatives come from these):
{siblings}
Background predicates ALREADY available:
{_EXISTING_PREDICATES}
{computed}
{chr(10).join(examples)}

{format_candidates(candidates)}
Choose up to {self.n_predicates} auxiliary predicates that would help distinguish
{ec_label(chebi_id)} reactions from reactions of the sibling classes. REUSE the candidates above
wherever they fit; only write NEW rules for properties they do not already cover.

{OUTPUT_CONTRACT}
  "program" is the clause text only — one head plus any helper clauses.
- "hypothesis": one rule `{chebi_id}(A) :- ...` that defines the class as a plain conjunction of
  the background and auxiliary predicates you chose (no aggregates, negation or arithmetic here;
  head variable A is the reaction).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_class_arguments(parser)
    parser.add_argument("--rhea_version", type=int, default=142)
    parser.add_argument("--min_pos_samples", type=int, default=25)
    parser.add_argument("--problem_dir", default=DEFAULT_PROBLEM_DIR)
    parser.add_argument("--predicate_dir", default=DEFAULT_LIBRARY_DIR, help="Auxiliary-rule library directory.")
    parser.add_argument("--model", default="claude-opus-5-5",
                        help="Bare Claude id (via the `claude` CLI) or 'provider/name' (OpenAI-compatible).")
    parser.add_argument("--n_predicates", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None)
    parser.add_argument("--no_computed_facts", dest="computed_facts", action="store_false", default=True)
    parser.add_argument("--prompt_samples", type=int, default=6)
    args = parser.parse_args()

    load_dotenv()
    dataset = RheaDataset(rhea_version=args.rhea_version, min_pos_samples=args.min_pos_samples)
    RheaRuleGenerator(
        args.predicate_dir, args.model, args.n_predicates, args.top_k,
        molecules=dataset.molecules, problem_dir=args.problem_dir, computed_facts=args.computed_facts,
        prompt_samples=args.prompt_samples, effort=args.effort,
    ).run(dataset.chebi_graph, resolve_classes(args))


if __name__ == "__main__":
    main()
