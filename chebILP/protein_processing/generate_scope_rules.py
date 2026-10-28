"""
LLM auxiliary ASP rules for SCOPe folds: ``RuleGenerator`` with a protein-structure prompt and SSE facts.

Example usage (makes one paid LLM call per class):
    python -m chebILP.protein_processing.generate_scope_rules \
        --classes scop_a_4 scop_b_1 scop_c_1 --model claude-opus-5-5 \
        --predicate_dir data/scope_208/llm_rules_opus

Then build the BK with ``scope_problem_builder --predicate_set llm_generated_rules --predicate_dir <same>``.
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from chebILP.predicate_generation.auxiliary_generation import OUTPUT_CONTRACT, format_candidates
from chebILP.predicate_generation.generate_auxiliary_rules import RuleGenerator, _load_train_samples
from chebILP.protein_processing.protein_to_fol import (
    LENGTH_BINS, SHORT_LOOP, build_background_protein, build_computed_facts_protein, topology_string,
)
from chebILP.protein_processing.scope_data import ScopeDataset
from chebILP.protein_processing.scope_problem_builder import DEFAULT_PROBLEM_DIR
from chebILP.utils import add_class_arguments, resolve_classes

DEFAULT_LIBRARY_DIR = os.path.join("data", "scope_208", "llm_rules")
MAX_LISTED_SIBLINGS = 40

_EXISTING_PREDICATES = f"""\
- has_sse(D, S): domain D contains secondary-structure element (SSE) S
- helix(S), strand(S): S is an alpha helix / a beta strand (DSSP H / E; 3-10 helices, turns and
  isolated bridges are not SSEs)
- first_sse(S), last_sse(S): S is the first / last SSE of its domain
- next_sse(S1, S2): S2 is the SSE directly after S1 in sequence
- short_sse(S), medium_sse(S), long_sse(S): length bin; helices <= {LENGTH_BINS['H'][0]} / <= {LENGTH_BINS['H'][1]} / longer,
  strands <= {LENGTH_BINS['E'][0]} / <= {LENGTH_BINS['E'][1]} / longer residues
- short_loop(S1, S2): next_sse(S1,S2) with at most {SHORT_LOOP} residues between them
- parallel(S1, S2), antiparallel(S1, S2): strands S1 and S2 are H-bonded neighbours in a sheet (symmetric)
- same_sheet(S1, S2): distinct strands in the same sheet (symmetric)
- edge_strand(S): strand with exactly one paired neighbour (edge of an open sheet)
- contact(S1, S2): SSEs packed against each other (>= 3 Cα pairs within 8 Å; symmetric;
  not given for sequence neighbours)
- has_gly(S), has_pro(S): S contains a glycine / proline\
"""

_COMPUTED_PREDICATES = """\
- sse_length(S, N): S has N residues
- sse_index(S, I): S is the I-th SSE of its domain (1-based; compare to order SSEs)
- loop_length(S1, S2, N): N residues between consecutive SSEs S1 and S2
- domain_length(D, N): D has N residues with backbone coordinates\
"""

_SYSTEM_PROMPT = """\
You are an expert in structural biology, protein topology, Answer Set Programming (clingo/ASP),
and Inductive Logic Programming (ILP). Your task is to write AUXILIARY PREDICATES that help an
ILP system decide whether a protein domain belongs to a SCOP fold, as opposed to domains of the
other folds in its SCOP class.

Each example is a DOMAIN D. Its secondary-structure elements (SSEs) hang off D through
has_sse(D,S), numbered in sequence order. The facts were computed by DSSP from the domain's
coordinates: helices and strands, strand pairing with direction, sheets, and packing contacts.
Assignments are noisy: a strand may be split or missing, a short helix may be absent, and two
domains of the same fold differ in their extra elements around the common core.

Each predicate has three parts. The program needs to be parsed by clingo.

    name:        aux_snake_case_name
    description: <one short line describing what the predicate captures>
    program:     aux_snake_case_name(...) :- <body over the allowed predicates>.
                 % ... optional helper clauses (give helpers aux_-prefixed, specific names)

Contract:
- "program" holds ONLY clauses. Do not repeat the name or description inside it as comments.
- You MAY use clingo aggregates (#count, #sum), comparisons (=, !=, <, <=, >, >=) and
  negation-as-failure (not ...). You MAY define recursive helper predicates, but never carry
  a growing counter through a recursion (it does not terminate).
- Don't use `a ; b` as an `or`. That is a syntax error.
- A variable that appears ONLY inside an aggregate is LOCAL to it and does NOT bind the head.
  Bind it in the body outside the aggregate first.
- `not p(X)` is valid, `not (p(X), q(X))` is NOT. To negate a conjunction, define a helper.
- Every variable must be joined to its domain: through has_sse(D,S), or through a relation to
  an SSE that is (next_sse, parallel, same_sheet, contact, ...). Count with
  #count{ S : has_sse(D,S), ... }; an aggregate without has_sse(D,...) counts over all domains
  at once and is rejected.

There is a SHARED LIBRARY of auxiliary rules already written for other folds. REUSE them where possible.

Examples of useful kinds of predicates:

An SSE-level motif:
    name:        aux_beta_alpha_beta
    description: strand, helix, strand in sequence with the two strands parallel in one sheet
    program:     aux_beta_alpha_beta(S1) :- strand(S1), next_sse(S1,H), helix(H), next_sse(H,S2), strand(S2), parallel(S1,S2).

A domain-level count:
    name:        aux_many_strands
    description: the domain has at least 8 strands
    program:     aux_many_strands(D) :- has_sse(D,_), 8 <= #count{ S : has_sse(D,S), strand(S) }.

A domain-level property using negation:
    name:        aux_all_helical
    description: the domain has no strand
    program:     aux_has_strand(D) :- has_sse(D,S), strand(S).
                 aux_all_helical(D) :- has_sse(D,_), not aux_has_strand(D).

A sheet property:
    name:        aux_closed_sheet_strand
    description: a strand in a sheet without edge strands (a closed barrel)
    program:     aux_open_sheet(S) :- same_sheet(S,E), edge_strand(E).
                 aux_open_sheet(S) :- edge_strand(S).
                 aux_closed_sheet_strand(S) :- strand(S), same_sheet(S,_), not aux_open_sheet(S).

Guidance:
- The fold description names the common core: the number and arrangement of SSEs, the sheet
  type (parallel / antiparallel / mixed), strand order, layers (a/b/a), barrels and sandwiches,
  characteristic units (beta-alpha-beta, beta hairpin, Greek key). Turn those into predicates.
- Prefer small predicates for one structural feature; the ILP system combines them. Favour
  predicates reusable across folds.
- Prefer predicates TRUE for many domains of the target fold and FALSE for the other folds of
  its class. Allow for DSSP noise: "at least 6 parallel strands" generalises better than "exactly 8".

Finally, write a HYPOTHESIS: a single rule that defines the target fold by ANDing the
predicates you chose. It becomes the ILP system's starting guess, so it must be a plain
conjunction — no aggregates, negation or arithmetic (put those inside the aux_ predicates).

    hypothesis: scop_<class>_<fold>(A) :- <literal>, <literal>, ... .

Rules for the hypothesis:
- The head is exactly the target predicate given in the prompt, with A the domain variable.
- Use ONLY the background predicates and the auxiliary predicates you reused or wrote above.
  Do NOT use sse_length/sse_index/loop_length/domain_length or #count/not/comparisons here.
- A domain-level predicate applies to A directly: aux_x(A). An SSE-level one goes through
  has_sse: has_sse(A,S), aux_y(S).
- Aim for the conjunction that best separates the target from the other folds.\
"""


def scop_label(graph, node_id: str) -> str:
    """``scop_c_1`` -> ``SCOP c.1`` (the node's sccs)."""
    return "SCOP " + graph.nodes[node_id].get("sccs", node_id)


class ScopeRuleGenerator(RuleGenerator):
    """``RuleGenerator`` whose examples are protein domains; ``molecules`` is the SCOPe domain frame."""

    def __init__(self, *args, graph=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = graph

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def class_context(self, chebi_id) -> dict:
        pos_rows, neg_rows = _load_train_samples(chebi_id, self.problem_dir, self.molecules)
        val_rows = pd.concat([pos_rows, neg_rows])
        val_facts, _ = build_background_protein(val_rows)
        if self.computed_facts:
            val_facts += build_computed_facts_protein(val_rows)
        describe = lambda rows: [f"{row.sid} ({row.protein}): {topology_string(row.mol)}"
                                 for row in rows[:self.prompt_samples].itertuples()]
        return {
            "pos_examples": describe(pos_rows),
            "neg_examples": describe(neg_rows),
            "val_facts": val_facts,
            "val_ids": [str(i) for i in val_rows.index],
            "val_rows": val_rows,
            "pos_ids": {str(i) for i in pos_rows.index},
        }

    def build_user_prompt(self, chebi_id, info, candidates, ctx) -> str:
        label = lambda node: scop_label(self.graph, node) if self.graph is not None else node
        kind = "class" if chebi_id.count("_") == 1 else "fold"
        parents = "".join(f"  - {label(p['id'])}: {p['name']}\n" for p in info["parents"])
        siblings = "".join(f"  - {label(s['id'])}: {s['name']}\n" for s in info["siblings"][:MAX_LISTED_SIBLINGS])
        if len(info["siblings"]) > MAX_LISTED_SIBLINGS:
            siblings += f"  - ... and {len(info['siblings']) - MAX_LISTED_SIBLINGS} more\n"
        examples = ["Positive example domains (sid (protein): residues, SSEs with lengths | strand pairs):"]
        examples += [f"  + {s}" for s in ctx["pos_examples"]]
        examples += ["Negative domains (from other folds of the class):"]
        examples += [f"  - {s}" for s in ctx["neg_examples"]]
        computed = f"\nExtra facts available in this mode:\n{_COMPUTED_PREDICATES}\n" if self.computed_facts else ""
        return f"""\
Target {kind}: {label(chebi_id)} {info['name']} (target predicate {chebi_id})
Description: {info['definition'] or '(none)'}
Parent:
{parents}Siblings (the negatives come from these):
{siblings}
Background predicates ALREADY available:
{_EXISTING_PREDICATES}
{computed}
{chr(10).join(examples)}

{format_candidates(candidates)}
Choose up to {self.n_predicates} auxiliary predicates that would help distinguish
{label(chebi_id)} domains from domains of the other folds. REUSE the candidates above
wherever they fit; only write NEW rules for properties they do not already cover.

{OUTPUT_CONTRACT}
  "program" is the clause text only — one head plus any helper clauses.
- "hypothesis": one rule `{chebi_id}(A) :- ...` that defines the fold as a plain conjunction of
  the background and auxiliary predicates you chose (no aggregates, negation or arithmetic here;
  head variable A is the domain).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_class_arguments(parser)
    parser.add_argument("--scope_version", default="2.08")
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
    dataset = ScopeDataset(scope_version=args.scope_version, min_pos_samples=args.min_pos_samples)
    ScopeRuleGenerator(
        args.predicate_dir, args.model, args.n_predicates, args.top_k,
        molecules=dataset.molecules, problem_dir=args.problem_dir, computed_facts=args.computed_facts,
        prompt_samples=args.prompt_samples, effort=args.effort, graph=dataset.chebi_graph,
    ).run(dataset.chebi_graph, resolve_classes(args))


if __name__ == "__main__":
    main()
