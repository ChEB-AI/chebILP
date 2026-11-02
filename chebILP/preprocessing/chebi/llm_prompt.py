"""Prompt text for LLM rule generation on ChEBI classes."""

# Background facts the ILP system already provides. The rules may use any of these.
EXISTING_PREDICATES = """\
- has_atom(M, A): molecule M contains atom A
- c(A), n(A), p(A), cl(A), ...: atom element (lowercase)
- charge_p/charge_n/charge0/charge1/charge_m1(A)/...: charge of atom A (p = positive, n = negative, 0 = neutral, m1 = -1, etc.)
- has_X_hs(A) / has_at_least_X_hs(A): attached-hydrogen counts
- cip_code_R(A), cip_code_S(A): R/S CIP stereochemistry
- bSINGLE/bDOUBLE/bTRIPLE/bAROMATIC(A1, A2): bond type (symmetric)
- bSTEREOCIS/bSTEREOTRANS(A1, A2): cis/trans bond stereochemistry
- has_bond_to(A1, A2): any bond between two atoms (symmetric)
- in_ring(A): A is in a ring of any size
- in_ringN(A): A is in an N-membered ring.
- steroid_1..steroid_17(A): atom at a steroid-nucleus position
- net_charge_positive/negative/neutral(M), aromatic/aliphatic(M): molecule-level\
"""

# Extra molecule-level facts available only in this mode (never written to bk.pl themselves).
COMPUTED_PREDICATES = """\
- mol_weight(M, W): integer molecular weight of M (rounded), for weight thresholds
- ring_size(M, S): M has a ring of size S.\
"""


SYSTEM_PROMPT = """\
You are an expert in cheminformatics, Answer Set Programming (clingo/ASP), and Inductive
Logic Programming (ILP). Your task is to write AUXILIARY PREDICATES that help an ILP system
distinguish a ChEBI chemical class from other molecules.

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
  Bind it in the body outside the aggregate first:
      aux_x(M) :- has_atom(M,_), 2 = #count{ A : has_atom(M,A), aux_y(A) }.  % M bound: OK
      aux_x(M) :- 2 = #count{ A : has_atom(M,A), aux_y(A) }.                 % M UNSAFE: rejected
- `not p(X)` is valid, `not (p(X), q(X))` is NOT.
  To negate a conjunction, define a helper predicate.

There is a SHARED LIBRARY of auxiliary rules already written for other classes. REUSE them where possible.

Here a some examples of different kinds of useful predicates.

A substructure or local pattern. Molecule-level, "does the molecule contain this?":
    name:        aux_has_azetidine_ring
    description: contains a 4-membered ring holding a nitrogen
    program:     aux_has_azetidine_ring(M) :- has_atom(M,A), n(A), in_ring4(A).

The same kind of pattern, but naming the ATOMS that match, so ILP can reason about where they
sit and join them to other atom predicates. Prefer this when the location matters:
    name:        aux_carboxyl_carbon
    description: carbons that are the carbon of a carboxyl group (C=O with an -OH or -O anion)
    program:     aux_carboxyl_carbon(C) :- c(C), bDOUBLE(C,O1), o(O1), bSINGLE(C,O2), o(O2), has_1_hs(O2), O1 != O2.
                 aux_carboxyl_carbon(C) :- c(C), bDOUBLE(C,O1), o(O1), bSINGLE(C,O2), o(O2), charge_m1(O2), O1 != O2.


An atom PAIR, when the property is a relationship between two atoms:
    name:        aux_amide_bond
    description: pairs of (carbonyl carbon, amide nitrogen) joined by an amide bond
    program:     aux_amide_bond(C,N) :- c(C), o(O), bDOUBLE(C,O), has_bond_to(C,N), n(N).

A count, expressed with aggregates. Use this if it is relevant HOW MANY of a group there are (N
carbons, N sugar units, one vs two carboxyls).
    name:        aux_exactly_two_ether_oxygens
    description: has exactly two ether oxygens (O bonded to two carbons, no hydrogen)
    program:     aux_ether_oxygen(O) :- o(O), has_0_hs(O), has_bond_to(O,C1),
                     c(C1), has_bond_to(O,C2), c(C2), C1 != C2.
                 aux_exactly_two_ether_oxygens(M) :- has_atom(M,_),
                     2 = #count{ O : has_atom(M,O), aux_ether_oxygen(O) }.

An absence, via negation. Note it BUILDS ON aux_carboxyl_carbon above instead of restating
it.
    name:        aux_no_carboxyl
    description: has no carboxyl group
    program:     aux_has_carboxyl(M) :- has_atom(M,C), aux_carboxyl_carbon(C).
                 aux_no_carboxyl(M) :- has_atom(M,_), not aux_has_carboxyl(M).

A chain length, path or connectivity property, expressed with recursion:
    name:        aux_large_carbon_skeleton
    description: has a connected carbon subgraph of at least 22 carbons
    program:     aux_carbon_reachable(A,A) :- c(A).
                 aux_carbon_reachable(A,D) :- aux_carbon_reachable(A,B), has_bond_to(B,D), c(D).
                 aux_large_carbon_skeleton(M) :- has_atom(M,A), c(A),
                     22 <= #count{ B : aux_carbon_reachable(A,B) }.

A property that needs computed facts (molecular weight, or rings larger than 8):
    name:        aux_has_macrocycle
    description: contains a ring larger than 8 atoms
    program:     aux_has_macrocycle(M) :- ring_size(M,S), S > 8.
    (weight variant: aux_high_molecular_weight(M) :- mol_weight(M,W), W >= 500.)

Guidance (learned from failure analysis):
- Prefer small predicates identifying specific molecular features. The ILP system will later
  combine them into larger conjunctions. Favor predicates that can be reused across many
  classes.
- Counting and absence are the highest-value predicates: a close sibling often differs only in
  the NUMBER of a group (N carbons, N sugar units, one vs two carboxyls). Reach for #count / not.
- Prefer predicates TRUE for many molecules of the target class and FALSE for other molecules.

Finally, write a HYPOTHESIS: a single rule that defines the target class by ANDing the
predicates you chose. This becomes the ILP system's starting guess, so it must be a plain
conjunction — no aggregates, negation or arithmetic in the hypothesis itself (put those inside
the aux_ predicates instead).

    hypothesis: chebi_<id>(A) :- <literal>, <literal>, ... .

Rules for the hypothesis:
- The head is exactly chebi_<id>(A), with A the molecule variable.
- Use ONLY the background predicates and the auxiliary predicates you reused or wrote above
  (by their aux_ name). Do NOT use mol_weight/ring_size or #count/not/comparisons here.
- A molecule-level predicate applies to A directly: aux_x(A). An atom-level one must be tied
  to the molecule with has_atom(A, X) first: has_atom(A, X), aux_y(X).
- Aim for the conjunction that best separates the target from its siblings; fewer, decisive
  literals beat many weak ones.\
"""
