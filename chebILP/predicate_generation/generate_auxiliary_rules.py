# Pipeline: for each ChEBI class, ask an LLM to write AUXILIARY PREDICATES as ASP/clingo
# rules that help ILP define a class.
# This is the rule-based counterpart of chebILP.predicate_generation.generate_auxiliary_predicates (Python programs).
# Rules are kept in ONE shared library and retrieved per class for reuse (hybrid BM25 + dense).
#
# Usage:
#   python -m chebILP.predicate_generation.generate_auxiliary_rules \
#       --labels_file labels.txt \
#       --chebi_version 251 \
#       --molecules_path data/chebi_v251/ChEBI25_3_STAR/molecules.pkl \
#       --problem_dir data/ilp_problems \
#       --n_predicates 4

import argparse
import os
import re

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from rdkit.Chem import Descriptors

from chebi_utils.extract_properties import mol_to_fol_atoms
from chebILP.predicate_generation.auxiliary_generation import (
    OUTPUT_CONTRACT,
    AuxiliaryGenerator,
    Selection,
    format_candidates,
    format_parents,
    generate_one,
    one_line,
)
from chebILP.predicate_generation.auxiliary_rules import (
    DEFAULT_AUX_RULE_LIBRARY_DIR,
    ERROR_UNBOUNDED_RECURSION,
    RuleProgram,
    _referenced_predicates,
    add_rule_to_library,
    analyze_hypothesis,
    aux_rule_path,
    derive_rule_extensions,
    load_class_rules,
    parse_rule_program,
    prune_hypothesis,
    resolve_rule_dependencies,
    rule_program_error,
    save_class_hypothesis,
    static_rule_errors,
)
from chebILP.ilp_path_manager import get_exs_path
from chebILP.utils import get_atom_id
from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever


# Background facts the ILP system already provides. The rules may use any of these.
_EXISTING_PREDICATES = """\
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
_COMPUTED_PREDICATES = """\
- mol_weight(M, W): integer molecular weight of M (rounded), for weight thresholds
- ring_size(M, S): M has a ring of size S.\
"""


_SYSTEM_PROMPT = """\
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

# Header comments the model may repeat inside "program"; the pipeline synthesizes them.
_HEADER_RE = re.compile(r"^\s*%\s*(PREDICATE_NAME|DESCRIPTION)\s*:", re.IGNORECASE)

# Validate-and-reject grounds positives and negatives, which takes
# well under a second; anything near this ceiling is pathological rather than merely slow.
_VALIDATION_GROUNDING_TIMEOUT = 60.0

# Structural error codes for the pipeline's own gates, alongside the static-analysis codes from
# auxiliary_rules. A predicate hitting one of these cannot be ground safely/at all; the feedback
# round branches on the code, and a predicate still carrying one after its one repair is excluded.
ERROR_UNPARSEABLE = "unparseable"
ERROR_CLINGO_SYNTAX = "clingo_syntax"
ERROR_NO_GROUNDING = "no_grounding"


def _load_train_samples(chebi_id, problem_dir, molecules, max_pos=None, max_neg=None):
    """Return ``(pos_rows, neg_rows)`` DataFrames from the class's train exs.pl."""
    exs_path = get_exs_path(chebi_id, base_dir=problem_dir, split="train")
    pos_ids, neg_ids = [], []
    with open(exs_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            mol_id = line.split("(")[-1].split(")")[0]
            (pos_ids if line.startswith("pos") else neg_ids).append(mol_id)
    if max_pos is not None:
        pos_ids = pos_ids[:max_pos]
    if max_neg is not None:
        neg_ids = neg_ids[:max_neg]
    pos_rows = molecules[[i in set(pos_ids) for i in molecules.index]]
    neg_rows = molecules[[i in set(neg_ids) for i in molecules.index]]
    return pos_rows, neg_rows


def _mol_facts(mol_id, mol, computed_facts: bool) -> list[str]:
    """Atom (+ optional computed) facts for one molecule, matching build_background_chemlog."""
    lines = [f"has_atom({mol_id},{get_atom_id(a.GetIdx(), mol_id)})." for a in mol.GetAtoms()]
    atom_extensions, mol_extensions = mol_to_fol_atoms(mol)
    for predicate, indices in atom_extensions.items():
        if predicate.startswith("cip_code_"):
            predicate = "cip_code_" + predicate[-1].upper()
        if (predicate in {"EQ", "atom", "*", "r", "r#"}
                or (predicate.startswith("r") and predicate[1:].isdigit() and int(predicate[1:]) > 0)
                or not indices):
            continue
        if isinstance(indices[0], tuple):
            for args in indices:
                lines.append(f"{predicate}({','.join(get_atom_id(a, mol_id) for a in args)}).")
        else:
            for idx in indices:
                lines.append(f"{predicate}({get_atom_id(idx, mol_id)}).")
    for predicate in mol_extensions:
        lines.append(f"{predicate}({mol_id}).")
    if computed_facts:
        lines.append(f"mol_weight({mol_id},{round(Descriptors.MolWt(mol))}).")
        for ring in mol.GetRingInfo().AtomRings():
            lines.append(f"ring_size({mol_id},{len(ring)}).")
    return lines


def _build_eval_facts(rows, computed_facts: bool) -> tuple[list[str], list[str]]:
    """Return ``(facts, mol_ids)`` for a set of molecule rows."""
    facts, mol_ids = [], []
    for row in rows.itertuples():
        mol_ids.append(str(row.Index))
        facts.extend(_mol_facts(row.Index, row.mol, computed_facts))
    return facts, mol_ids


def _load_library_rule(stem: str, library_dir: str):
    """A reused library program, so new programs may build on the ones being reused."""
    path = aux_rule_path(stem, library_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return parse_rule_program(f.read(), source_file=path)


def _is_molecule_level(by_mol: dict[str, list[tuple[str, ...]]]) -> bool:
    """True when every derived atom is just ``pred(<molecule>)`` — a per-molecule flag."""
    return all(
        len(args) == 1 and args[0] == mol
        for mol, arg_tuples in by_mol.items()
        for args in arg_tuples
    )


class RuleSelection(Selection):
    """A rule pipeline's answer: the base selection plus a class hypothesis.

    ``hypothesis`` is last so the model commits to its predicates before defining the class
    in terms of them.
    """

    hypothesis: str


class PredicateRepair(BaseModel):
    """The model's answer to a single-predicate feedback round: a rewritten program."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    program: str


class HypothesisRepair(BaseModel):
    """The model's answer to a hypothesis feedback round: a rewritten class hypothesis."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    hypothesis: str


class RuleGenerator(AuxiliaryGenerator):
    """Auxiliary predicates written as ASP/clingo rule programs."""

    noun = "rule"
    selection_model = RuleSelection

    def __init__(self, library_dir, model, n_predicates, top_k, *, molecules,
                 problem_dir, computed_facts, prompt_samples, hypothesis_min_f1=0.5):
        super().__init__(library_dir, model, n_predicates, top_k)
        self.molecules = molecules
        self.problem_dir = problem_dir
        self.computed_facts = computed_facts
        self.prompt_samples = prompt_samples
        self.hypothesis_min_f1 = hypothesis_min_f1

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_retriever(self):
        return HybridPredicateRetriever.from_rule_library(base_dir=self.library_dir)

    def class_context(self, chebi_id) -> dict:
        import pandas as pd

        pos_rows, neg_rows = _load_train_samples(
            chebi_id, self.problem_dir, self.molecules
        )
        val_facts, val_ids = _build_eval_facts(pd.concat([pos_rows, neg_rows]), self.computed_facts)
        # SMILES by molecule id, so feedback can name the exact molecules a predicate missed or
        # wrongly fired on (not only the first prompt_samples used in the initial prompt).
        smiles_by_id = {}
        for rows in (pos_rows, neg_rows):
            for idx, mol in zip(rows.index, rows["mol"]):
                if mol is not None:
                    smiles_by_id[str(idx)] = Chem.MolToSmiles(mol)
        return {
            "pos_smiles": [Chem.MolToSmiles(m) for m in pos_rows["mol"][:self.prompt_samples] if m is not None],
            "neg_smiles": [Chem.MolToSmiles(m) for m in neg_rows["mol"][:self.prompt_samples] if m is not None],
            "smiles_by_id": smiles_by_id,
            "val_facts": val_facts,
            "val_ids": val_ids,
            "pos_ids": {str(i) for i in pos_rows.index},
            # Feedback-round bookkeeping (filled by repair; read back by prepare on re-runs).
            "excluded_labels": set(),
            "excluded_names": set(),
            "excluded_reasons": {},
            "repairs": [],
            "extra_attempts": [],
        }

    def build_user_prompt(self, chebi_id, info, candidates, ctx) -> str:
        definition_str = f"Definition: {info['definition']}\n" if info["definition"] else ""

        sibling_str = ""
        if info["siblings"]:
            parts = "\n".join(
                f"  - {s['name']} (CHEBI:{s['id']}): {s['definition'] or 'no definition'}"
                for s in info["siblings"][:8]
            )
            sibling_str = (
                "Sibling classes (look for the contrast, e.g. they may differ only by a count or chain length):\n" + parts + "\n"
            )

        mol_lines = []
        if ctx["pos_smiles"]:
            mol_lines.append("Positive example molecules (SMILES):")
            mol_lines += [f"  + {s}" for s in ctx["pos_smiles"]]
        if ctx["neg_smiles"]:
            mol_lines.append("Negative molecules (SMILES):")
            mol_lines += [f"  - {s}" for s in ctx["neg_smiles"]]
        mol_str = ("\n".join(mol_lines) + "\n") if mol_lines else ""

        computed_str = (
            f"\nExtra facts available in this mode:\n{_COMPUTED_PREDICATES}\n" if self.computed_facts else ""
        )

        return f"""\
Target class: {info['name']} (CHEBI:{chebi_id})
{definition_str}{format_parents(info)}{sibling_str}
Background predicates ALREADY available:
{_EXISTING_PREDICATES}
{computed_str}
{mol_str}
{format_candidates(candidates)}
Choose up to {self.n_predicates} auxiliary predicates that would help distinguish
"{info['name']}" from other molecules. REUSE the candidates above wherever they
fit; only write NEW rules for properties they do not already cover. 

{OUTPUT_CONTRACT}
  "program" is the clause text only — one molecule-level head plus any helper clauses.
- "hypothesis": one rule `chebi_{chebi_id}(A) :- ...` that defines the class as a plain
  conjunction of the background and auxiliary predicates you chose (no aggregates, negation or
  arithmetic here; head variable A is the molecule).
"""

    def to_source(self, item) -> str:
        return self._source_text(item.name, item.description, item.program)

    def _source_text(self, name, description, program) -> str:
        """The library's on-disk program text for a predicate, header synthesized from name/desc."""
        body = "\n".join(l for l in program.splitlines() if not _HEADER_RE.match(l)).strip()
        return f"% PREDICATE_NAME: {name}\n% DESCRIPTION: {one_line(description)}\n{body}\n"

    def prepare(self, blocks, ctx) -> None:
        """Syntax-check each program alone, then ground everything that survives together.

        A class's programs are allowed to build on each other, so they cannot be validated in
        isolation — a program referencing a sibling's predicate would derive nothing. Each is
        first checked on its own, which localises a syntax error to the one program that has
        it; the rest are then grounded as a set, alongside the library programs being reused,
        exactly as ``build_bk`` will ground them. A program that only fails behaviourally
        (fires on nothing, or on everything) is kept — the feedback round handles those; only a
        structural failure (unparseable, clingo error, static-safety, does-not-ground) is
        recorded here, and excludes the program only if it survives its one repair.
        """
        ctx["errors"] = {}
        ctx["error_codes"] = {}
        ctx["progs"] = {}
        # A predicate excluded by an earlier feedback round stays out of grounding — its
        # program is the one that could not ground safely — but keeps its recorded error so the
        # gate loop still reports it. ``repair`` re-runs prepare after each change.
        excluded = ctx.get("excluded_labels", set())
        for label, (code, message) in ctx.get("excluded_reasons", {}).items():
            ctx["errors"][label] = message
            ctx["error_codes"][label] = code
        for label, source in blocks:
            if label in excluded:
                continue
            prog = parse_rule_program(source, source_file=label)
            if prog is None:
                ctx["errors"][label] = "unparseable program"
                ctx["error_codes"][label] = ERROR_UNPARSEABLE
                continue
            error = rule_program_error(prog.source)
            if error is not None:
                ctx["errors"][label] = f"clingo error: {error}"
                ctx["error_codes"][label] = ERROR_CLINGO_SYNTAX
                continue
            ctx["progs"][label] = prog

        # Keyed by stem so accept_reused can find each reuse's own extension afterwards.
        reused_progs = {}
        for stem in ctx["reused_stems"]:
            prog = _load_library_rule(stem, self.library_dir)
            if prog is not None:
                reused_progs[stem] = prog
        ctx["reused_progs"] = reused_progs
        ctx["extensions_valid"] = False
        context = list(reused_progs.values())

        # Static gates that must run before clingo sees the programs, because grounding either
        # cannot survive the failure (non-terminating recursion runs the machine out of memory)
        # or cannot detect it (a cross product grounds "successfully", just wrongly and
        # enormously). Judged over the whole set, since a recursion cycle may close through a
        # sibling or a reused program.
        static = static_rule_errors(context + list(ctx["progs"].values()))
        # A reused program blamed for non-terminating recursion means the NEW programs are what
        # closed the cycle — the library grounds without them — so they go as a group. A cross
        # product is local to one clause, so a reused program carrying one says nothing about
        # the new programs and must not take them down with it.
        via_reused = sorted(
            name for name in {p.name for p in context} & set(static)
            if static[name][0] == ERROR_UNBOUNDED_RECURSION
        )
        for label, prog in list(ctx["progs"].items()):
            entry = static.get(prog.name)
            if entry is None and via_reused:
                entry = (ERROR_UNBOUNDED_RECURSION,
                         f"non-terminating recursion through reused {via_reused[0]}")
            if entry is not None:
                ctx["error_codes"][label], ctx["errors"][label] = entry
                del ctx["progs"][label]

        together = context + list(ctx["progs"].values())
        if not together or not ctx["val_ids"]:
            ctx["extensions"] = {}
            return
        try:
            ctx["extensions"] = derive_rule_extensions(
                together, ctx["val_facts"], ctx["val_ids"], timeout=_VALIDATION_GROUNDING_TIMEOUT
            )
            ctx["extensions_valid"] = True
        except Exception as e:
            # The set as a whole will not ground (unstratified negation across programs, say).
            # Nothing can be attributed, so every program is rejected with the shared cause.
            ctx["extensions"] = {}
            for label in ctx["progs"]:
                ctx["errors"][label] = f"class rules do not ground together: {str(e).strip().splitlines()[0]}"
                ctx["error_codes"][label] = ERROR_NO_GROUNDING
            ctx["progs"] = {}

    def accept(self, source, label, ctx) -> tuple[bool, str]:
        """Keep every program that parsed and grounds; drop only a surviving structural failure.

        Behavioural weakness (fires on nothing / on everything) no longer rejects — the feedback
        round in :meth:`repair` addresses it, and a still-weak predicate is kept anyway. So the
        only ``False`` here is a structural error (including one carried over from an exclusion).
        """
        if label in ctx["errors"]:
            return False, ctx["errors"][label]
        prog = ctx["progs"].get(label)
        if prog is None:
            return False, "no program"
        return True, self._fire_summary(prog, ctx)

    def accept_reused(self, stem, ctx) -> tuple[bool, str]:
        # Reuses are library programs already validated when first written; keep them all.
        return True, ""

    # --- feedback round ----------------------------------------------------------

    def _fire_summary(self, prog, ctx) -> str:
        if not ctx.get("val_ids"):
            return "kept (no validation molecules)"
        by_mol = ctx["extensions"].get(prog.name, {})
        return f"fires on {len(by_mol) / len(ctx['val_ids']):.0%}"

    def _predicate_status(self, label, ctx) -> tuple[str, str]:
        """Classify one new predicate's current state as ``ok``/``structural``/``no_fire``/``over_fire``.

        The second element is the feedback text for a repair prompt (empty when ``ok``).
        """
        if label in ctx["errors"]:
            return "structural", ctx["errors"][label]
        prog = ctx["progs"].get(label)
        if prog is None or not ctx.get("val_ids"):
            return "ok", ""
        by_mol = ctx["extensions"].get(prog.name, {})
        n = len(ctx["val_ids"])
        frac = len(by_mol) / n
        if frac == 0.0:
            samples = ctx.get("pos_smiles", [])[:self.prompt_samples]
            feedback = (
                f"This predicate fires on 0% of the {n} training molecules — it never matches, "
                "so it adds nothing. Rewrite it so it matches at least the positive examples.\n"
                "Positive example molecules it SHOULD match (SMILES):\n"
                + "\n".join(f"  + {s}" for s in samples)
            )
            return "no_fire", feedback
        # A molecule-level flag true of ~every molecule carries no information. An atom-/pair-level
        # predicate that fires everywhere still says WHICH atoms, so it is fine.
        if frac >= 0.95 and _is_molecule_level(by_mol):
            neg = {str(i) for i in ctx["val_ids"]} - ctx["pos_ids"]
            wrong = [ctx["smiles_by_id"].get(m) for m in (set(by_mol) & neg)]
            wrong = [s for s in wrong if s][:self.prompt_samples]
            feedback = (
                f"This predicate fires as a molecule-level flag on {frac:.0%} of molecules, "
                "including negatives, so it is too unspecific to separate the class. Make it "
                "more selective.\nNegative molecules it WRONGLY fires on (SMILES):\n"
                + "\n".join(f"  - {s}" for s in wrong)
            )
            return "over_fire", feedback
        return "ok", ""

    def _dependency_order(self, blocks, ctx) -> list[int]:
        """Block indices in dependency order: a predicate another builds on comes first.

        Ties and cycles keep the model's proposal order. Processing this way means a dependent
        is (re)evaluated only after the predicate it builds on has taken its final form, so a
        fix upstream is seen before we decide whether the dependent needs its own repair.
        """
        n = len(blocks)
        progs, name_to_idx = [], {}
        for i, (label, source) in enumerate(blocks):
            prog = ctx["progs"].get(label) or parse_rule_program(source, source_file=label)
            progs.append(prog)
            if prog is not None:
                name_to_idx[prog.name] = i
        needs = {i: set() for i in range(n)}
        for i, prog in enumerate(progs):
            if prog is None:
                continue
            for ref in _referenced_predicates(prog):
                j = name_to_idx.get(ref)
                if j is not None and j != i:
                    needs[i].add(j)
        order, done = [], set()
        while len(order) < n:
            ready = [i for i in range(n) if i not in done and needs[i] <= done]
            if not ready:  # cycle: emit the rest in proposal order
                ready = [i for i in range(n) if i not in done]
            for i in ready:
                order.append(i)
                done.add(i)
        return order

    def _exclude(self, label, name, ctx) -> None:
        """Give up on a predicate that stayed structurally broken; advertise it to later prompts."""
        ctx["excluded_labels"].add(label)
        ctx["excluded_names"].add(name)
        code = ctx["error_codes"].get(label, ERROR_UNPARSEABLE)
        message = ctx["errors"].get(label, "excluded: still invalid after one feedback round")
        ctx["excluded_reasons"][label] = (code, message)

    def repair(self, parsed, blocks, ctx, chebi_id, info) -> None:
        """One targeted feedback round per failing predicate, in dependency order.

        Each predicate is (re)evaluated against the current state; a still-failing one that has
        not yet had its round is re-prompted once with the specific reason (error text, missed
        positives, or wrongly-fired negatives) plus the running list of excluded predicates.
        The reply replaces the program in ``blocks`` and everything is re-validated. A predicate
        that stays structurally broken is excluded; behavioural weakness is kept. The hypothesis
        is handled later, in :meth:`finalize`, once every predicate is final.
        """
        if not parsed.new:
            return
        repaired: set[int] = set()
        for i in self._dependency_order(blocks, ctx):
            label = blocks[i][0]
            if label in ctx["excluded_labels"]:
                continue
            item = parsed.new[i]
            kind, feedback = self._predicate_status(label, ctx)
            if kind == "ok":
                continue
            if i in repaired:
                # No second round: keep a behaviourally weak one, drop a structural one.
                if kind == "structural":
                    self._exclude(label, item.name, ctx)
                    self.prepare(blocks, ctx)
                continue
            record = {"name": item.name, "kind": kind, "feedback": feedback,
                      "before": item.program, "after": None, "outcome": "kept"}
            ctx["repairs"].append(record)
            repaired.add(i)
            new_program = self._repair_predicate_call(chebi_id, info, ctx, item, feedback)
            if new_program is None:
                record["outcome"] = "call failed"
                if kind == "structural":
                    self._exclude(label, item.name, ctx)
                    self.prepare(blocks, ctx)
                continue
            record["after"] = new_program
            blocks[i] = (label, self._source_text(item.name, item.description, new_program))
            self.prepare(blocks, ctx)  # re-validate + re-ground with the repaired program
            kind2, _ = self._predicate_status(label, ctx)
            if kind2 == "structural":
                self._exclude(label, item.name, ctx)
                self.prepare(blocks, ctx)
                record["outcome"] = "excluded"
            print(f"    repaired {item.name} ({kind} -> {kind2})")

    def _repair_predicate_call(self, chebi_id, info, ctx, item, feedback):
        """Ask the model to rewrite one predicate; return the new program text or ``None``."""
        prompt = self.build_predicate_repair_prompt(chebi_id, info, ctx, item, feedback)
        try:
            parsed, _raw, attempts = generate_one(prompt, self.system_prompt, self.model, PredicateRepair)
        except Exception as e:
            ctx["extra_attempts"].extend(getattr(e, "_chebilp_attempts", []))
            print(f"    repair of {item.name} failed: {e}")
            return None
        ctx["extra_attempts"].extend(attempts)
        return parsed.program

    def _excluded_note(self, ctx) -> str:
        if not ctx["excluded_names"]:
            return ""
        names = ", ".join(sorted(ctx["excluded_names"]))
        return ("\nThese predicates were EXCLUDED (they could not be made to work) and are NOT "
                f"available — do not reference them:\n  {names}\n")

    def build_predicate_repair_prompt(self, chebi_id, info, ctx, item, feedback) -> str:
        definition_str = f"Definition: {info['definition']}\n" if info["definition"] else ""
        return f"""\
Target class: {info['name']} (CHEBI:{chebi_id})
{definition_str}
You previously wrote this auxiliary predicate, but it needs fixing.

  name:        {item.name}
  description: {item.description}
  program:
{item.program.strip()}

Problem:
{feedback}
{self._excluded_note(ctx)}
Rewrite the program to fix this problem. Keep the SAME predicate name ({item.name}) and the
same overall intent. Follow the same contract and safety rules as before.

Answer with a single JSON object:
- "reasoning": think briefly about what to change.
- "program": the rewritten clause text only (one head plus any helper clauses).
"""

    def rejection_code(self, label, ctx) -> str | None:
        return ctx.get("error_codes", {}).get(label)

    def add_to_library(self, source, chebi_id):
        return add_rule_to_library(source, library_dir=self.library_dir, chebi_id=chebi_id)

    def describe(self, saved, stem, reason) -> str:
        return f"{saved.name} -> library/{stem}.pl ({reason}): {saved.description}"

    def _score_hypothesis(self, chebi_id, hypothesis, programs, dependencies, ctx):
        """Ground one candidate hypothesis and return ``(kind, feedback, metrics)``.

        ``kind`` is ``ok`` / ``structural`` (does not parse or ground) / ``low_accuracy``
        (grounds but train-F1 below the threshold). ``metrics`` is the confusion matrix + F1
        when it ground, else ``None``.
        """
        head = f"chebi_{chebi_id}"
        if analyze_hypothesis(hypothesis) is None:
            return "structural", "The hypothesis is not a valid clause `chebi_<id>(A) :- ... .`", None
        hyp_prog = RuleProgram(name=head, description="LLM class hypothesis",
                               source=hypothesis, source_file="<hypothesis>")
        try:
            extensions = derive_rule_extensions(
                programs + dependencies + [hyp_prog],
                ctx["val_facts"], ctx["val_ids"], timeout=_VALIDATION_GROUNDING_TIMEOUT,
            )
        except Exception as e:
            return "structural", f"The hypothesis did not ground: {str(e).strip().splitlines()[0]}", None
        fired = set(extensions.get(head, {}))
        pos = ctx["pos_ids"]
        neg = {str(i) for i in ctx["val_ids"]} - pos
        tp, fp = len(fired & pos), len(fired & neg)
        fn, tn = len(pos - fired), len(neg - fired)
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        metrics = {"train_f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "groundable": True}
        if f1 < self.hypothesis_min_f1:
            fp_smiles = [ctx["smiles_by_id"].get(m) for m in (fired & neg)]
            fp_smiles = [s for s in fp_smiles if s][:self.prompt_samples]
            fn_smiles = [ctx["smiles_by_id"].get(m) for m in (pos - fired)]
            fn_smiles = [s for s in fn_smiles if s][:self.prompt_samples]
            feedback = (
                f"The hypothesis scores train-F1 {f1:.3f} (TP {tp}, FP {fp}, TN {tn}, FN {fn}), "
                f"below the target of {self.hypothesis_min_f1:.2f}.\n"
                + ("Negative molecules it WRONGLY matches (SMILES):\n"
                   + "\n".join(f"  - {s}" for s in fp_smiles) + "\n" if fp_smiles else "")
                + ("Positive molecules it MISSES (SMILES):\n"
                   + "\n".join(f"  + {s}" for s in fn_smiles) if fn_smiles else "")
            )
            return "low_accuracy", feedback, metrics
        return "ok", "", metrics

    def finalize(self, chebi_id, info, parsed, stems, ctx) -> dict | None:
        """Evaluate the model's class hypothesis, give it one feedback round, and store it.

        The hypothesis is grounded alongside the class's kept predicates (and their library
        dependencies), exactly as ``build_bk`` will assemble them, and a molecule is a positive
        prediction when it derives the ``chebi_<id>`` head. Because this runs after every
        predicate is final, a hypothesis lifted above the threshold by an upstream predicate fix
        needs no repair — it is scored against the final predicates and only re-prompted (once)
        if it still does not parse or scores too low. The train-F1 and clause are saved to the
        library's ``hypotheses.json`` for ``--seed_hypothesis`` / ``--heuristic_guidance``.
        """
        raw_hypothesis = (getattr(parsed, "hypothesis", "") or "").strip()
        if not raw_hypothesis:
            return None

        programs = load_class_rules(chebi_id, library_dir=self.library_dir)
        dependencies = resolve_rule_dependencies(programs, self.library_dir)
        available = {p.name for p in programs + dependencies}

        hypothesis = raw_hypothesis
        repaired = False
        if ctx["val_ids"]:
            kind, feedback, _ = self._score_hypothesis(chebi_id, hypothesis, programs, dependencies, ctx)
            if kind != "ok":
                record = {"name": "hypothesis", "kind": kind, "feedback": feedback,
                          "before": hypothesis, "after": None, "outcome": "kept"}
                ctx["repairs"].append(record)
                new_hypothesis = self._repair_hypothesis_call(
                    chebi_id, info, ctx, hypothesis, feedback, available
                )
                repaired = True
                if new_hypothesis:
                    hypothesis = new_hypothesis.strip()
                    record["after"] = hypothesis
                else:
                    record["outcome"] = "call failed"

        # Safety net: drop any lingering reference to an excluded/unavailable aux predicate so
        # the clause stays usable as a Popper seed even if the repair left one in.
        pruned = prune_hypothesis(hypothesis, available)
        final_hypothesis = pruned or hypothesis

        entry = {"hypothesis": final_hypothesis, "train_f1": None, "tp": None, "fp": None,
                 "tn": None, "fn": None, "groundable": False}
        if final_hypothesis != raw_hypothesis:
            entry["pruned_from"] = raw_hypothesis
        if repaired:
            entry["repaired"] = True
        if ctx["val_ids"]:
            _, _, metrics = self._score_hypothesis(chebi_id, final_hypothesis, programs, dependencies, ctx)
            if metrics is not None:
                entry.update(metrics)
                print(f"    hypothesis train-F1 {metrics['train_f1']:.3f} "
                      f"(TP {metrics['tp']}, FP {metrics['fp']}, TN {metrics['tn']}, FN {metrics['fn']})")
            else:
                print("    hypothesis did not ground")

        save_class_hypothesis(chebi_id, entry, library_dir=self.library_dir)
        return entry

    def _repair_hypothesis_call(self, chebi_id, info, ctx, hypothesis, feedback, available):
        """Ask the model to rewrite the class hypothesis; return the new clause or ``None``."""
        prompt = self.build_hypothesis_repair_prompt(chebi_id, info, ctx, hypothesis, feedback, available)
        try:
            parsed, _raw, attempts = generate_one(prompt, self.system_prompt, self.model, HypothesisRepair)
        except Exception as e:
            ctx["extra_attempts"].extend(getattr(e, "_chebilp_attempts", []))
            print(f"    hypothesis repair failed: {e}")
            return None
        ctx["extra_attempts"].extend(attempts)
        return parsed.hypothesis

    def build_hypothesis_repair_prompt(self, chebi_id, info, ctx, hypothesis, feedback, available) -> str:
        definition_str = f"Definition: {info['definition']}\n" if info["definition"] else ""
        aux_available = sorted(n for n in available if n.startswith("aux_"))
        avail_str = ("Auxiliary predicates you may use (plus the background predicates):\n  "
                     + ", ".join(aux_available) + "\n") if aux_available else ""
        return f"""\
Target class: {info['name']} (CHEBI:{chebi_id})
{definition_str}
You previously wrote this class hypothesis, but it needs fixing.

  {hypothesis}

Problem:
{feedback}
{avail_str}{self._excluded_note(ctx)}
Rewrite the hypothesis: one rule `chebi_{chebi_id}(A) :- ...` defining the class as a plain
conjunction of the available predicates (no aggregates, negation or arithmetic; head variable
A is the molecule; tie an atom-level predicate to the molecule with has_atom(A, X) first).

Answer with a single JSON object:
- "reasoning": think briefly about what to change.
- "hypothesis": the rewritten rule.
"""


def main():
    parser = argparse.ArgumentParser(description="Generate LLM auxiliary predicates as ASP rules for ChEBI classes.")
    parser.add_argument("--labels_file", required=True, help="File with one ChEBI ID per line.")
    parser.add_argument("--chebi_version", type=int, default=251)
    parser.add_argument("--molecules_path", required=True, help="Path to molecules.pkl (mols + SMILES).")
    parser.add_argument("--problem_dir", default=os.path.join("data", "ilp_problems"),
                        help="ILP problem tree holding each class's train exs.pl (for samples/validation).")
    parser.add_argument("--predicate_dir", default=DEFAULT_AUX_RULE_LIBRARY_DIR,
                        help="Shared auxiliary-RULE library directory.")
    parser.add_argument("--model", default="claude-haiku-4-5",
                        help="Claude model id for the local `claude` CLI (e.g. claude-opus-5, "
                             "claude-haiku-4-5). Calls run through the logged-in CLI and bill to your "
                             "Claude subscription; set CHEBILP_CLAUDE_CLI if it is not on PATH.")
    parser.add_argument("--n_predicates", type=int, default=4, help="Target number of predicates per class.")
    parser.add_argument("--top_k", type=int, default=16, help="Reuse candidates retrieved per class.")
    parser.add_argument("--computed_facts", dest="computed_facts", action="store_true", default=True,
                        help="Offer mol_weight/ring_size facts to the model (default: on).")
    parser.add_argument("--no_computed_facts", dest="computed_facts", action="store_false",
                        help="Do not offer the computed facts (Tier D predicates unavailable).")
    parser.add_argument("--prompt_samples", type=int, default=6,
                        help="Number of pos and neg example SMILES shown in the prompt (and in feedback).")
    parser.add_argument("--hypothesis_min_f1", type=float, default=0.7,
                        help="Train-F1 below which the class hypothesis gets one feedback round.")
    args = parser.parse_args()

    load_dotenv()

    import pickle as _pickle

    import pandas as pd

    data_dir = os.path.join("data", f"chebi_v{args.chebi_version}")
    with open(os.path.join(data_dir, "chebi_graph.pkl"), "rb") as _f:
        chebi_graph = _pickle.load(_f)
    molecules = pd.read_pickle(args.molecules_path)

    with open(args.labels_file) as f:
        chebi_ids = [line.strip() for line in f if line.strip()]

    RuleGenerator(
        args.predicate_dir, args.model, args.n_predicates, args.top_k,
        molecules=molecules, problem_dir=args.problem_dir, computed_facts=args.computed_facts,
        prompt_samples=args.prompt_samples, hypothesis_min_f1=args.hypothesis_min_f1,
    ).run(chebi_graph, chebi_ids)


if __name__ == "__main__":
    main()
