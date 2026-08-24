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
from rdkit import Chem
from rdkit.Chem import Descriptors

from chebi_utils.extract_properties import mol_to_fol_atoms
from chebILP.predicate_generation.auxiliary_generation import (
    OUTPUT_CONTRACT,
    AuxiliaryGenerator,
    format_candidates,
    format_parents,
    one_line,
)
from chebILP.predicate_generation.auxiliary_rules import (
    DEFAULT_AUX_RULE_LIBRARY_DIR,
    ERROR_UNBOUNDED_RECURSION,
    add_rule_to_library,
    aux_rule_path,
    derive_rule_extensions,
    parse_rule_program,
    rule_program_error,
    static_rule_errors,
)
from chebILP.ilp_path_manager import get_exs_path
from chebILP.utils import get_atom_id
from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever


# Background facts the ILP system already provides. The rules may use any of these.
_EXISTING_PREDICATES = """\
- has_atom(M, A): molecule M contains atom A
- c(A), n(A), p(A), cl(A), ...: atom element (lowercase)
- charge_[p|n|0|1|-1|...](A): charge of atom A (p = positive, n = negative, 0 = neutral, m1 = -1, etc.)
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
"""

# Header comments the model may repeat inside "program"; the pipeline synthesizes them.
_HEADER_RE = re.compile(r"^\s*%\s*(PREDICATE_NAME|DESCRIPTION)\s*:", re.IGNORECASE)

# Validate-and-reject grounds at most ``validate_max`` positives and negatives, which takes
# well under a second; anything near this ceiling is pathological rather than merely slow.
_VALIDATION_GROUNDING_TIMEOUT = 60.0

# Rejection codes for the pipeline's own gates, alongside the static-analysis codes from
# auxiliary_rules. Every rejection carries one, so a repair pass can branch on the failure
# type rather than reading the message.
ERROR_UNPARSEABLE = "unparseable"
ERROR_CLINGO_SYNTAX = "clingo_syntax"
ERROR_NO_GROUNDING = "no_grounding"
ERROR_DEGENERATE = "degenerate"
ERROR_NO_VALIDATION_MOLECULES = "no_validation_molecules"


def _load_train_samples(chebi_id, problem_dir, molecules, max_pos, max_neg):
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
    pos_ids, neg_ids = pos_ids[:max_pos], neg_ids[:max_neg]
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


class RuleGenerator(AuxiliaryGenerator):
    """Auxiliary predicates written as ASP/clingo rule programs."""

    noun = "rule"

    def __init__(self, library_dir, model, n_predicates, top_k, *, molecules,
                 problem_dir, computed_facts, validate_max, prompt_samples, api_base=None):
        super().__init__(library_dir, model, n_predicates, top_k, api_base=api_base)
        self.molecules = molecules
        self.problem_dir = problem_dir
        self.computed_facts = computed_facts
        self.validate_max = validate_max
        self.prompt_samples = prompt_samples

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_retriever(self):
        return HybridPredicateRetriever.from_rule_library(base_dir=self.library_dir)

    def class_context(self, chebi_id) -> dict:
        import pandas as pd

        pos_rows, neg_rows = _load_train_samples(
            chebi_id, self.problem_dir, self.molecules, self.validate_max, self.validate_max
        )
        val_facts, val_ids = _build_eval_facts(pd.concat([pos_rows, neg_rows]), self.computed_facts)
        return {
            "pos_smiles": [Chem.MolToSmiles(m) for m in pos_rows["mol"][:self.prompt_samples] if m is not None],
            "neg_smiles": [Chem.MolToSmiles(m) for m in neg_rows["mol"][:self.prompt_samples] if m is not None],
            "val_facts": val_facts,
            "val_ids": val_ids,
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
"""

    def to_source(self, item) -> str:
        body = "\n".join(l for l in item.program.splitlines() if not _HEADER_RE.match(l)).strip()
        return f"% PREDICATE_NAME: {item.name}\n% DESCRIPTION: {one_line(item.description)}\n{body}\n"

    def prepare(self, blocks, ctx) -> None:
        """Syntax-check each program alone, then ground everything that survives together.

        A class's programs are allowed to build on each other, so they cannot be validated in
        isolation — a program referencing a sibling's predicate would derive nothing. Each is
        first checked on its own, which localises a syntax error to the one program that has
        it; the rest are then grounded as a set, alongside the library programs being reused,
        exactly as ``build_bk`` will ground them. A program whose dependency was dropped at
        the syntax gate simply derives nothing and is rejected as degenerate below.
        """
        ctx["errors"] = {}
        ctx["error_codes"] = {}
        ctx["progs"] = {}
        for label, source in blocks:
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

    def _extension_gate(self, prog, key, ctx) -> tuple[bool, str]:
        """Judge one program by the extension it derived over the class's train molecules."""
        if not ctx["val_ids"]:
            ctx["error_codes"][key] = ERROR_NO_VALIDATION_MOLECULES
            return False, "no validation molecules"
        by_mol = ctx["extensions"].get(prog.name, {})
        n = len(ctx["val_ids"])
        frac = len(by_mol) / n
        if frac == 0.0:
            ctx["error_codes"][key] = ERROR_DEGENERATE
            return False, f"degenerate: fires on 0% of {n} train molecules"
        # A molecule-level flag true of every molecule carries no information. An atom- or
        # pair-level predicate that fires everywhere still says WHICH atoms, so it is kept.
        if frac >= 0.95 and _is_molecule_level(by_mol):
            ctx["error_codes"][key] = ERROR_DEGENERATE
            return False, f"degenerate: molecule-level flag on {frac:.0%} of {n} train molecules"
        return True, f"fires on {frac:.0%}"

    def accept(self, source, label, ctx) -> tuple[bool, str]:
        if label in ctx["errors"]:
            return False, ctx["errors"][label]
        return self._extension_gate(ctx["progs"][label], label, ctx)

    def accept_reused(self, stem, ctx) -> tuple[bool, str]:
        prog = ctx.get("reused_progs", {}).get(stem)
        # No grounding means no evidence about the reuse — the new programs already carry the
        # shared failure, and dropping the reuses on top of it would report a cause that was
        # never measured.
        if prog is None or not ctx.get("extensions_valid"):
            return True, ""
        return self._extension_gate(prog, stem, ctx)

    def rejection_code(self, label, ctx) -> str | None:
        return ctx.get("error_codes", {}).get(label)

    def add_to_library(self, source, chebi_id):
        return add_rule_to_library(source, library_dir=self.library_dir, chebi_id=chebi_id)

    def describe(self, saved, stem, reason) -> str:
        return f"{saved.name} -> library/{stem}.pl ({reason}): {saved.description}"


def main():
    parser = argparse.ArgumentParser(description="Generate LLM auxiliary predicates as ASP rules for ChEBI classes.")
    parser.add_argument("--labels_file", required=True, help="File with one ChEBI ID per line.")
    parser.add_argument("--chebi_version", type=int, default=248)
    parser.add_argument("--molecules_path", required=True, help="Path to molecules.pkl (mols + SMILES).")
    parser.add_argument("--problem_dir", default=os.path.join("data", "ilp_problems"),
                        help="ILP problem tree holding each class's train exs.pl (for samples/validation).")
    parser.add_argument("--predicate_dir", default=DEFAULT_AUX_RULE_LIBRARY_DIR,
                        help="Shared auxiliary-RULE library directory.")
    parser.add_argument("--model", default="anthropic/claude-haiku-4-5",
                        help="LLM as 'provider/name' (LiteLLM). Must support structured outputs, e.g. "
                             "anthropic/claude-haiku-4-5, openai/gpt-4o, gemini/gemini-2.5-pro, "
                             "ollama/llama3.1, hosted_vllm/<name> (with --api_base). The provider's API "
                             "key is read from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).")
    parser.add_argument("--api_base", default=None,
                        help="Base URL for self-hosted / OpenAI-compatible endpoints (e.g. vLLM, Ollama).")
    parser.add_argument("--n_predicates", type=int, default=4, help="Target number of predicates per class.")
    parser.add_argument("--top_k", type=int, default=16, help="Reuse candidates retrieved per class.")
    parser.add_argument("--computed_facts", dest="computed_facts", action="store_true", default=True,
                        help="Offer mol_weight/ring_size facts to the model (default: on).")
    parser.add_argument("--no_computed_facts", dest="computed_facts", action="store_false",
                        help="Do not offer the computed facts (Tier D predicates unavailable).")
    parser.add_argument("--validate_max", type=int, default=20,
                        help="Max pos and max neg train molecules used for validate-and-reject.")
    parser.add_argument("--prompt_samples", type=int, default=4,
                        help="Number of pos and neg example SMILES shown in the prompt.")
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
        validate_max=args.validate_max, prompt_samples=args.prompt_samples, api_base=args.api_base,
    ).run(chebi_graph, chebi_ids)


if __name__ == "__main__":
    main()
