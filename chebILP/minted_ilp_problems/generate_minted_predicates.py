"""Generate LLM auxiliary rules for the minted ILP problems and write them into the problems.

For each minted problem, one LLM call (the rule pipeline's one-try policy) proposes reusable ASP
predicates over the problem's background facts. The structurally valid ones enter the problem
type's rule library, shared by the problems of that type; the ones the problem selected are
grounded over its facts and written in as extra background predicates (``aux_*``).

Usage (from repo root):
    wsl -e bash -lc '.wslvenv/bin/python -m chebILP.minted_ilp_problems.generate_minted_predicates --model claude-opus-5'
    wsl -e bash -lc '.wslvenv/bin/python -m chebILP.minted_ilp_problems.generate_minted_predicates --print_prompt'
"""
import argparse
import os
from collections import Counter

from dotenv import load_dotenv

from chebILP.evaluation.clingo_eval import ground_extensions_isolated
from chebILP.ilp_problem import ILPProblem, _atom
from chebILP.minted_ilp_problems.build_minted_problems import MINTERS
from chebILP.predicate_generation.auxiliary_predicates import load_class_map
from chebILP.predicate_generation.auxiliary_rules import (
    DEFAULT_GROUNDING_TIMEOUT,
    RuleProgram,
    analyze_hypothesis,
    load_class_rules,
    resolve_rule_dependencies,
)
from chebILP.predicate_generation.generate_auxiliary_rules import RuleGenerator


def problem_fact_lines(problem: ILPProblem) -> list[str]:
    """The problem's background facts as clingo facts; the grounding input for its rules."""
    return [f"{_atom(pred, fact)}." for pred, facts in problem.bk_facts.items() for fact in facts]


def _key(args) -> tuple[str, ...]:
    """An example's id or a derived head's argument tuple, as comparable strings."""
    return tuple(str(a) for a in args) if isinstance(args, tuple) else (str(args),)


class MintedRuleGenerator(RuleGenerator):
    """Auxiliary rules for one minted problem, with its examples in place of ChEBI molecules.

    Only the prompts, grounding and hypothesis scoring differ from the ChEBI pipeline. The
    per-program gates (syntax, static cross-product and recursion checks, the library, the
    class map and the log) are inherited unchanged.
    """

    def __init__(self, name, minter, problem, model, n_predicates, top_k, effort=None, seed=0):
        super().__init__(minter.library_dir, model, n_predicates, top_k, molecules=None, problem_dir=None,
                         computed_facts=False, prompt_samples=0, effort=effort)
        self.name = name
        self.minter = minter
        self.problem = problem
        self.seed = seed

    @property
    def system_prompt(self) -> str:
        return self.minter.get_system_prompt()

    def _info(self) -> dict:
        """Stands in for a ChEBI class: the retrieval query is the target plus the glossary, as in
        ``ILPProblemMinter.retrieve_candidates``."""
        return {"name": self.problem.head_predicate[0], "definition": self.minter.get_glossary(),
                "parents": [], "siblings": []}

    def class_context(self, chebi_id) -> dict:
        # ``val_ids`` only has to be non-empty here; grounding and scoring read the problem directly.
        examples = self.problem.pos_samples + self.problem.neg_samples
        return {"val_facts": problem_fact_lines(self.problem), "val_ids": examples}

    def build_user_prompt(self, chebi_id, info, candidates, ctx) -> str:
        return self.minter.get_user_prompt(self.problem, candidates=candidates, seed=self.seed)

    def ground(self, progs, ctx) -> dict:
        """Every program's derived atoms over the whole problem, as ``{name: [argument tuples]}``.

        One fact group for the whole problem: a minted problem's facts carry no ``has_atom`` links,
        so the molecule split of :func:`derive_rule_extensions` cannot attribute them.
        """
        return ground_extensions_isolated(
            [p.source for p in progs], [ctx["val_facts"]], [p.name for p in progs],
            timeout=DEFAULT_GROUNDING_TIMEOUT, total_timeout=DEFAULT_GROUNDING_TIMEOUT,
        )

    def _fire_summary(self, prog, ctx) -> str:
        return f"derives {len(ctx['extensions'].get(prog.name, []))} tuple(s)"

    def _score_hypothesis(self, chebi_id, hypothesis, programs, dependencies, ctx):
        """Train-F1 of the hypothesis over the problem's labelled examples, or ``None`` if it does not ground.

        The labels are used for this diagnostic only; the class selection never sees them.
        """
        head = self.problem.head_predicate[0]
        if analyze_hypothesis(hypothesis) is None:
            return None
        hyp_prog = RuleProgram(name=head, description="LLM class hypothesis",
                               source=hypothesis, source_file="<hypothesis>")
        try:
            derived = self.ground(programs + dependencies + [hyp_prog], ctx)
        except Exception:
            return None
        fired = {_key(t) for t in derived.get(head, [])}
        pos = {_key(s) for s in self.problem.pos_samples}
        neg = {_key(s) for s in self.problem.neg_samples}
        tp, fp = len(fired & pos), len(fired & neg)
        fn, tn = len(pos - fired), len(neg - fired)
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        return {"train_f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "groundable": True}

    def generate(self) -> int:
        """One LLM call for this problem, unless its class map entry is already recorded (resume)."""
        self.retriever = self.build_retriever()
        if self.name in load_class_map(self.library_dir):
            print(f"[{self.name}] already in class_map.json, skipping the LLM call.")
            return 0
        n = self.generate_for_class(self.name, self._info())
        print(f"[{self.name}] {n} auxiliary rule(s) recorded.")
        return n

    def print_prompt(self) -> None:
        print(f"===== {self.name}: system prompt =====\n{self.system_prompt}")
        print(f"===== {self.name}: user prompt =====\n{self.minter.get_user_prompt(self.problem, seed=self.seed)}")


def augment_problem(problem: ILPProblem, library_dir: str, name: str) -> ILPProblem:
    """``problem`` plus one background predicate per auxiliary rule its class map entry selected.

    Each selected rule is grounded with the library programs it builds on, but only the selected
    programs are emitted, as ``build_bk`` does for ChEBI classes. Returns ``problem`` unchanged
    when the problem has no selection.
    """
    progs = load_class_rules(name, library_dir=library_dir)
    if not progs:
        return problem
    deps = resolve_rule_dependencies(progs, library_dir)
    derived = ground_extensions_isolated(
        [p.source for p in progs + deps], [problem_fact_lines(problem)], [p.name for p in progs],
        timeout=DEFAULT_GROUNDING_TIMEOUT, total_timeout=DEFAULT_GROUNDING_TIMEOUT,
    )
    aux = {}
    for prog in progs:
        tuples = sorted(set(derived.get(prog.name, [])))
        arities = Counter(len(t) for t in tuples)
        if not arities:
            continue
        if len(arities) > 1 or 0 in arities:
            print(f"  skipping {prog.name}: derives tuples of arities {sorted(arities)}")
            continue
        aux[prog.name] = [t[0] if len(t) == 1 else t for t in tuples]
    return ILPProblem(problem.head_predicate, problem.pos_samples, problem.neg_samples,
                      {**problem.bk_facts, **aux})


def main():
    ap = argparse.ArgumentParser(description="Generate LLM auxiliary rules for the minted problems and add them as background.")
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--out_root", default=os.path.join("data", "minted_problems_llm"))
    ap.add_argument("--library_root", default=None,
                    help="One rule library per problem type, under this directory (default: data/minted_libraries).")
    ap.add_argument("--problem_type", default=None)
    ap.add_argument("--problems", nargs="+", default=None, help="Only these problems (default: all)")
    ap.add_argument("--model", default="claude-opus-5",
                    help="A bare Claude id runs through the logged-in `claude` CLI (CHEBILP_CLAUDE_CLI if it is "
                         "not on PATH). A 'provider/name' id runs through OPENAI_API_BASE / OPENAI_API_KEY.")
    ap.add_argument("--n_predicates", type=int, default=5, help="Target number of predicates per problem.")
    ap.add_argument("--top_k", type=int, default=16, help="Reuse candidates retrieved per problem.")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                    help="Effort level for the `claude` CLI backend (default: the CLI's per-model default).")
    ap.add_argument("--print_prompt", action="store_true",
                    help="Print the prompts for the selected problems and exit. Makes no LLM call.")
    args = ap.parse_args()

    load_dotenv()

    for name, (minter_cls, minter_kwargs) in MINTERS.items():
        if args.problem_type is not None and args.problem_type != minter_cls.problem_type:
            continue
        if args.problems is not None and name not in args.problems:
            continue
        minter = minter_cls(args.out_root, library_root=args.library_root, **minter_kwargs)
        problem = minter.mint(args.seed)
        generator = MintedRuleGenerator(name, minter, problem, args.model, args.n_predicates, args.top_k,
                                        effort=args.effort, seed=args.seed)
        if args.print_prompt:
            generator.print_prompt()
            continue

        generator.generate()
        augmented = augment_problem(problem, minter.library_dir, name)
        out_dir = os.path.join(args.out_root, name)
        augmented.write_popper(out_dir)
        augmented.write_aleph(out_dir)
        added = sorted(set(augmented.bk_facts) - set(problem.bk_facts))
        print(f"{name}: +{len(added)} aux predicate(s) {added} -> {out_dir}")


if __name__ == "__main__":
    main()
