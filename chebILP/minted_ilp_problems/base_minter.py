from abc import abstractmethod, ABC
import os
import random
from popper.util import MAX_BODY, MAX_VARS

from chebILP.ilp_problem import ILPProblem, _atom
from chebILP.predicate_generation.auxiliary_generation import format_candidates
from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever


class ILPProblemMinter(ABC):
    """
    Abstract base class for classes that mint ILP problems.
    """

    # Names the predicate library; all minters of one problem type share it.
    problem_type: str

    def __init__(self, out_root=None, library_root=None):
        """
        Initialize the ILPProblemMinter.

        Args:
            out_root: The root directory where minted problems will be saved.
            library_root: The root directory holding one predicate library per problem type.
        """
        if out_root is None:
            out_root = os.path.join("data", "minted_problems")
        if library_root is None:
            library_root = os.path.join("data", "minted_libraries")
        self.out_root = out_root
        self.library_dir = os.path.join(library_root, self.problem_type)
        self._retriever = None

    @property
    def retriever(self) -> HybridPredicateRetriever:
        """Retriever over this problem type's library, built on first use."""
        if self._retriever is None:
            self._retriever = HybridPredicateRetriever.from_rule_library(
                base_dir=self.library_dir
            )
        return self._retriever

    def retrieve_candidates(
        self, ilp_problem: ILPProblem, top_k: int = 16
    ) -> list[dict]:
        """Library predicates most relevant to the problem's target and glossary."""
        if not len(self.retriever):
            return []
        target, _ = ilp_problem.head_predicate
        return self.retriever.retrieve(f"{target} {self.get_glossary()}", top_k=top_k)

    @staticmethod
    def sample_bk(
        ilp_problem: ILPProblem, per_predicate: int = 3, seed: int = 0
    ) -> str:
        """Up to ``per_predicate`` random facts of every background predicate, one per line."""
        rng = random.Random(seed)
        lines = []
        for pred, facts in ilp_problem.bk_facts.items():
            sample = rng.sample(facts, k=min(per_predicate, len(facts)))
            lines += [f"{_atom(pred, fact)}." for fact in sample]
        return "\n".join(lines)

    def static_predicates(self, ilp_problem: ILPProblem) -> frozenset:
        """Background predicates whose facts do not grow with the examples, exempt from the
        rule pipeline's cross-product check. None by default."""
        return frozenset()

    @staticmethod
    def state_free_predicates(ilp_problem: ILPProblem) -> frozenset:
        """Predicates none of whose facts mention an example id. Only a sound
        :meth:`static_predicates` where every per-example fact carries the example id, as in
        the IGGP-style problems; a trains car fact, say, names the car but not the train."""

        def first(s):
            return s[0] if isinstance(s, tuple) else s

        ids = {str(first(s)) for s in ilp_problem.pos_samples + ilp_problem.neg_samples}

        def args(f):
            return f if isinstance(f, tuple) else (f,)

        return frozenset(
            pred
            for pred, facts in ilp_problem.bk_facts.items()
            if facts and not any(str(a) in ids for f in facts for a in args(f))
        )

    def max_vars(self):
        """
        Popper parameter: Max number of variables. Set this to the value needed for the current problem.
        """
        return MAX_VARS

    def max_body(self):
        """
        Popper parameter: Max number of body literals. Set this to the value needed for the current problem.
        """
        return MAX_BODY

    @abstractmethod
    def mint(self, seed, *args, **kwargs) -> ILPProblem:
        """
        Mint an ILP problem.

        Args:
            seed: The random seed to use for the minting process.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        """
        pass

    def get_glossary(self):
        """
        Get a description of the target predicates and their arities (used as input to the LLM).
        """
        return ""

    def get_task_description(self):
        """
        Get a description of the task (used as input to the LLM, should be leakage-free).
        """
        return ""

    def get_system_prompt(self):
        """
        Get the system prompt for the minting process.

        Returns:
            A string representing the system prompt.
        """
        system_prompt = (
            "You are an expert in inductive logic programming (ILP). You design auxiliary background "
            "predicates that make a target relation easier to learn with ILP systems, and you propose a "
            "plausible target rule. \n"
            "Write clause bodies in clingo/ASP syntax. In particular use `!=` for "
            "inequality (never Prolog's `\\=`), and `=`, `<`, `<=`, `>`, `>=` for the other comparisons; "
            "`_` is an anonymous variable. The clauses are grounded with clingo.\n\n"
            "Contract for each new predicate:\n"
            "- Its name is snake_case; it is stored as `aux_<name>`, and its clauses define exactly that predicate.\n"
            "- Every variable must be bound by a positive background literal (e.g. `piece(S,A)`), in particular the "
            "example variable of the head. A variable that occurs only in a comparison or only inside an aggregate "
            "does not bind.\n"
            "- `not p(X)` is valid; `not (p(X), q(X))` is not, so define a helper predicate for the conjunction. "
            "`a ; b` is not a disjunction; write separate clauses.\n"
            "- A recursive clause that increments a counter must carry a constant upper bound in the same clause, "
            "otherwise it never finishes grounding.\n"
            "- Do not define or redefine background predicates.\n\n"
            "`auxiliary_predicates`: up to 5 predicates for concept that cannot be expressed trivially "
            "with the existing background predicates. Define each only in terms of the existing background "
            "predicates (or earlier auxiliary predicates). Aim for predicates that are meaningful and contribute to the learning process.\n"
            "`seed_hypothesis`: one clause that you believe is a reasonable (not necessarily perfect) "
            "approximation of the target, using your auxiliary predicates and/or the existing ones."
        )
        return system_prompt

    def get_user_prompt(
        self,
        ilp_problem: ILPProblem,
        candidates: list[dict] | None = None,
        seed: int = 0,
    ):
        """
        Get the user prompt for the minting process. Includes a generic task description, a predicate glossary and samples from the background knowledge
        Args:
            ilp_problem: The problem to prompt for.
            candidates: Reuse candidates; retrieved from this problem type's library if None.
            seed: Seed for the background-fact sample.
        Returns:
            A string representing the user prompt.
        """
        task_description = self.get_task_description()
        glossary = self.get_glossary()
        target, arity = ilp_problem.head_predicate
        if candidates is None:
            candidates = self.retrieve_candidates(ilp_problem)
        bk_sample = self.sample_bk(ilp_problem, seed=seed)
        return f"""\
# Task
{task_description}

The target predicate to learn is `{target}/{arity}`.

# Predicate glossary
{glossary}

# Sample background facts (a few per predicate, to show the constant vocabulary)
```prolog
{bk_sample}
```

{format_candidates(candidates)}

Answer with a single JSON object:
- "reasoning": think here first — decide which auxiliary predicates are most relevant.
  Keep it concise.
- "reuse": the candidate numbers you are reusing, as integers (empty list if none).
- "new": one entry per NEW predicate, each with "name" (snake_case), "description" (one
  short line) and "program". "program" is the clause text only — one head over the example variable plus any helper clauses.
- "hypothesis": one rule `{target}({",".join([f"V{i}" for i in range(1, arity + 1)])}) :- ...` that defines the class as a plain
  conjunction of the background and auxiliary predicates you chose (no aggregates, negation or
  arithmetic here).
"""
