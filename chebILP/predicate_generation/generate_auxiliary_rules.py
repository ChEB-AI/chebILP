# Pipeline: for each target class of a dataset (ChEBI, Rhea, SCOPe), ask an LLM to write
# AUXILIARY PREDICATES as ASP/clingo rules that help ILP define the class.
# Rules are kept in ONE shared library and retrieved per class for reuse (hybrid BM25 + dense).
# Everything dataset-specific (prompt text, BK facts, example descriptions) comes from the
# dataset; the class's train examples come from its exs.pl, so build_samples must run first.
#
# Usage:
#   python -m chebILP.predicate_generation.generate_auxiliary_rules \
#       --dataset chebi --version 251 \
#       --labels_file labels.txt \
#       --problem_dir data/ilp_problems \
#       --n_predicates 4

import argparse
import os
import re

import pandas as pd
from dotenv import load_dotenv

from chebILP.predicate_generation.auxiliary_generation import (
    OUTPUT_CONTRACT,
    AuxiliaryGenerator,
    Selection,
    format_candidates,
    format_parents,
    one_line,
)
from chebILP.predicate_generation.auxiliary_predicates import sanitize_predicate_name
from chebILP.predicate_generation.auxiliary_rules import (
    ERROR_UNBOUNDED_RECURSION,
    RuleProgram,
    add_rule_to_library,
    analyze_hypothesis,
    aux_rule_path,
    derive_rule_extensions,
    load_class_rules,
    parse_rule_program,
    prune_hypothesis,
    referenced_fact_predicates,
    resolve_rule_dependencies,
    rule_program_error,
    save_class_hypothesis,
    static_rule_errors,
)
from chebILP.preprocessing.base_problem_builder import read_examples
from chebILP.preprocessing.chebi.fg_matching import (
    FG_SEED_PREFIXES,
    FG_SEED_SOURCES,
    is_fg_seed_name,
    seed_fg_library,
)
from chebILP.preprocessing.registry import add_dataset_arguments, dataset_from_args
from chebILP.ilp_path_manager import class_head
from chebILP.utils import add_class_arguments, resolve_classes
from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever

# Header comments the model may repeat inside "program"; the pipeline synthesizes them.
_HEADER_RE = re.compile(r"^\s*%\s*(PREDICATE_NAME|DESCRIPTION)\s*:", re.IGNORECASE)


def _new_rule_renames(items) -> dict[str, str]:
    """``{name: aux_name}`` for each model-written predicate whose name carries a seed prefix.

    A seed prefix (``chembl_fg_``, ``efg_``, ``efga_``) marks a header-only group whose facts come
    from an RDKit matcher, so a *clause* under such a name is treated as a seed downstream: its
    body is never grounded and, with no matcher of that name, it gets no facts at all. Only
    ``seed_fg_library`` may write those names; a program the model writes is forced to ``aux_``.
    """
    renames = {}
    for item in items:
        name = sanitize_predicate_name(item.name)
        if is_fg_seed_name(name):
            renames[name] = "aux_" + name
    return renames


def _apply_renames(text: str, renames: dict[str, str]) -> str:
    for old, new in renames.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


# Validate-and-reject grounds positives and negatives, which takes
# well under a second; anything near this ceiling is pathological rather than merely slow.
_VALIDATION_GROUNDING_TIMEOUT = 60.0

# Structural error codes for the pipeline's own gates, alongside the static-analysis codes from
# auxiliary_rules. A predicate hitting one of these cannot be ground safely/at all; the feedback
# round branches on the code, and a predicate still carrying one after its one repair is excluded.
ERROR_UNPARSEABLE = "unparseable"
ERROR_CLINGO_SYNTAX = "clingo_syntax"
ERROR_NO_GROUNDING = "no_grounding"


def _load_library_rule(stem: str, library_dir: str):
    """A reused library program, so new programs may build on the ones being reused."""
    path = aux_rule_path(stem, library_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return parse_rule_program(f.read(), source_file=path)


class RuleSelection(Selection):
    """A rule pipeline's answer: the base selection plus a class hypothesis.

    ``hypothesis`` is last so the model commits to its predicates before defining the class
    in terms of them.
    """

    hypothesis: str


class RuleGenerator(AuxiliaryGenerator):
    """Auxiliary predicates written as ASP/clingo rule programs."""

    noun = "rule"
    selection_model = RuleSelection
    # Background predicates exempt from the cross-product check (see ``cross_product_errors``).
    static_predicates: frozenset = frozenset()

    def __init__(
        self,
        library_dir,
        model,
        n_predicates,
        top_k,
        *,
        dataset,
        problem_dir,
        computed_facts,
        prompt_samples,
        effort=None,
    ):
        super().__init__(library_dir, model, n_predicates, top_k, effort)
        self.dataset = dataset
        self.problem_dir = problem_dir
        self.computed_facts = computed_facts
        self.prompt_samples = prompt_samples

    @property
    def system_prompt(self) -> str:
        return self.dataset.prompt_spec.system_prompt

    def build_retriever(self):
        return HybridPredicateRetriever.from_rule_library(base_dir=self.library_dir)

    def class_context(self, class_id) -> dict:
        """The class's train examples (from its exs.pl) and their BK, for the prompt and validation."""
        pos, neg = read_examples(class_id, self.problem_dir, "train")
        examples = self.dataset.molecules
        pos_rows = examples[examples.index.isin(pos)]
        neg_rows = examples[examples.index.isin(neg)]
        val_rows = pd.concat([pos_rows, neg_rows])
        val_facts = list(self.dataset.build_background(val_rows)[0])
        if self.computed_facts:
            val_facts += self.dataset.build_computed_facts(val_rows)
        return {
            "pos_examples": [
                self.dataset.describe_example(i)
                for i in pos_rows.index[: self.prompt_samples]
            ],
            "neg_examples": [
                self.dataset.describe_example(i)
                for i in neg_rows.index[: self.prompt_samples]
            ],
            "val_facts": val_facts,
            "val_ids": [str(i) for i in val_rows.index],
            "val_rows": val_rows,
            "pos_ids": {str(i) for i in pos_rows.index},
        }

    def build_user_prompt(self, class_id, info, candidates, ctx) -> str:
        spec = self.dataset.prompt_spec
        noun = spec.example_noun
        head = class_head(class_id)
        definition_str = (
            f"Definition: {info['definition']}\n" if info["definition"] else ""
        )

        sibling_str = ""
        siblings = info["siblings"]
        if siblings:
            shown = siblings[: spec.max_siblings] if spec.max_siblings else siblings
            parts = [
                f"  - {s['name']} ({s['label']})"
                + (
                    f": {s['definition'] or 'no definition'}"
                    if spec.relative_definitions
                    else ""
                )
                for s in shown
            ]
            if spec.count_hidden_siblings and len(shown) < len(siblings):
                parts.append(f"  - ... and {len(siblings) - len(shown)} more")
            sibling_str = spec.sibling_header + "\n" + "\n".join(parts) + "\n"

        example_lines = []
        if ctx["pos_examples"]:
            example_lines.append(f"Positive example {noun}s ({spec.example_format}):")
            example_lines += [f"  + {s}" for s in ctx["pos_examples"]]
        if ctx["neg_examples"]:
            example_lines.append(f"Negative {noun}s ({spec.example_format}):")
            example_lines += [f"  - {s}" for s in ctx["neg_examples"]]
        example_str = ("\n".join(example_lines) + "\n") if example_lines else ""

        computed_str = (
            f"\nExtra facts available in this mode:\n{spec.computed_predicates}\n"
            if self.computed_facts
            else ""
        )

        return f"""\
Target class: {info["name"]} ({info["label"]})
{definition_str}{format_parents(info, spec.relative_definitions)}{sibling_str}
Background predicates ALREADY available:
{spec.existing_predicates}
{computed_str}
{example_str}
{format_candidates(candidates)}{self.dataset.candidate_notes(c["name"] for c in candidates)}
Choose up to {self.n_predicates} auxiliary predicates that would help distinguish
"{info["name"]}" from other {noun}s. REUSE the candidates above wherever they
fit; only write NEW rules for properties they do not already cover.

{OUTPUT_CONTRACT}
  "program" is the clause text only — one {noun}-level head plus any helper clauses.
- "hypothesis": one rule `{head}(A) :- ...` that defines the class as a plain
  conjunction of the background and auxiliary predicates you chose (no aggregates, negation or
  arithmetic here; head variable A is the {noun}).
"""

    def to_source(self, item) -> str:
        renames = _new_rule_renames([item])
        name = renames.get(sanitize_predicate_name(item.name), item.name)
        return self._source_text(
            name, item.description, _apply_renames(item.program, renames)
        )

    def _source_text(self, name, description, program) -> str:
        """The library's on-disk program text for a predicate, header synthesized from name/desc."""
        body = "\n".join(
            ln for ln in program.splitlines() if not _HEADER_RE.match(ln)
        ).strip()
        return f"% PREDICATE_NAME: {name}\n% DESCRIPTION: {one_line(description)}\n{body}\n"

    def prepare(self, blocks, ctx) -> None:
        """Syntax-check each program alone, then ground everything that survives together.

        A class's programs are allowed to build on each other, so they cannot be validated in
        isolation — a program referencing a sibling's predicate would derive nothing. Each is
        first checked on its own, which localises a syntax error to the one program that has
        it; the rest are then grounded as a set, alongside the library programs being reused,
        exactly as ``build_bk`` will ground them. A program that only fails behaviourally
        (fires on nothing, or on everything) is kept; only a structural failure (unparseable,
        clingo error, static-safety, does-not-ground) is recorded here, and such a program is
        the only kind :meth:`accept` drops.
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
        static = static_rule_errors(
            context + list(ctx["progs"].values()), self.static_predicates
        )
        # A reused program blamed for non-terminating recursion means the NEW programs are what
        # closed the cycle — the library grounds without them — so they go as a group. A cross
        # product is local to one clause, so a reused program carrying one says nothing about
        # the new programs and must not take them down with it.
        via_reused = sorted(
            name
            for name in {p.name for p in context} & set(static)
            if static[name][0] == ERROR_UNBOUNDED_RECURSION
        )
        for label, prog in list(ctx["progs"].items()):
            entry = static.get(prog.name)
            if entry is None and via_reused:
                entry = (
                    ERROR_UNBOUNDED_RECURSION,
                    f"non-terminating recursion through reused {via_reused[0]}",
                )
            if entry is not None:
                ctx["error_codes"][label], ctx["errors"][label] = entry
                del ctx["progs"][label]

        together = context + list(ctx["progs"].values())
        if not together or not ctx["val_ids"]:
            ctx["extensions"] = {}
            return
        try:
            ctx["extensions"] = self.ground(together, ctx)
            ctx["extensions_valid"] = True
        except Exception as e:
            # The set as a whole will not ground (unstratified negation across programs, say).
            # Nothing can be attributed, so every program is rejected with the shared cause.
            ctx["extensions"] = {}
            for label in ctx["progs"]:
                ctx["errors"][label] = (
                    f"class rules do not ground together: {str(e).strip().splitlines()[0]}"
                )
                ctx["error_codes"][label] = ERROR_NO_GROUNDING
            ctx["progs"] = {}

    def ground(self, progs, ctx) -> dict:
        """Derived atoms of ``progs`` per validation example, grounded over the class's facts."""
        return derive_rule_extensions(
            progs,
            self._with_seed_facts(progs, ctx),
            ctx["val_ids"],
            timeout=_VALIDATION_GROUNDING_TIMEOUT,
        )

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

    def _fire_summary(self, prog, ctx) -> str:
        if not ctx.get("val_ids"):
            return "kept (no validation molecules)"
        by_mol = ctx["extensions"].get(prog.name, {})
        return f"fires on {len(by_mol) / len(ctx['val_ids']):.0%}"

    def _with_seed_facts(self, progs, ctx) -> list[str]:
        """``ctx["val_facts"]`` plus the seeded functional-group facts ``progs`` reference.

        A seeded ``chembl_fg_*`` / ``efg_*`` predicate has no clause body — its extension is the
        RDKit match — so any program (a reused seed, a rule that builds on one, or the
        hypothesis) referencing it needs those facts supplied at grounding, exactly as
        ``build_bk`` does.
        """
        needed = {p.name for p in progs if is_fg_seed_name(p.name)}
        needed |= referenced_fact_predicates(progs, FG_SEED_PREFIXES)
        if not needed:
            return ctx["val_facts"]
        return ctx["val_facts"] + self.dataset.seed_fact_lines(needed, ctx["val_rows"])

    def rejection_code(self, label, ctx) -> str | None:
        return ctx.get("error_codes", {}).get(label)

    def add_to_library(self, source, class_id):
        return add_rule_to_library(
            source, library_dir=self.library_dir, chebi_id=class_id
        )

    def describe(self, saved, stem, reason) -> str:
        return f"{saved.name} -> library/{stem}.pl ({reason}): {saved.description}"

    def _score_hypothesis(self, class_id, hypothesis, programs, dependencies, ctx):
        """Ground one hypothesis and return its confusion-matrix + train-F1 metrics, or ``None``.

        ``None`` means the hypothesis is not a valid clause or did not ground. Grounded alongside
        the class's kept predicates (and their library dependencies) plus any referenced ChEMBL
        facts, exactly as ``build_bk`` will assemble them; a molecule is a positive prediction
        when it derives the ``chebi_<id>`` head.
        """
        head = class_head(class_id)
        if analyze_hypothesis(hypothesis) is None:
            return None
        hyp_prog = RuleProgram(
            name=head,
            description="LLM class hypothesis",
            source=hypothesis,
            source_file="<hypothesis>",
        )
        all_progs = programs + dependencies + [hyp_prog]
        try:
            extensions = self.ground(all_progs, ctx)
        except Exception:
            return None
        fired = set(extensions.get(head, {}))
        pos = ctx["pos_ids"]
        neg = {str(i) for i in ctx["val_ids"]} - pos
        tp, fp = len(fired & pos), len(fired & neg)
        fn, tn = len(pos - fired), len(neg - fired)
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        return {
            "train_f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "groundable": True,
        }

    def finalize(self, class_id, info, parsed, stems, ctx) -> dict | None:
        """Score the model's one-shot class hypothesis and store it (no feedback round).

        The hypothesis is grounded alongside the class's kept predicates (and their library
        dependencies), exactly as ``build_bk`` will assemble them. ``prune_hypothesis`` drops any
        reference to a predicate that is not available, so the clause stays usable as a Popper
        seed; if no safe clause survives, the stored hypothesis is ``None`` and the model's clause
        is kept under ``pruned_from``. The train-F1 and clause are saved to the library's ``hypotheses.json`` for
        ``--seed_hypothesis`` / ``--heuristic_guidance``.
        """
        raw_hypothesis = (getattr(parsed, "hypothesis", "") or "").strip()
        if not raw_hypothesis:
            return None
        # Follow the aux_ renames to_source gave seed-prefixed new predicates.
        raw_hypothesis = _apply_renames(raw_hypothesis, _new_rule_renames(parsed.new))

        programs = load_class_rules(class_id, library_dir=self.library_dir)
        # Seeded chembl_fg_*/efg_* predicates have no clause body to depend on; resolve only ASP.
        asp_programs = [p for p in programs if not is_fg_seed_name(p.name)]
        dependencies = resolve_rule_dependencies(asp_programs, self.library_dir)
        available = {p.name for p in programs + dependencies}

        final_hypothesis = prune_hypothesis(raw_hypothesis, available)

        entry = {
            "hypothesis": final_hypothesis,
            "train_f1": None,
            "tp": None,
            "fp": None,
            "tn": None,
            "fn": None,
            "groundable": False,
        }
        if final_hypothesis != raw_hypothesis:
            entry["pruned_from"] = raw_hypothesis
        if final_hypothesis is None:
            print(
                "    hypothesis unusable: unreadable, or it needs a predicate that was not accepted"
            )
        elif ctx["val_ids"]:
            metrics = self._score_hypothesis(
                class_id, final_hypothesis, programs, dependencies, ctx
            )
            if metrics is not None:
                entry.update(metrics)
                print(
                    f"    hypothesis train-F1 {metrics['train_f1']:.3f} "
                    f"(TP {metrics['tp']}, FP {metrics['fp']}, TN {metrics['tn']}, FN {metrics['fn']})"
                )
            else:
                print("    hypothesis did not ground")

        save_class_hypothesis(class_id, entry, library_dir=self.library_dir)
        return entry


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM auxiliary predicates as ASP rules for a dataset's classes."
    )
    add_class_arguments(parser)
    add_dataset_arguments(parser)
    parser.add_argument(
        "--problem_dir",
        default=None,
        help="ILP problem tree holding each class's train exs.pl (for samples/validation). "
        "Default: <dataset dir>/ilp_problems.",
    )
    parser.add_argument(
        "--predicate_dir",
        required=True,
        help="Shared auxiliary-RULE library directory.",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="A bare Claude id (claude-opus-5, claude-haiku-4-5) runs through the "
        "logged-in `claude` CLI and bills your Claude subscription (set "
        "CHEBILP_CLAUDE_CLI if it is not on PATH). A 'provider/name' id "
        "(openai/gpt-4o) runs through an OpenAI-compatible endpoint set by "
        "OPENAI_API_BASE and OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--n_predicates",
        type=int,
        default=4,
        help="Target number of predicates per class.",
    )
    parser.add_argument(
        "--top_k", type=int, default=16, help="Reuse candidates retrieved per class."
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Effort level for the `claude` CLI backend. Default: the CLI's per-model "
        "default (medium for Opus 5.5, high for Sonnet 5.5 and Haiku 5.5). "
        "Ignored for 'provider/name' models.",
    )
    parser.add_argument(
        "--computed_facts",
        dest="computed_facts",
        action="store_true",
        default=True,
        help="Offer mol_weight/ring_size facts to the model (default: on).",
    )
    parser.add_argument(
        "--no_computed_facts",
        dest="computed_facts",
        action="store_false",
        help="Do not offer the computed facts (Tier D predicates unavailable).",
    )
    parser.add_argument(
        "--prompt_samples",
        type=int,
        default=6,
        help="Number of pos and neg example SMILES shown in the prompt.",
    )
    parser.add_argument(
        "--seed_predicates",
        choices=["none", *FG_SEED_SOURCES],
        default="none",
        help="Before generating, seed the library with a functional-group set the "
        "model can reuse: 'chembl_fgs' (RDKit ChEMBL alerts) or 'efg' (Extended "
        "Functional Groups). Default 'none'. Idempotent — existing seed files are "
        "kept, so it is safe to resume a run with the same flag.",
    )
    args = parser.parse_args()

    if args.seed_predicates != "none" and args.dataset != "chebi":
        parser.error("--seed_predicates applies to --dataset chebi only")

    load_dotenv()

    if args.seed_predicates != "none":
        written = seed_fg_library(args.predicate_dir, args.seed_predicates)
        print(
            f"Seeded {written} new '{args.seed_predicates}' functional-group predicate(s) "
            f"into {args.predicate_dir}"
        )

    dataset = dataset_from_args(args)
    RuleGenerator(
        args.predicate_dir,
        args.model,
        args.n_predicates,
        args.top_k,
        dataset=dataset,
        problem_dir=args.problem_dir or os.path.join(dataset.data_dir, "ilp_problems"),
        computed_facts=args.computed_facts,
        prompt_samples=args.prompt_samples,
        effort=args.effort,
    ).run(dataset, resolve_classes(args))


if __name__ == "__main__":
    main()
