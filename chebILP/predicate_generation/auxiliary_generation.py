"""Shared scaffolding for the LLM auxiliary-predicate generation pipelines.

Two pipelines sit on this base and differ only in what an auxiliary predicate *is*:

- :mod:`chebILP.predicate_generation.generate_auxiliary_predicates` — a Python program over an RDKit Mol.
- :mod:`chebILP.predicate_generation.generate_auxiliary_rules` — an ASP/clingo rule program.

Everything else (class lookup, reuse retrieval, the LLM call, validate-and-reject,
library storage, class_map bookkeeping, the exchange log) is shared and lives here.

The model answers with a structured output — one JSON object per class — so prose can
never reach the parser and there are no fenced code blocks to scrape. The leading
``reasoning`` field is the model's scratchpad: it is written to the generation log and
otherwise ignored. ``name``/``description`` arrive as fields and the pipeline synthesizes
the on-disk header from them, so the storage contract of each library is unchanged.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from chebILP.predicate_generation.auxiliary_predicates import load_class_map, set_class_predicates
from chebILP.ilp_path_manager import get_aux_generation_log_path
from chebILP.predicate_generation.llm_client import structured_completion
from chebILP.utils import sort_labels_by_hierarchy


class NewItem(BaseModel):
    """One newly written auxiliary predicate."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    program: str


class Selection(BaseModel):
    """The model's answer for one class.

    Field order is generation order: ``reasoning`` comes first so the model can think
    before committing to a selection.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str
    reuse: list[int]
    new: list[NewItem]


def get_class_info(chebi_graph, chebi_id: str) -> dict:
    """Name, definition, parents and sibling classes (the parent's other children)."""
    node_data = dict(chebi_graph.nodes.get(chebi_id, {}))
    parents = []
    siblings: dict[str, dict] = {}
    for pid in chebi_graph.successors(chebi_id):
        pdata = dict(chebi_graph.nodes.get(pid, {}))
        parents.append({"id": pid, "name": pdata.get("name", f"CHEBI:{pid}"), "definition": pdata.get("definition")})
        for sid in chebi_graph.predecessors(pid):
            if sid == chebi_id or sid in siblings:
                continue
            sdata = dict(chebi_graph.nodes.get(sid, {}))
            siblings[sid] = {"id": sid, "name": sdata.get("name", f"CHEBI:{sid}"), "definition": sdata.get("definition")}
    return {
        "name": node_data.get("name", f"CHEBI:{chebi_id}"),
        "definition": node_data.get("definition"),
        "parents": parents,
        "siblings": list(siblings.values()),
    }


def one_line(text: str) -> str:
    """Collapse text to a single line — both libraries' headers are line-oriented."""
    return " ".join(text.split())


def format_parents(info: dict) -> str:
    if not info["parents"]:
        return ""
    parts = "\n".join(
        f"  - {p['name']} (CHEBI:{p['id']}): {p['definition'] or 'no definition'}" for p in info["parents"]
    )
    return f"Superclass(es):\n{parts}\n"


def format_candidates(candidates: list[dict], with_kind: bool = False) -> str:
    if not candidates:
        return "Reuse candidates from the shared library: (none yet)\n"
    lines = []
    for i, c in enumerate(candidates, 1):
        label = f"{c['name']} ({c['kind']})" if with_kind else c["name"]
        lines.append(f"[{i}] {label}: {c['description'] or 'no description'}")
    return "Reuse candidates from the shared library (most relevant first):\n" + "\n".join(lines) + "\n"


OUTPUT_CONTRACT = """\
Answer with a single JSON object:
- "reasoning": think here first — decide what separates the target from other molecules.
  Keep it concise.
- "reuse": the candidate numbers you are reusing, as integers (empty list if none).
- "new": one entry per NEW predicate, each with "name" (snake_case), "description" (one
  short line) and "program".\
"""


def generate_one(prompt: str, system: str, model: str, selection_model, api_base: str | None = None):
    """Ask the model for one class's selection. Returns ``(parsed, raw_json_text, attempts)``."""
    return structured_completion(model, system, prompt, selection_model, api_base=api_base)


class AuxiliaryGenerator(ABC):
    """Per-class generate -> validate -> store loop, shared by both pipelines.

    Subclasses supply the prompt, the program contract and the validation step; see the
    abstract methods below.
    """

    noun = "predicate"  # log/console wording
    selection_model: type[Selection] = Selection

    def __init__(self, library_dir, model, n_predicates, top_k, api_base=None):
        self.library_dir = library_dir
        self.model = model
        self.n_predicates = n_predicates
        self.top_k = top_k
        self.api_base = api_base
        self.retriever = None

    # --- hooks -------------------------------------------------------------------

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The contract the model writes programs against."""

    @abstractmethod
    def build_user_prompt(self, chebi_id, info, candidates, ctx) -> str:
        """Render the per-class user prompt."""

    @abstractmethod
    def to_source(self, item) -> str:
        """Turn one ``NewItem`` into the library's on-disk program text."""

    @abstractmethod
    def accept(self, source, label, ctx) -> tuple[bool, str]:
        """Validate a program. Returns ``(accepted, reason)``; reason is shown when rejected."""

    @abstractmethod
    def add_to_library(self, source, chebi_id):
        """Store the program. Returns ``(stem, saved)`` or ``None``."""

    @abstractmethod
    def build_retriever(self):
        """Hybrid retriever over this pipeline's shared library."""

    def class_context(self, chebi_id) -> dict:
        """Per-class data the prompt and/or validation need (example molecules, facts...)."""
        return {}

    def prepare(self, blocks, ctx) -> None:
        """Pre-pass over a class's whole selection, before any single program is validated.

        ``blocks`` is ``[(label, source), ...]`` for the new programs, in the order the model
        wrote them; ``ctx["reused_stems"]`` names the library programs already selected. A
        pipeline whose programs are validated as a set rather than one at a time does that
        work here and leaves the result in ``ctx`` for :meth:`accept` to read back.
        """

    def retriever_entry(self, stem, saved) -> dict:
        return {"name": saved.name, "description": saved.description, "kind": "rule", "stem": stem}

    def rejection_code(self, label, ctx) -> str | None:
        """Stable identifier for *why* ``label`` was rejected, or ``None`` if uncategorised.

        Recorded alongside the human-readable reason so a later pass — a repair prompt that
        feeds the failure back to the model, say — can branch on the error type instead of
        parsing prose. Only pipelines that categorise their rejections override this.
        """
        return None

    def describe(self, saved, stem, reason) -> str:
        return f"{saved.name} -> library/{stem} ({reason}): {saved.description}"

    # --- template ----------------------------------------------------------------

    def generate_for_class(self, chebi_id, info) -> int:
        """Select auxiliary predicates for one class; returns how many were recorded."""
        ctx = self.class_context(chebi_id)
        query = f"{info['name']} {info['definition'] or ''}"
        candidates = self.retriever.retrieve(query, top_k=self.top_k) if len(self.retriever) else []
        prompt = self.build_user_prompt(chebi_id, info, candidates, ctx)

        parsed, raw, attempts = None, None, []
        try:
            parsed, raw, attempts = generate_one(
                prompt, self.system_prompt, self.model, self.selection_model, self.api_base
            )
        except Exception as e:
            attempts = getattr(e, "_chebilp_attempts", [])
            raise
        finally:
            # Log the exchange even if the request failed; rewritten below once resolved.
            log_path = self._write_log(chebi_id, info, prompt, parsed, raw, None, attempts)
            print(f"    logged full exchange to {log_path}")

        reused_stems: list[str] = []
        for i in parsed.reuse:
            if not 1 <= i <= len(candidates) or candidates[i - 1]["stem"] in reused_stems:
                continue
            reused_stems.append(candidates[i - 1]["stem"])
            print(f"    reused [{i}] {candidates[i - 1]['name']}")
        ctx["reused_stems"] = reused_stems

        blocks = [(f"chebi_{chebi_id}_block_{i}", self.to_source(item)) for i, item in enumerate(parsed.new)]
        self.prepare(blocks, ctx)

        new_stems: list[str] = []
        new_records: list[dict] = []
        rejected: list[dict] = []
        seen: set[str] = set()
        for item, (label, source) in zip(parsed.new, blocks):
            ok, reason = self.accept(source, label, ctx)
            if not ok:
                code = self.rejection_code(label, ctx)
                print(f"    rejected {item.name}" + (f" [{code}]" if code else "") + f": {reason}")
                record = {"name": item.name, "reason": reason}
                if code is not None:
                    record["code"] = code
                rejected.append(record)
                continue
            added = self.add_to_library(source, chebi_id)
            if added is None:
                rejected.append({"name": item.name, "reason": "invalid program"})
                continue
            stem, saved = added
            if saved.name in seen:
                continue
            seen.add(saved.name)
            new_stems.append(stem)
            new_records.append({"name": saved.name, "stem": stem, "reason": reason})
            self.retriever.add_entry(self.retriever_entry(stem, saved))
            print(f"    new {self.describe(saved, stem, reason)}")

        stems = reused_stems + new_stems
        set_class_predicates(chebi_id, stems, problem_dir=self.library_dir)

        selection = {"reused": reused_stems, "new": new_records, "rejected": rejected}
        self._write_log(chebi_id, info, prompt, parsed, raw, selection, attempts)
        return len(stems)

    def run(self, chebi_graph, chebi_ids) -> int:
        print(f"Generating auxiliary {self.noun}s for {len(chebi_ids)} class(es).")
        print(f"Building retrieval index over the shared {self.noun} library...")
        self.retriever = self.build_retriever()
        print(f"  {len(self.retriever)} {self.noun}(s) in the library.")

        done = set(load_class_map(self.library_dir))
        if done:
            print(f"  Resuming: {len(done)} class(es) already in class_map.json will be skipped.")

        total = 0
        for chebi_id in sort_labels_by_hierarchy(chebi_ids, chebi_graph):
            if str(chebi_id) in done:
                continue
            info = get_class_info(chebi_graph, chebi_id)
            print(f"CHEBI:{chebi_id} ({info['name']})...")
            try:
                n = self.generate_for_class(chebi_id, info)
                total += n
                print(f"  -> {n} auxiliary {self.noun}(s) recorded")
            except Exception as e:
                print(f"  -> Failed: {e}")
        print(f"Done. Recorded {total} auxiliary {self.noun} selection(s) across {len(chebi_ids)} class(es).")
        return total

    # --- logging -----------------------------------------------------------------

    def _write_log(self, chebi_id, info, prompt, parsed, raw, selection, attempts=None):
        attempts = attempts or []
        failed = [a for a in attempts if a.get("error")]
        costs = [a["cost"] for a in attempts if a.get("cost") is not None]
        total_cost = sum(costs) if costs else None
        log_path = get_aux_generation_log_path(chebi_id, base_dir=self.library_dir)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# Auxiliary-{self.noun} generation log — CHEBI:{chebi_id} ({info['name']})\n\n")
            f.write(f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- Model: {self.model}\n")
            if attempts:
                cost_str = f"${total_cost:.4f}" if total_cost is not None else "unknown"
                f.write(f"- Cost: {cost_str} across {len(attempts)} call(s), {len(failed)} reasked\n")
            f.write("\n")
            if selection is not None:
                f.write(self._format_selection(selection))
            if parsed is not None:
                f.write(f"## Model reasoning\n\n{parsed.reasoning}\n\n")
                f.write(self._format_output(parsed))
            f.write("## System prompt\n\n```\n" + self.system_prompt + "\n```\n\n")
            f.write("## User prompt\n\n```\n" + prompt + "\n```\n\n")
            if parsed is None and not failed:
                f.write("## Raw LLM response\n\n```json\n")
                f.write(raw if raw is not None else "(no response — request failed)")
                f.write("\n```\n\n")
            if failed:
                # Kept at the bottom: the resolved result is what matters; the reasks are
                # diagnostic context for why an extra call (or several) was needed.
                f.write(self._format_failed_attempts(failed))
        return log_path

    def _format_selection(self, selection) -> str:
        """Render the resolved selection: reused, accepted (with fire fraction), rejected."""
        parts = ["## Resolved selection\n\n"]
        parts.append(f"- Reused from library: {', '.join(selection['reused']) or '(none)'}\n")
        if selection["new"]:
            parts.append(f"- New {self.noun}s added:\n")
            for r in selection["new"]:
                parts.append(f"    - `{r['name']}` — {r['reason']}\n")
        else:
            parts.append(f"- New {self.noun}s added: (none)\n")
        if selection["rejected"]:
            parts.append("- Rejected:\n")
            for r in selection["rejected"]:
                code = f"**[{r['code']}]** " if r.get("code") else ""
                parts.append(f"    - `{r['name']}` — {code}{r['reason']}\n")
        else:
            parts.append("- Rejected: (none)\n")
        parts.append("\n")
        return "".join(parts)

    def _format_failed_attempts(self, failed) -> str:
        """Render reasked calls (malformed / schema-invalid output) at the log's tail."""
        parts = [f"## Failed attempts ({len(failed)} reasked)\n\n"]
        for i, a in enumerate(failed, 1):
            cost = a.get("cost")
            cost_str = f" — cost ${cost:.4f}" if cost is not None else ""
            parts.append(f"### Attempt {i}{cost_str}\n\n")
            parts.append(f"- Error: {a['error']}\n\n")
            raw = a.get("raw")
            parts.append("```json\n" + (raw if raw else "(no raw response captured)") + "\n```\n\n")
        return "".join(parts)

    def _format_output(self, parsed) -> str:
        """Render the model's parsed answer as readable Markdown."""
        parts = ["## Model output\n\n"]
        reuse = ", ".join(f"[{i}]" for i in parsed.reuse) or "(none)"
        parts.append(f"- Reuse candidates: {reuse}\n\n")
        if not parsed.new:
            parts.append(f"No new {self.noun}s proposed.\n\n")
            return "".join(parts)
        parts.append(f"### New {self.noun}s\n\n")
        for item in parsed.new:
            parts.append(f"#### `{item.name}`\n\n{item.description}\n\n")
            parts.append("```\n" + item.program.strip("\n") + "\n```\n\n")
        return "".join(parts)
