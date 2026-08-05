import os
import re

from chebILP.utils import split_prolog_literals


def _patch_clingo_logger_decode() -> None:
    """Decode clingo logger messages with ``errors="replace"``.

    clingo's logger callback strict-decodes each message as UTF-8, and its
    ``onerror`` handler hard-exits the process (``os._exit``) on failure — so a
    single non-UTF-8 byte in a warning/info message kills the whole run and no
    ``try/except`` can catch it. A custom ``logger=`` does not help: the decode
    runs before the handler is called. Patching ``clingo.core._to_str`` makes the
    decode tolerant so grounding messages can never take the process down.
    """
    from clingo import core

    if getattr(core, "_chebilp_safe_to_str", False):
        return

    def _safe_to_str(c_str) -> str:
        return core._ffi.string(c_str).decode(errors="replace")

    core._to_str = _safe_to_str
    core._chebilp_safe_to_str = True


def filter_impossible_rules(rules: list[str], predicates_in_bk: list[str]):
    # for every predicate name in the rule body, check if it exists in background_facts
    predicates_in_bk = [p[0] for p in predicates_in_bk]
    rules_filtered = []
    for rule in rules:
        body = rule.split(":-")[1] if ":-" in rule else ""
        literals = split_prolog_literals(body)
        if any(literal.split("(")[0].strip() not in predicates_in_bk for literal in literals):
            continue
        rules_filtered.append(rule)
    print(f"Filtered out {len(rules) - len(rules_filtered)} impossible rules. Remaining rules: {len(rules_filtered)}")
    return rules_filtered

def ground_extensions(rules: list[str], background_facts: list[str], target_labels: list[str], timeout: float|None=None, message_sink: list|None=None) -> dict[str, list[tuple[str, ...]]]:
    """Ground ``rules`` over ``background_facts``; return each target's derived argument tuples.

    The target predicates may have any arity — the arguments are returned as they were
    derived and it is up to the caller to interpret them. Note that clingo treats ``p/1``
    and ``p/2`` as different predicates, so one name can yield tuples of differing width.

    Pass ``message_sink`` to collect clingo's diagnostics into it instead of letting them go
    to stderr — the caller can then report them once rather than once per grounding.
    """
    import clingo

    _patch_clingo_logger_decode()
    ctl = clingo.Control(logger=(lambda code, msg: message_sink.append(msg)) if message_sink is not None else None)
    ctl.add("base", [], "\n".join(background_facts))
    try:
        ctl.add("base", [], "\n".join(rules))
    except RuntimeError as e:
        print(f"Error parsing rules: {e}")
        print("Rules were:")
        for rule in rules:
            print(rule)
        raise e
    ctl.ground([("base", [])])

    wanted = set(target_labels)
    extensions: dict[str, list[tuple[str, ...]]] = {}
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def _collect(model):
        for atom in model.symbols(atoms=True):
            if atom.name not in wanted:
                continue
            key = (atom.name, tuple(str(a) for a in atom.arguments))
            if key in seen:
                continue
            seen.add(key)
            extensions.setdefault(atom.name, []).append(key[1])

    if timeout is None:
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                _collect(model)
    else:
        with ctl.solve(on_model=_collect, async_=True) as handle:
            finished = handle.wait(timeout)
            if not finished:
                handle.cancel()
                print(f"Clingo solving timed out after {timeout}s")
    return extensions


# How much address space a grounding child may claim ON TOP OF what it already inherited at
# fork. It is a budget, not an absolute ``RLIMIT_AS``: the auxiliary-rule generator loads
# sentence-transformers/torch before it ever grounds, and torch's reserved VA arenas put the
# parent's VSZ around 42 GiB while its RSS stays near 2 GiB. An absolute cap below that is
# worse than none — ``setrlimit`` accepts it, then every allocation in the child fails, so
# even a program that grounds in a millisecond comes back as a memory failure.
GROUNDING_MEMORY_BUDGET_BYTES = 4 * 1024**3

# Molecules per clingo instance. A clause whose variables are not all joined to the molecule
# grounds as a cross product over EVERY atom in the instance, so the cost of one bad rule is
# quadratic in the number of molecules ground together. Grounding a whole class at once (650+
# molecules, ~13k ring atoms) puts that at ~10^8 ground atoms; batching caps it near 10^6.
# For a well-formed program — every variable joined to its molecule — no clause can span two
# molecules, so the batch size does not affect the result at all.
GROUNDING_BATCH_SIZE = 25


def _address_space_limit(budget: int) -> int | None:
    """``RLIMIT_AS`` value leaving ``budget`` bytes of headroom above what is mapped now."""
    try:
        with open("/proc/self/statm") as f:
            mapped_pages = int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return mapped_pages * os.sysconf("SC_PAGE_SIZE") + budget


# "atom does not occur in any rule head" spans two lines, the atom on the second.
_UNDEFINED_ATOM_RE = re.compile(
    r"atom does not occur in any rule head:\s*([a-z_][A-Za-z0-9_]*)\s*(\(([^)]*)\))?", re.IGNORECASE
)


def _summarize_clingo_messages(per_group: list[list[str]]) -> None:
    """Print one line for what clingo said, however many groundings it said it in.

    Every batch re-reports the same diagnostics, so the raw stream is a handful of messages
    times the batch count. Undefined predicates are the signal worth keeping — a rule
    referencing something nothing defines has an empty body — but "undefined" is reported
    *per grounding*, and a batch whose molecules happen to contain no sulfur reports ``s/1``
    exactly like a genuine typo. Only a predicate missing from **every** batch is really
    undefined; the rest is molecule-to-molecule variation and is dropped. Other messages are
    printed once, deduplicated.
    """
    seen_in: dict[str, set[int]] = {}
    other: set[str] = set()
    for index, messages in enumerate(per_group):
        for message in messages:
            m = _UNDEFINED_ATOM_RE.search(message)
            if m is None:
                other.add(message.strip())
                continue
            arity = len(m.group(3).split(",")) if m.group(3) and m.group(3).strip() else 0
            seen_in.setdefault(f"{m.group(1)}/{arity}", set()).add(index)

    undefined = sorted(name for name, groups in seen_in.items() if len(groups) == len(per_group))
    if undefined:
        print(f"  clingo: {len(undefined)} predicate(s) referenced but never defined "
              f"(their rule bodies are empty): {', '.join(undefined)}")
    for message in sorted(other):
        print(f"  clingo: {message}")


def _ground_groups(rules, fact_groups, target_labels, timeout):
    """Ground ``rules`` over each group of facts separately, merging the derived tuples."""
    merged: dict[str, list[tuple[str, ...]]] = {}
    per_group: list[list[str]] = []
    for group in fact_groups:
        messages: list = []
        per_group.append(messages)
        derived = ground_extensions(rules, group, target_labels, timeout=timeout, message_sink=messages)
        for name, tuples in derived.items():
            merged.setdefault(name, []).extend(tuples)
    _summarize_clingo_messages(per_group)
    return merged


def _ground_extensions_child(queue, rules, fact_groups, target_labels, timeout, memory_budget):
    """Child-process body: cap address space, then ground and return the result via ``queue``."""
    try:
        import resource

        limit = _address_space_limit(memory_budget)
        if limit is not None:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError):
        pass  # cap unavailable: the parent's wall-clock ceiling is then the only backstop
    try:
        queue.put(("ok", _ground_groups(rules, fact_groups, target_labels, timeout)))
    except MemoryError as e:
        # Reporting the failure allocates, and under an exhausted RLIMIT_AS that raises a
        # second MemoryError which kills the child before it can say why — leaving only its
        # traceback on the shared stderr. Clearing the traceback releases the grounding
        # frames it pins, and with them clingo's Control and the partial result.
        e.__traceback__ = None
        queue.put(("err", "grounding exceeded the memory limit"))
    except Exception as e:  # report any failure back rather than dying silently
        message = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
        e.__traceback__ = None
        queue.put(("err", message))


def ground_extensions_isolated(
    rules: list[str],
    fact_groups: list[list[str]],
    target_labels: list[str],
    timeout: float | None = None,
    total_timeout: float | None = None,
    memory_budget: int = GROUNDING_MEMORY_BUDGET_BYTES,
) -> dict[str, list[tuple[str, ...]]]:
    """Ground ``rules`` over each group in ``fact_groups``, in a memory-capped forked child.

    Each group is a separate clingo instance and the derived tuples are merged, so a clause
    can never join facts from two different groups. Callers with a single flat fact list
    pass ``[facts]``; :func:`derive_rule_extensions` splits by molecule instead, which is
    what keeps an unjoined-variable clause from grounding as a class-wide cross product.

    A runaway grounding dies inside the child (memory cap or, as a last resort, the OOM
    killer) instead of taking the whole process down uncatchably. On any child failure —
    memory cap hit, killing signal, timeout, or grounding error — this raises
    ``RuntimeError`` so the caller can reject the offending programs and continue.

    ``timeout`` is the per-group solve budget; ``total_timeout`` is the wall-clock ceiling
    for the whole child, and defaults to ``timeout + 60`` for a single group. Pass one of
    them: without any ceiling a child that wedges rather than dying leaves the parent
    polling for ever.

    Falls back to an in-process call where forking is unavailable (e.g. Windows).
    """
    import multiprocessing
    import queue as queue_mod
    import time

    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        return _ground_groups(rules, fact_groups, target_labels, timeout)

    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_ground_extensions_child,
        args=(result_queue, rules, fact_groups, target_labels, timeout, memory_budget),
    )
    proc.start()

    # Wall-clock ceiling for the whole child, distinct from the per-group ``timeout``.
    join_timeout = total_timeout
    if join_timeout is None and timeout is not None:
        join_timeout = timeout * max(len(fact_groups), 1) + 60
    deadline = None if join_timeout is None else time.monotonic() + join_timeout

    # Drain the queue while the child runs: a large result can exceed the pipe buffer, so
    # the child blocks on flush until we read — joining first would deadlock. Polling also
    # lets us notice a child that was killed (OOM/SIGKILL) without producing a result.
    while True:
        try:
            status, payload = result_queue.get(timeout=0.5)
            break
        except queue_mod.Empty:
            if not proc.is_alive():
                proc.join()
                raise RuntimeError(f"grounding died before returning (exit code {proc.exitcode})")
            if deadline is not None and time.monotonic() > deadline:
                proc.terminate()
                proc.join()
                raise RuntimeError(f"grounding exceeded the {join_timeout}s time limit")

    proc.join()
    if status == "ok":
        return payload
    raise RuntimeError(payload)


def evaluate_with_clingo(rules: list[str], background_facts: list[str], target_labels: list[str], examples: list, predicates_in_bk: list[str]|None=None, timeout: float|None=None):
    """Which of ``examples`` each target predicate holds for, matching ``label(example)``.

    Only unary extensions are considered; for predicates of other arities use
    :func:`ground_extensions`.
    """
    if predicates_in_bk is not None:
        rules = filter_impossible_rules(rules, predicates_in_bk)
    extensions = ground_extensions(rules, background_facts, target_labels, timeout=timeout)

    positives = dict()
    for target_label in target_labels:
        derived = {args[0] for args in extensions.get(target_label, []) if len(args) == 1}
        hits = [example for example in examples if str(example) in derived]
        if hits:
            positives[target_label] = hits

    return positives


def run_ilp_validation_clingo(chebi_id: str, prog_str: str, exs_file: str, bk_file: str, return_details: bool = False):
    with open(exs_file, "r") as f:
        # each line is of the form "pos(molecule_id)." or "neg(molecule_id)."
        # extract molecule_id as an integer and ignore pos/neg for now (we'll compute confusion matrix later)
        examples = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("%")]
        examples_posneg = [line.strip().split("(")[0].strip() for line in examples]
        examples_ids = [line.strip().split("(")[-1].split(")")[0].strip() for line in examples]
    with open(bk_file, "r") as f:
        background_facts = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("%")]

    target_labels = set()
    for line in prog_str.split("\n"):
        if ":-" in line:
            head = line.split(":-")[0].strip()
            if "(" in head:
                target_labels.add(head.split("(")[0].strip())

    positives = evaluate_with_clingo(prog_str.split("\n"), background_facts, list(target_labels), examples_ids)
    if f"chebi_{chebi_id}" not in positives:
        positives = []
    else:
        positives = [ex for ex in examples_ids if ex in positives[f"chebi_{chebi_id}"]]
    tps, fps, tns, fns = 0, 0, 0, 0
    details = []
    for ex_id, posneg in zip(examples_ids, examples_posneg):
        predicted_pos = ex_id in positives
        true_pos = posneg.startswith("pos")
        if predicted_pos:
            if true_pos:
                tps += 1
                outcome = "TP"
            else:
                fps += 1
                outcome = "FP"
        else:
            if true_pos:
                fns += 1
                outcome = "FN"
            else:
                tns += 1
                outcome = "TN"
        details.append({"id": ex_id, "true_label": "pos" if true_pos else "neg", "outcome": outcome})

    conf_matrix = {"TP": tps, "FP": fps, "TN": tns, "FN": fns}
    if return_details:
        return conf_matrix, details
    return conf_matrix

if __name__ == "__main__":
    # Example usage
    rule = "target_c(X) :- has_atom(X, A), c(A)."
    rule += "target_o(X) :- has_atom(X, A), o(A)."
    background_facts = "has_atom(molecule1, carbon). c(carbon). has_atom(molecule2, oxygen). o(oxygen)."
    examples = ["molecule1", "molecule2"]
    target_labels = ["target_c", "target_o"]

    positives = evaluate_with_clingo(rule, background_facts, target_labels, examples)
    print("Positives:", positives)
