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

def ground_extensions(rules: list[str], background_facts: list[str], target_labels: list[str], timeout: float|None=None) -> dict[str, list[tuple[str, ...]]]:
    """Ground ``rules`` over ``background_facts``; return each target's derived argument tuples.

    The target predicates may have any arity — the arguments are returned as they were
    derived and it is up to the caller to interpret them. Note that clingo treats ``p/1``
    and ``p/2`` as different predicates, so one name can yield tuples of differing width.
    """
    import clingo

    _patch_clingo_logger_decode()
    ctl = clingo.Control()
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
