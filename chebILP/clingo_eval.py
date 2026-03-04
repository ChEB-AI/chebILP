from chebILP.ilp_classifier import split_prolog_literals

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

def evaluate_with_clingo(rules: list[str], background_facts: list[str], target_labels: list[int], examples: list, predicates_in_bk: list[str]|None=None):
    import clingo

    if predicates_in_bk is not None:
        rules = filter_impossible_rules(rules, predicates_in_bk)
    ctl = clingo.Control()
    ctl.add("base", [], "\n".join(background_facts))
    ctl.add("base", [], "\n".join(rules))
    ctl.ground([("base", [])])

    atoms = set()
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            atoms.update(str(atom) for atom in model.symbols(atoms=True))
    positives = dict()
    for target_label in target_labels:
        for example in examples:
            if f"{target_label}({example})" in atoms:
                if target_label not in positives:
                    positives[target_label] = []
                positives[target_label].append(example)

    return positives


def run_ilp_validation_clingo(chebi_id: str, prog_str: str, exs_file: str, bk_file: str):
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
    for ex_id, posneg in zip(examples_ids, examples_posneg):
        if ex_id in positives:
            if posneg.startswith("pos"):
                tps += 1
            else:
                fps += 1
        else:
            if posneg.startswith("neg"):
                tns += 1
            else:
                fns += 1
    return {
        "TP": tps,
        "FP": fps,
        "TN": tns,
        "FN": fns
    }

if __name__ == "__main__":
    # Example usage
    rule = "target_c(X) :- has_atom(X, A), c(A)."
    rule += "target_o(X) :- has_atom(X, A), o(A)."
    background_facts = "has_atom(molecule1, carbon). c(carbon). has_atom(molecule2, oxygen). o(oxygen)."
    examples = ["molecule1", "molecule2"]
    target_labels = ["target_c", "target_o"]

    positives = evaluate_with_clingo(rule, background_facts, target_labels, examples)
    print("Positives:", positives)
