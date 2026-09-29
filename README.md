# chebILP

Inductive Logic Programming (ILP) for classifying chemical compounds into [ChEBI](https://www.ebi.ac.uk/chebi/) classes.
Molecules are translated into logic facts, rules are learned per class with [Popper](https://github.com/logic-and-learning-lab/Popper) or Aleph,
and evaluated with [Clingo](https://potassco.org/clingo/). Optionally, an LLM invents auxiliary predicates (ASP rules) that Popper can use in its rules.

## Installation

Requirements:
- Python ≥ 3.11
- [SWI-Prolog](https://www.swi-prolog.org/Download.html) on `PATH` (used by Popper and Aleph)
- Popper: `pip install git+https://github.com/logic-and-learning-lab/Popper`, or the fork
  `pip install git+https://github.com/sfluegel05/Popper`, which adds the `--mdl_weight_*` options of `learn`

Then install chebILP from the repository root:

```bash
pip install -e .            # core
pip install -e ".[llm]"     # + LLM predicate invention
pip install -e ".[explain]" # + the explain command (xclingo, Pillow)
```

Aleph ships with the package. List all commands with `python -m chebILP -h`, and options with `python -m chebILP <command> -h`.

## Workflow

The examples use ChEBI version 251; adjust `-v` / paths as needed.

### 1. Create a dataset

```bash
python -m chebILP prepare_dataset -v 251
```

This downloads ChEBI and writes `data/chebi_v251/`: the class graph (`chebi_graph.pkl`) and, under `ChEBI25_3_STAR/`,
the molecules (`molecules.pkl`), the selected classes (`labels.txt`) and a train/validation/test split (`splits.csv`).
Use `-2` to include 2-star entries and `--min_pos_samples` to change the class-size threshold.

### 2. Build samples and background knowledge

```bash
LABELS=data/chebi_v251/ChEBI25_3_STAR/labels.txt

python -m chebILP build_samples --labels_file $LABELS -v 251
python -m chebILP build_bk      --labels_file $LABELS -v 251 --predicate_set atoms
```

`build_samples` picks positive and negative molecules per class; `build_bk` writes the molecules as logic facts.
Both write to `data/ilp_problems/chebi_<id>/`. Main predicate sets (`--predicate_set`):

| Set | Content |
|---|---|
| `atoms` (default) | atoms, elements, charges, hydrogens, bonds, ring membership |
| `chembl_fgs` | `atoms` + RDKit ChEMBL functional-group alerts |
| `efg`, `efg_atoms` | `atoms` + Extended Functional Groups (per molecule / anchored at atoms) |
| `llm_generated_rules` | `atoms` + LLM-invented auxiliary predicates (see step 3) |

### 3. Optional: LLM predicate invention

An LLM proposes, for each class, a few auxiliary predicates written as ASP rules, plus a hypothesis for the class.
Predicates are stored in a shared library and reused across classes. Programs that do not parse or cannot be grounded safely are rejected.

```bash
python -m chebILP.predicate_generation.generate_auxiliary_rules \
  --labels_file $LABELS --chebi_version 251 \
  --molecules_path data/chebi_v251/ChEBI25_3_STAR/molecules.pkl \
  --model claude-opus-5 \
  --predicate_dir data/llm_generated_rules \
  --seed_predicates efg
```

- `--model`: a bare Claude id runs through the local `claude` CLI and bills the logged-in Claude subscription.
  A `provider/name` id (e.g. `openai/gpt-4o`) uses an OpenAI-compatible endpoint set by `OPENAI_API_BASE` and `OPENAI_API_KEY`.
- `--seed_predicates {none,chembl_fgs,efg}` pre-fills the library with functional groups the LLM can reuse.
- Each class costs one LLM call. The run can be resumed; classes already in the library are skipped.

Then rebuild the background knowledge with the new predicates:

```bash
python -m chebILP build_bk --labels_file $LABELS -v 251 \
  --predicate_set llm_generated_rules --predicate_dir data/llm_generated_rules --computed_facts
```

### 4. Learn rules with Popper or Aleph

```bash
python -m chebILP learn --labels_file $LABELS --predicate_set atoms --timeout 120            # Popper
python -m chebILP learn --labels_file $LABELS --predicate_set atoms --timeout 120 --tool aleph
```

Results go to `data/results/run_<timestamp>/` (`results.json` with one learned program per class, `config.yml`, `run.log`).
With `llm_generated_rules`, `--seed_hypothesis` starts Popper's search from the LLM's class hypothesis and
`--heuristic_guidance` steers it toward the predicates that hypothesis uses (both need `--predicate_dir`).

Evaluate a run on the validation or test split:

```bash
python -m chebILP test --run_to_evaluate data/results/run_<timestamp> --test_on test
```

## Further commands

- `predict` — apply a rule file to new SMILES.
- `rule_to_nl` / `explain` — translate a rule into natural language, or show why a molecule satisfies it (needs `[explain]`).
- `build_ilp_preds_for_ensemble`, `ensemble_construct`, `ensemble_aggregate` — combine ILP rules with a deep-learning
  model: per leaf class, the better of ILP and DL on the validation set is used, and ILP predictions are gated by the class hierarchy.
