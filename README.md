# chebILP

An Inductive Logic Programming (ILP) framework for classifying chemical compounds into [ChEBI](https://www.ebi.ac.uk/chebi/) classes. Rules are learned with [Popper](https://github.com/logic-and-learning-lab/Popper) and evaluated with [Clingo](https://potassco.org/clingo/) (Answer Set Programming).

---

## Installation

**Prerequisites:** [SWI-Prolog](https://www.swi-prolog.org/Download.html) must be installed and on `PATH` (required by Popper and `janus-swi`).

Install the core package:
```bash
pip install ".[explain,llm]"
```

- `explain` adds `xclingo` and `Pillow` for the `explain` command
- `llm` adds `anthropic`, `langsmith`, and `python-dotenv` for LLM-enhanced rule learning (`enhance_with_llms`)

The `prepare_dl_preds` utility (one-time DL tensor extraction) additionally requires `torch`, which must be installed separately in an environment that has the DL model checkpoint.

## Usage
To get a list of available commands, run
```bash
python -m chebILP -h
```
To get help for a specific command, run
```bash
python -m chebILP {command} -h
```

## Workflows

### 1. Generating new data

An ILP dataset for ChEBI version 248 is available on [HuggingFace](https://huggingface.co/datasets/chebai/ChEBI25-3STAR-ILP). However, you can also create your own dataset.

Prepare training examples and background knowledge for a set of ChEBI classes.

**Build example files** (positive/negative molecules per class):
```bash
python -m chebILP build_samples \
  --labels_file data/labels.txt \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/processed/splits.csv \
  --chebi_version 248 \
  --predicate_set atoms \
  --min_pos_samples 25 --max_pos_samples 200 \
  --min_neg_samples 25 --max_neg_samples 200
```

**Build background knowledge files** (molecule features as logic facts):
```bash
python -m chebILP build_bk \
  --labels_file data/labels.txt \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/processed/splits.csv \
  --chebi_version 248 \
  --predicate_set atoms
```

Both commands write files into `data/ilp_problems/` (one subdirectory per class). `labels.txt` contains one ChEBI ID per line. Available predicate sets: `atoms`, `atoms_bonds`, `atoms_bonds_stereo`.

---

### 2. Learning ILP rules

Learn Prolog classification rules for each class using the examples and background knowledge from workflow 1.

**Learn rules:**
```bash
python -m chebILP learn \
  --labels_file data/labels.txt \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/processed/splits.csv \
  --chebi_version 248 \
  --predicate_set atoms \
  --timeout 60
```

Output is written to a timestamped directory `data/results/run_YYYYMMDD_HHMMSS/` containing `results.json` (one entry per class with the learned program and training score) and `config.yml`.

**Evaluate on test/validation set:**
```bash
python -m chebILP test \
  --run_to_evaluate data/results/run_20260101_120000 \
  --test_on test
```

**Optional: LLM-enhanced rules**

To improve learned programs with an LLM (requires `ANTHROPIC_API_KEY` in `.env`):
```bash
python -m chebILP.enhance_with_llms \
  --input data/enhance_with_llms/best_ilp_programs_for_leaves.csv \
  --output data/enhance_with_llms/enhanced_run \
  --chebi_version 248
```

Input CSV must have columns `chebi_id`, `program`, `run_name`. The output directory is readable by the `test` command.

---

### 3. Building an ensemble (ILP + DL)

Combine ILP rules with a deep learning (DL) model for hierarchical multi-label classification. The ensemble uses DL predictions for non-leaf classes and selects either ILP or DL for each leaf class based on validation F1.

**Step 1 — Build full ILP prediction tensors** (run once per ILP run, for the validation and/or test split):
```bash
python -m chebILP build_ilp_preds_for_ensemble \
  --run_dir data/results_val/run_20260101_120000 \
  --predict_on validation \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/processed/splits.csv \
  --chebi_version 248
```

This writes `full_val_preds.npy` and `full_val_preds_metadata.json` into the run directory. Repeat with `--predict_on test` for the test split.

**Step 2 — Model selection and ILP tensor assembly:**
```bash
python -m chebILP ensemble_construct \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/processed/splits.csv \
  --dl_val_preds_npy data/preds/val_preds.npy \
  --dl_val_preds_meta data/preds/val_preds_metadata.json \
  --ilp_val_runs data/results_val/run_A data/results_val/run_B \
  --label_stats data/chebi_v248/ChEBI25_3_STAR/processed/class_stats.csv \
  --predict_on test \
  --output data/ensemble_predictions/ensemble
```

For each leaf class, selects the ILP run whose ensemble F1 (ILP prediction AND all DL parent predictions >= 0.5) is highest; falls back to DL if no ILP run beats it. Outputs:
- `ensemble_trusted_models.csv` — which model is used per class
- `ensemble_ilp_preds.npy` + `ensemble_ilp_preds_metadata.json` — ILP tensor for the target split

**Step 3 — Aggregate into final predictions:**
```bash
python -m chebILP ensemble_aggregate \
  --dl_preds_npy data/preds/test_preds.npy \
  --dl_preds_meta data/preds/test_preds_metadata.json \
  --ilp_preds_npy data/ensemble_predictions/ensemble_ilp_preds.npy \
  --ilp_preds_meta data/ensemble_predictions/ensemble_ilp_preds_metadata.json \
  --trusted_models data/ensemble_predictions/ensemble_trusted_models.csv \
  --label_stats data/chebi_v248/ChEBI25_3_STAR/processed/class_stats.csv \
  --output data/ensemble_predictions/final_predictions.npy
```

DL predictions propagate freely through the class hierarchy; ILP and always-positive classes only predict a class if all label-set parents are already predicted positive. Output is a boolean NumPy array with a matching `_metadata.json`.

---

## Other utilities

**Translate a rule to natural language:**
```bash
python -m chebILP rule_to_nl --rule_file my_rule.pl --class_parents data/class_parents.json
```

**Explain why a molecule satisfies a rule:**
```bash
python -m chebILP explain \
  --smiles "CCO" \
  --rule_file my_rule.pl \
  --label_parents_json data/class_parents.json \
  --output explanation.png
```
