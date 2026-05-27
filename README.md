<<<<<<< Updated upstream
# chebILP

An Inductive Logic Programming (ILP) framework for classifying chemical compounds into [ChEBI](https://www.ebi.ac.uk/chebi/) classes. Rules are learned with [Popper](https://github.com/logic-and-learning-lab/Popper) and evaluated with [Clingo](https://potassco.org/clingo/) (Answer Set Programming).

---

## Installation

### Prerequesites

[SWI-Prolog](https://www.swi-prolog.org/Download.html) must be installed and on `PATH` (required by Popper).
Popper must be installed as well. You can either install the [latest version of Popper](https://github.com/logic-and-learning-lab/Popper) with
```
pip install https://github.com/logic-and-learning-lab/Popper
```
or a forked, slightly outdated version with
```
pip install https://github.com/sfluegel05/Popper
```
With the latter, you can use the `--mdl_weight_fn`, `--mdl_weight_fp` and `--mdl_weight_seize` options of the learn command.

### Core package

```bash
pip install chebILP
```

Extras:
- `pip install chebILP[explain]` adds `xclingo` and `Pillow` for the `explain` command
- `pip install chebILP[llm]` adds `anthropic`, `langsmith`, and `python-dotenv` for LLM-enhanced rule learning (`enhance_with_llms`, experimental)


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

**Step 1 — Download ChEBI data and build the dataset** (downloads `chebi.obo` and `chebi.sdf.gz`, builds cached graph and molecule files, selects label classes, and creates a train/val/test split):
```bash
python -m chebILP prepare_dataset \
  --chebi_version 248 \
  --min_pos_samples 25
```

This writes to `data/chebi_v248/`:
- `chebi_graph.pkl` — hierarchy graph (networkx DiGraph)
- `molecules.pkl` — molecule DataFrame (index = ChEBI ID)
- `min50/labels.txt` — selected class IDs (one per line)
- `min50/splits.csv` — molecule-level train/val/test split

**Step 2 — Build ILP example files** (positive/negative molecules per class):
```bash
python -m chebILP build_samples \
  --labels_file data/chebi_v248/ChEBI25_3_STAR/labels.txt \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/splits.csv \
  --chebi_graph_path data/chebi_v248/chebi_graph.pkl \
  --molecules_path data/chebi_v248/ChEBI25_3_STAR/molecules.pkl
```

**Step 3 — Build ILP background knowledge files** (molecule features as logic facts):
```bash
python -m chebILP build_bk \
  --labels_file data/chebi_v248/ChEBI25_3_STAR/labels.txt \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/splits.csv \
  --chebi_graph_path data/chebi_v248/chebi_graph.pkl \
  --molecules_path data/chebi_v28/ChEBI25_3_STAR/molecules.pkl
```

Steps 2 and 3 write files into `data/ilp_problems/` (one subdirectory per class). Available predicate sets: `atoms`, `chembl_fgs`, `chebi_fgs`, `chebi_fg_rules` and `chebi_fg_learned_rules`.

---

### 2. Learning ILP rules

Learn Prolog classification rules for each class using the examples and background knowledge from workflow 1. 
The learn function will create an updated bias file based on the `max_vars`, `max_body` and `max_clauses` parameters.

**Learn rules:**
```bash
python -m chebILP learn \
  --labels_file data/chebi_v248/ChEBI25_3_STAR/labels.txt \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/splits.csv \
  --timeout 60
```

Output is written to a timestamped directory `data/results/run_YYYYMMDD_HHMMSS/` containing `results.json` (one entry per class with the learned program and training score) and `config.yml`.

**Evaluate on test/validation set:**
```bash
python -m chebILP test \
  --run_to_evaluate data/results/run_20260101_120000 \
  --test_on test
```

**Optional: LLM-enhanced rules (experimental)**

To improve learned programs with an LLM (requires `ANTHROPIC_API_KEY` in `.env`):
```bash
python -m chebILP.enhance_with_llms \
  --input data/ilp_programs.csv \
  --output data/enhanced_run \
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
