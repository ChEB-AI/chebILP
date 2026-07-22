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
- `pip install chebILP[llm]` adds `litellm` (multi-provider access for auxiliary-predicate generation), `anthropic`, `langsmith`, and `python-dotenv` for LLM-enhanced rule learning (`enhance_with_llms`, experimental)


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
- `ChEBI25_3_STAR/molecules.pkl` — molecule DataFrame (index = ChEBI ID)
- `ChEBI25_3_STAR/labels.txt` — selected class IDs (one per line)
- `ChEBI25_3_STAR/splits.csv` — molecule-level train/val/test split

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
  --molecules_path data/chebi_v248/ChEBI25_3_STAR/molecules.pkl
```

Steps 2 and 3 write files into `data/ilp_problems/` (one subdirectory per class). Available predicate sets: `atoms`, `chembl_fgs`, `chebi_fgs`, `chebi_fg_rules`, `chebi_fg_learned_rules` and `llm_generated_fgs`.

**Optional: LLM-generated auxiliary predicates (experimental)**

The `llm_generated_fgs` predicate set augments the plain `atoms` predicates with
class-specific *auxiliary predicates* invented by an LLM — either shortcuts for
recurring functional groups or concepts that are hard to express with the atom/bond
predicates (e.g. "molecule has exactly 40 carbons"). Predicates live in a shared
library — one `programs/<aux_name>.py` file per distinct program (RDKit `Mol` →
extension) plus a `class_map.json` recording which predicates each class uses — kept
separate from the ILP problem directory (default `data/llm_generated_predicates`).

Generate them before `build_bk`:
```bash
python -m chebILP.generate_auxiliary_predicates \
  --labels_file data/chebi_v248/ChEBI25_3_STAR/labels.txt \
  --chebi_version 248 \
  --n_predicates 8 \
  --predicate_dir data/llm_generated_predicates
```
The model provider is chosen with `--model provider/name` (via [LiteLLM](https://github.com/BerriAI/litellm)); it defaults to
`anthropic/claude-haiku-4-5`. Other examples: `openai/gpt-4o`, `gemini/gemini-2.5-pro`,
`ollama/llama3.1`, or `hosted_vllm/<name> --api_base http://localhost:8000/v1` for a
self-hosted / OpenAI-compatible server. The model must support structured outputs. The
provider's API key is read from `.env` / the environment under its standard name
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, ...). The same `--model` /
`--api_base` flags apply to `python -m chebILP.generate_auxiliary_rules`.

Then run `build_bk` with `--predicate_set llm_generated_fgs --predicate_dir <library>`;
the predicates a class uses are merged into its background knowledge (predicate names are
`aux_`-prefixed). `--predicate_dir` defaults to `data/llm_generated_predicates` and
`build_bk` errors if no library exists there. Classes with no recorded programs fall back
to plain atom predicates.

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
  --run_dir data/results/run_20260101_120000 \
  --predict_on validation \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/splits.csv \
  --chebi_version 248
```

This writes `full_val_preds.npy` and `full_val_preds_metadata.json` into the run directory. Repeat with `--predict_on test` for the test split.

**Step 2 — Model selection and ILP tensor assembly:**
```bash
python -m chebILP ensemble_construct \
  --chebi_split data/chebi_v248/ChEBI25_3_STAR/splits.csv \
  --dl_val_preds_npy data/preds/val_preds.npy \
  --dl_val_preds_meta data/preds/val_preds_metadata.json \
  --ilp_val_runs data/results_val/run_A data/results_val/run_B \
  --labels_file data/chebi_v248/ChEBI25_3_STAR/labels.txt \
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

**Translate a rule to natural language (global explanation):**
```bash
python -m chebILP rule_to_nl --rule "chebi_15734(V0) :- has_atom(V0,V1), c(V1), has_2_hs(V1), bSINGLE(V1,V2), o(V2), has_1_hs(V2)." --chebi_graph_path data/chebi_v248/chebi_graph.pkl
```

**Explain why a molecule satisfies a rule (local explanation):**
```bash
python -m chebILP explain \
  --smiles "CCO" \
  --rule "chebi_15734(V0) :- has_atom(V0,V1), c(V1), has_2_hs(V1), bSINGLE(V1,V2), o(V2), has_1_hs(V2)." \
  --chebi_graph_path data/chebi_v248/chebi_graph.pkl \
  --output explanation.png
```
