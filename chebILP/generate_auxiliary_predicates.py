# Pipeline: for each ChEBI class, ask an LLM to invent auxiliary predicates that
# help ILP tell the class apart from its siblings. Each auxiliary predicate is a
# small self-contained Python program mapping an RDKit Mol to a predicate
# extension (see chebILP.auxiliary_predicates for the contract). Programs are
# written to data/ilp_problems/chebi_{id}/auxiliary_predicates/ so that
# `build_bk --predicate_set llm_generated_fgs` can merge them into the BK.
#
# Input to the LLM: the ChEBI class name/definition, its superclasses, and the
# predicates already available from mol_to_fol_atoms (so it proposes
# complementary properties rather than duplicating existing ones).
#
# Usage:
#   python -m chebILP.generate_auxiliary_predicates \
#       --labels_file labels.txt \
#       --chebi_version 248 \
#       --n_predicates 8

import argparse
import os
import re
import time

import anthropic
from dotenv import load_dotenv
from rdkit import Chem

# Import from submodules rather than the package root: chebi_utils/__init__.py
# pulls in dataset_builder -> pandas, a heavy import this script does not need
# (and which is painfully slow when the venv lives on a WSL /mnt/c mount).
from chebi_utils.obo_extractor import build_chebi_graph
from chebi_utils.downloader import download_chebi_obo
from chebILP.auxiliary_predicates import load_program_source
from chebILP.ilp_path_manager import get_aux_predicates_dir


# Reuse the exact predicate reference the enhancement pipeline shows the model,
# so it knows what mol_to_fol_atoms already provides and avoids duplicating it.
_EXISTING_PREDICATES = """\
- has_atom(Molecule, Atom): Molecule contains Atom
- c(Atom), n(Atom), o(Atom), s(Atom), p(Atom), cl(Atom), br(Atom), f(Atom), i(Atom), se(Atom), ...: Atom type (element symbol, lowercase)
- charge0/charge_p/charge_n/charge1/charge_m1(Atom): formal charge of Atom
- has_X_hs(Atom) / has_at_least_X_hs(Atom): attached-hydrogen counts
- cip_code_R(Atom), cip_code_S(Atom): R/S CIP stereochemistry
- bSINGLE/bDOUBLE/bTRIPLE/bAROMATIC(Atom1, Atom2): bond type
- bSTEREOCIS/bSTEREOTRANS(Atom1, Atom2): cis/trans bond stereochemistry
- has_bond_to(Atom1, Atom2): any bond between two atoms
- in_ring(Atom), in_ringN(Atom), ringN(A1..AN): ring membership / N-membered rings (N up to 8)
- net_charge_positive/negative/neutral(Molecule), aromatic/aliphatic(Molecule): molecule-level
- steroid_1..steroid_17(Atom): atom at a steroid-nucleus position\
"""

_SYSTEM_PROMPT = """\
You are an expert in cheminformatics, RDKit, and Inductive Logic Programming (ILP).
Your task is to invent AUXILIARY PREDICATES that help an ILP system distinguish a
ChEBI chemical class from its sibling classes under the same parent.

An auxiliary predicate is a short, self-contained Python program that maps an RDKit
Mol object to a predicate extension. Two kinds are useful:
  1. SHORTCUTS: functional groups / substructures that recur in this class
     (e.g. carboxylic acid, glycosidic bond) which would otherwise need many
     atom/bond literals to express.
  2. CONCEPTS: properties that are hard or impossible to express with the existing
     atom/bond predicates (e.g. "molecule has exactly 40 carbon atoms",
     "molecular weight above 500", "contains at least 3 rings").

Each program MUST follow this exact contract:

    \"\"\"<one short line describing what the predicate captures>\"\"\"
    PREDICATE_NAME = "snake_case_name"
    KIND = "atom_unary"   # one of: "atom_unary", "atom_binary", "molecule"

    def extension(mol):
        # mol is an RDKit Mol object (already sanitized).
        # KIND == "molecule"    -> return a bool (does the property hold?)
        # KIND == "atom_unary"  -> return a list of 0-based atom indices
        # KIND == "atom_binary" -> return a list of (i, j) 0-based atom-index pairs
        ...

Rules:
- Use only the RDKit API. `Chem`, `rdMolDescriptors`, and `Descriptors` are already
  in scope; you may also `from rdkit ... import` inside the function if needed.
- Return atom INDICES (mol.GetAtomWithIdx(i)), never RDKit atom objects.
- Never raise; return an empty list / False when the property does not apply.
- Keep each program to a single predicate. Do not print or read files.
- Prefer predicates that are TRUE for many molecules of the target class and
  FALSE for its siblings.\
"""


def _get_class_info(chebi_graph, chebi_id: str) -> dict:
    node_data = dict(chebi_graph.nodes.get(chebi_id, {}))
    parents = []
    for pid in chebi_graph.successors(chebi_id):
        pdata = dict(chebi_graph.nodes.get(pid, {}))
        parents.append(
            {"id": pid, "name": pdata.get("name", f"CHEBI:{pid}"), "definition": pdata.get("definition")}
        )
    return {
        "name": node_data.get("name", f"CHEBI:{chebi_id}"),
        "definition": node_data.get("definition"),
        "parents": parents,
    }


def _build_prompt(chebi_id: str, info: dict, n_predicates: int) -> str:
    parent_str = ""
    if info["parents"]:
        parts = "\n".join(
            f"  - {p['name']} (CHEBI:{p['id']}): {p['definition'] or 'no definition'}"
            for p in info["parents"]
        )
        parent_str = f"Superclass(es):\n{parts}\n"

    definition_str = f"Definition: {info['definition']}\n" if info["definition"] else ""

    return f"""\
Target class: {info['name']} (CHEBI:{chebi_id})
{definition_str}{parent_str}
Predicates ALREADY available (do NOT duplicate these):
{_EXISTING_PREDICATES}

Propose up to {n_predicates} auxiliary predicates that would help distinguish
"{info['name']}" from its sibling classes. Favour a mix of shortcut substructures
and hard-to-express global concepts grounded in the definition above.

Output ONLY the Python programs, each in its own fenced code block:

```python
\"\"\"...\"\"\"
PREDICATE_NAME = "..."
KIND = "..."

def extension(mol):
    ...
```

No prose before, between, or after the code blocks.
"""


# A few structurally diverse molecules used to smoke-test generated programs.
_TEST_SMILES = [
    "CCO",  # ethanol
    "c1ccccc1",  # benzene
    "CC(=O)O",  # acetic acid
    "OC(=O)CCC(N)C(=O)O",  # glutamic acid-ish
    "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",  # a steroid
]
_TEST_MOLS = [m for m in (Chem.MolFromSmiles(s) for s in _TEST_SMILES) if m is not None]


def _parse_code_blocks(text: str) -> list[str]:
    """Extract the body of every ```python ...``` fenced block (falling back to bare ```)."""
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]


def _validate(source: str, source_label: str) -> bool:
    """Compile a program and run it against the smoke-test molecules."""
    pred = load_program_source(source, source_file=source_label)
    if pred is None:
        return False
    for mol in _TEST_MOLS:
        try:
            result = pred.fn(mol)
        except Exception as e:
            print(f"    rejected {pred.name}: crashed on a test molecule ({e})")
            return False
        if pred.kind == "molecule":
            bool(result)
        else:
            try:
                list(result)
            except TypeError:
                print(f"    rejected {pred.name}: {pred.kind} did not return an iterable")
                return False
    return True


def _generate_one(client, prompt, model, max_retries=5) -> str:
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"  Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)
    raise last_exc


def _write_generation_log(aux_dir, chebi_id, info, model, prompt, raw):
    """Persist the full LLM exchange (system + user prompt + raw response) for one class."""
    log_path = os.path.join(aux_dir, "generation_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Auxiliary-predicate generation log — CHEBI:{chebi_id} ({info['name']})\n\n")
        f.write(f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Model: {model}\n\n")
        f.write("## System prompt\n\n```\n")
        f.write(_SYSTEM_PROMPT)
        f.write("\n```\n\n## User prompt\n\n```\n")
        f.write(prompt)
        f.write("\n```\n\n## Raw LLM response\n\n")
        f.write(raw if raw is not None else "(no response — request failed)")
        f.write("\n")
    return log_path


def generate_for_class(client, chebi_id, info, problem_dir, model, n_predicates) -> int:
    """Generate, validate and persist auxiliary predicates for one ChEBI class.

    Also writes ``generation_log.md`` (the full prompt + raw response) into the
    class's ``auxiliary_predicates/`` folder for later inspection.

    Returns the number of programs successfully written.
    """
    aux_dir = get_aux_predicates_dir(chebi_id, base_dir=problem_dir)
    prompt = _build_prompt(chebi_id, info, n_predicates)

    raw = None
    try:
        raw = _generate_one(client, prompt, model)
    finally:
        # Log the exchange even if the request ultimately failed.
        log_path = _write_generation_log(aux_dir, chebi_id, info, model, prompt, raw)
        print(f"    logged full exchange to {log_path}")

    blocks = _parse_code_blocks(raw)
    if not blocks:
        print(f"  -> No code blocks parsed from LLM response for CHEBI:{chebi_id}")
        return 0

    # Clear any previous generation so stale programs don't linger.
    for existing in os.listdir(aux_dir):
        if existing.endswith(".py"):
            os.remove(os.path.join(aux_dir, existing))

    written = 0
    seen_names: set[str] = set()
    for i, block in enumerate(blocks):
        label = f"chebi_{chebi_id}_block_{i}"
        if not _validate(block, label):
            continue
        pred = load_program_source(block, source_file=label)
        if pred is None or pred.name in seen_names:
            continue
        seen_names.add(pred.name)
        out_path = os.path.join(aux_dir, f"{pred.name}.py")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(block.rstrip() + "\n")
        written += 1
        print(f"    wrote {pred.name} ({pred.kind}): {pred.description}")

    return written


def main():
    parser = argparse.ArgumentParser(description="Generate LLM auxiliary predicates for ChEBI classes.")
    parser.add_argument("--labels_file", required=True, help="File with one ChEBI ID per line.")
    parser.add_argument("--chebi_version", type=int, default=248)
    parser.add_argument("--problem_dir", default=None, help="Base ILP problems directory.")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model to use.")
    parser.add_argument("--n_predicates", type=int, default=8, help="Target number of predicates per class.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Add it to your .env file.")
    client = anthropic.Anthropic(api_key=api_key)

    data_dir = os.path.join("data", f"chebi_v{args.chebi_version}")
    obo_path = os.path.join(data_dir, "raw", "chebi.obo")
    with open(os.path.join(data_dir, "chebi_graph.pkl"), "rb") as _f:
        import pickle as _pickle
        chebi_graph = _pickle.load(_f)


    with open(args.labels_file) as f:
        chebi_ids = [line.strip() for line in f if line.strip()]
    print(f"Generating auxiliary predicates for {len(chebi_ids)} class(es).")

    problem_dir = args.problem_dir or os.path.join("data", "ilp_problems")
    total = 0
    for chebi_id in chebi_ids:
        info = _get_class_info(chebi_graph, chebi_id)
        print(f"CHEBI:{chebi_id} ({info['name']})...")
        try:
            n = generate_for_class(client, chebi_id, info, problem_dir, args.model, args.n_predicates)
            total += n
            print(f"  -> {n} auxiliary predicate(s) written")
        except Exception as e:
            print(f"  -> Failed: {e}")

    print(f"Done. Wrote {total} auxiliary predicate program(s) across {len(chebi_ids)} class(es).")


if __name__ == "__main__":
    main()
