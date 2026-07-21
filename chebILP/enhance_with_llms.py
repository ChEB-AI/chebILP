# Pipeline: read programs from a CSV file (columns: chebi_id, program, run_name).
# For each program, call an LLM to get a potentially better program.
# Input to the LLM: available predicates with explanations, ChEBI class name and
# definition, superclass information, and the original ILP program.
# The ILP program does global classification: it must distinguish the class from all other
# molecules, with its siblings as the hardest negatives. The LLM may also explain the improved program.
#
# Usage:
#   python -m chebILP.enhance_with_llms \
#       --input data/enhance_with_llms/best_ilp_programs_for_leaves.csv \
#       --output data/enhance_with_llms/enhanced_programs.csv \
#       --chebi_version 248 \
#       --predicate_set atoms

import argparse
import csv
import json
import os
import re
import time

import anthropic
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_anthropic

from chebi_utils import build_chebi_graph, download_chebi_obo
from chebILP.ilp_path_manager import get_bias_path
from chebILP.select_predicates import load_bias_predicates


_PREDICATE_EXPLANATIONS = """\
- has_atom(Molecule, Atom): Molecule contains Atom
- c(Atom), n(Atom), o(Atom), s(Atom), p(Atom), cl(Atom), br(Atom), f(Atom), i(Atom), se(Atom), ...: Atom type (element symbol, lowercase)
- charge0(Atom): no formal charge; charge_p(Atom) / charge_n(Atom): unspecified positive / negative charge; charge1(Atom) / charge_m1(Atom): exact +1 / -1 charge; etc.
- has_X_hs(Atom): Atom has exactly X attached hydrogens
- has_at_least_X_hs(Atom): Atom has at least X attached hydrogens
- cip_code_R(Atom), cip_code_S(Atom): R or S CIP stereochemistry at Atom
- bSINGLE(Atom1,Atom2), bDOUBLE(Atom1,Atom2), bTRIPLE(Atom1,Atom2), bAROMATIC(Atom1,Atom2): bond type
- bSTEREOCIS(Atom1,Atom2), bSTEREOTRANS(Atom1,Atom2): cis/trans bond stereochemistry
- has_bond_to(Atom1,Atom2): any bond between Atom1 and Atom2
- net_charge_positive(Molecule), net_charge_negative(Molecule), net_charge_neutral(Molecule): net charge for the whole molecule
- aromatic(Molecule), aliphatic(Molecule): whether the molecule is aromatic or aliphatic
- steroid_X(Atom): Atom is at steroid position X (up to position 17)\
"""

_SYSTEM_PROMPT = """\
You are an expert in chemistry and Inductive Logic Programming (ILP).
Your task is to improve Prolog-based classification rules for chemical compounds (ChEBI classes).

Rules use this format:
  chebi_{ID}(V0) :- literal1(V0, V1), literal2(V1, V2), literal3(V2).
where V0 is the molecule and V1, V2, ... are atom variables.
Multiple clauses are allowed; a molecule is classified positively if ANY clause is satisfied.

You must only use the predicates listed in the prompt. The predicates 'aromatic' and 'aliphatic' apply to the whole molecule, not individual atoms.
There are no predicates referring to ring membership, but you can use bond predicates to capture local structural patterns.

Rules are evaluated with Clingo (Answer Set Programming), not standard Prolog. Do not use inequalities or negations.\
"""


def _get_class_info(chebi_graph, chebi_id: str) -> dict:
    node_data = dict(chebi_graph.nodes.get(chebi_id, {}))
    parents = []
    for pid in chebi_graph.successors(chebi_id):
        pdata = dict(chebi_graph.nodes.get(pid, {}))
        parents.append({"id": pid, "name": pdata.get("name", f"CHEBI:{pid}"), "definition": pdata.get("definition")})
    return {
        "name": node_data.get("name", f"CHEBI:{chebi_id}"),
        "definition": node_data.get("definition"),
        "parents": parents,
    }


def _build_prompt(chebi_id: str, info: dict, original_program: str) -> str:

    parent_str = ""
    if info["parents"]:
        parts = ", ".join(f"{p['name']} (CHEBI:{p['id']} - {p['definition']})" for p in info["parents"])
        parent_str = f"Superclass(es): {parts}\n"

    definition_str = f"Definition: {info['definition']}\n" if info["definition"] else ""

    return f"""\
Target class: {info['name']} (CHEBI:{chebi_id})
{definition_str}{parent_str}
The rule must distinguish this class from all other molecules

Predicate reference:
{_PREDICATE_EXPLANATIONS}

Original program:
{original_program}

Improve this program if possible. You may add/remove literals, add clauses for edge cases,
or adjust atom/bond predicates to better match the chemistry. Keep only predicates listed above.
Keep it as close to the original program as possible. If the class is not definable with the given predicates, return an empty program and explain why in the explanation section.

Reply in exactly this format (no extra text before or after):

EXPLANATION:
<a brief explanation about what this class means and how the rule captures it>

PROGRAM:
<improved Prolog clauses, each ending with a period>
"""


def _parse_response(text: str) -> tuple[str | None, str | None]:
    exp_match = re.search(r'EXPLANATION:\s*\n(.*?)(?=\nPROGRAM:)', text, re.DOTALL)
    prog_match = re.search(r'PROGRAM:\s*\n(.*)', text, re.DOTALL)

    explanation = exp_match.group(1).strip() if exp_match else None

    program = None
    if prog_match:
        candidate = prog_match.group(1).strip()
        lines = [l.strip() for l in candidate.splitlines() if l.strip()]
        prolog_lines = [l for l in lines if l.endswith('.') and ':-' in l]
        if prolog_lines:
            program = "\n".join(prolog_lines)

    return program, explanation


@traceable(name="enhance_ilp_program")
def _enhance_one(
    client: anthropic.Anthropic,
    chebi_id: str,
    info: dict,
    original_program: str,
    model: str,
    max_retries: int = 5,
) -> dict:
    prompt = _build_prompt(chebi_id, info, original_program)
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            program, explanation = _parse_response(raw)
            return {
                "enhanced_program": program,
                "enhancement_explanation": explanation,
                "raw_llm_response": raw,
            }
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"  Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
            time.sleep(wait)
    raise last_exc


def main():
    parser = argparse.ArgumentParser(description="Enhance ILP programs using an LLM.")
    parser.add_argument("--input", required=True,
                        help="CSV file with columns: chebi_id, program, run_name.")
    parser.add_argument("--output", required=True,
                        help="Output directory (readable by the 'test' command).")
    parser.add_argument("--chebi_version", type=int, default=248)
    parser.add_argument("--problem_dir", default=None, help="Base ILP problems directory.")
    parser.add_argument("--predicate_set", default="atoms", help="Predicate set used in the original run.")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Claude model to use.")
    parser.add_argument("--labels_file", default=None, help="Only enhance ChEBI IDs listed in this file.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Add it to your .env file.")
    client = wrap_anthropic(anthropic.Anthropic(api_key=api_key))

    data_dir = os.path.join("data", f"chebi_v{args.chebi_version}")
    obo_path = os.path.join(data_dir, "raw", "chebi.obo")
    if not os.path.exists(obo_path):
        download_chebi_obo(args.chebi_version, dest_dir=data_dir)
    chebi_graph = build_chebi_graph(obo_path)

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows from {args.input}")

    filter_ids = None
    if args.labels_file:
        with open(args.labels_file) as f:
            filter_ids = {l.strip() for l in f if l.strip()}

    problem_dir = args.problem_dir or os.path.join("data", "ilp_problems")
    os.makedirs(args.output, exist_ok=True)

    # Write config.yml so the 'test' command can locate bk/exs files
    with open(os.path.join(args.output, "config.yml"), "w") as cfg_f:
        cfg_f.write(f"llm_model: {args.model}\n")
        cfg_f.write(f"predicate_set: {args.predicate_set}\n")
        cfg_f.write(f"selection_mode: None\n")
        cfg_f.write(f"selection_k: 0\n")
        cfg_f.write(f"problem_dir: {problem_dir}\n")
        cfg_f.write(f"fg_mode: None\n")

    # Clear results.json
    results_path = os.path.join(args.output, "results.json")
    open(results_path, "w").close()

    n_enhanced = 0
    with open(results_path, "a", encoding="utf-8") as out_f:
        for row in rows:
            chebi_id = str(row["chebi_id"])
            original_program = row.get("program", "").strip()

            if filter_ids and chebi_id not in filter_ids:
                continue

            info = _get_class_info(chebi_graph, chebi_id)

            print(f"Enhancing CHEBI:{chebi_id} ({info['name']})...")
            try:
                result = _enhance_one(client, chebi_id, info, original_program, args.model)
                n_enhanced += 1
                print(f"  -> {result['enhanced_program']}")
            except Exception as e:
                print(f"  -> Failed: {e}")
                result = {"enhanced_program": None, "enhancement_explanation": None}

            # 'program' is what the test command evaluates: use enhanced if available
            program = result["enhanced_program"] or original_program

            entry = {
                "chebi_id": chebi_id,
                "program": program,
                "original_program": original_program,
                "enhancement_explanation": result["enhancement_explanation"],
                "run_name": row.get("run_name", ""),
            }
            out_f.write(json.dumps(entry) + "\n")

    print(f"Done. Enhanced {n_enhanced} programs. Run directory: {args.output}")


if __name__ == "__main__":
    main()
