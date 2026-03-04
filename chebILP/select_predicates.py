"""
Select predicates for ChEBI classes using Claude as a filter.

This module loads the ChEBI dataset, reads predicates from bias.pl files created
by ILPProblemBuilder, and uses Claude to select which predicates are useful
for defining each ChEBI class.
"""

import os
import re
import argparse
from typing import Literal

import networkx as nx

from chebILP.ilp_path_manager import get_bias_path, get_bk_path
from chebi_utils import build_chebi_graph, download_chebi_obo

def load_bias_predicates(bias_path: str) -> list[tuple[str, int]]:
    """
    Load body predicates and max_vars / max_body / max_clauses from a bias.pl file.
    
    Args:
        bias_path: Path to the bias.pl file.
        
    Returns:
        List of (predicate_name, arity) tuples.
    """
    predicates = []
    with open(bias_path, "r") as f:
        for line in f:
            line = line.strip()
            match = re.match(r"body_pred\((\w+),\s*(\d+)\)\.", line)
            if match:
                predicates.append((match.group(1), int(match.group(2))))
    return predicates


def get_chebi_class_info(chebi_graph: nx.DiGraph, chebi_id: int) -> dict:
    """
    Get information about a ChEBI class from the ontology graph.

    Args:
        chebi_graph: ChEBI ontology graph built by ``build_chebi_graph``.
        chebi_id: The ChEBI ID (integer).

    Returns:
        Dictionary with 'name', 'definition', and other properties.
    """
    node_id = str(chebi_id)
    if node_id in chebi_graph.nodes:
        return dict(chebi_graph.nodes[node_id])
    return {"name": f"CHEBI:{chebi_id}", "definition": None}


def ask_claude_for_predicates(
    chebi_id: int,
    chebi_name: str,
    chebi_definition: str | None,
    predicates: list[tuple[str, int]],
    model: str = "claude-opus-4-6",
    top_k: int = 10,
) -> list[tuple[str, int]]:
    import anthropic
    """
    Ask Claude which predicates are useful for defining a ChEBI class.
    
    Args:
        chebi_id: The ChEBI ID.
        chebi_name: Human-readable name of the ChEBI class.
        chebi_definition: Definition text from ChEBI ontology (if available).
        predicates: List of (predicate_name, arity) tuples available.
        model: Claude model to use.
        top_k: Number of predicates to select (used in prompt to guide selection).
        
    Returns:
        List of selected (predicate_name, arity) tuples.
    """
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file, including ANTHROPIC_API_KEY
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in your .env file.")
    client = anthropic.Anthropic(api_key=api_key)
    
    # Format predicates for the prompt
    predicates_str = "\n".join([f"- {pred}(arity={arity})" for pred, arity in predicates])
    
    definition_section = ""
    if chebi_definition:
        definition_section = f"\nDefinition from ChEBI: {chebi_definition}\n"
    
    prompt = f"""You are helping to select predicates for Inductive Logic Programming (ILP) to learn a classification rule for a chemical class.

ChEBI Class: CHEBI:{chebi_id} - {chebi_name}
{definition_section}
The following predicates are available for building classification rules. Each predicate represents a chemical property or structural feature:

{predicates_str}

Predicate explanations:
- has_atom(Mol, Atom): Molecule Mol contains Atom
- c, n, o, s, p, fe, cl, se, etc.: Atom type predicates (carbon, nitrogen, oxygen, etc.)
- charge0, charge1, charge_p, charge_n, charge_m1, charge_m3: Atom charge predicates
- has_X_hs (e.g., has_0_hs, has_1_hs): Atom has exactly X hydrogens attached
- has_at_least_X_hs: Atom has at least X hydrogens attached
- cip_code_R, cip_code_S: Stereochemistry (R/S configuration)
- bSINGLE, bDOUBLE, bTRIPLE, bAROMATIC: Bond type between two atoms
- bSTEREOCIS, bSTEREOTRANS: Cis/trans stereochemistry of bonds
- has_bond_to(Atom1, Atom2): Two atoms are bonded
- net_charge_positive, net_charge_negative, net_charge_neutral: Overall molecular charge
- global(Mol): Represents the whole molecule (for global properties)

Your task: Select exactly {top_k} the predicates that are likely to be useful for defining molecules belonging to the class "{chebi_name}". Think about what structural features characterize this class.

Respond with ONLY the predicate names (without arity), one per line, with no additional text or explanation. For example:
has_atom
c
o
bSINGLE
"""

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response to extract selected predicates
    response_text = response.content[0].text
    selected_names = set(line.strip() for line in response_text.strip().splitlines() if line.strip())
    
    # Filter predicates to only those selected by Claude
    selected_predicates = [(pred, arity) for pred, arity in predicates if pred in selected_names]
    
    return selected_predicates


def select_most_common_predicates(bk_path, predicates, top_k: int) -> list[tuple[str, int]]:
    """
    Select the most common predicates for a given ChEBI class based on training examples.
    
    Args:
        bk_path: Path to the bk.pl file containing the background knowledge with predicates.
        predicates: List of (predicate_name, arity) tuples available.
        top_k: Number of top predicates to select.
        
    Returns:
        List of selected (predicate_name, arity) tuples.
    """
    predicate_counts = {pred: 0 for pred, _ in predicates}
    
    with open(bk_path, "r") as f:
        for line in f:
            line = line.strip()
            for pred, _ in predicates:
                if line.startswith(f"{pred}("):
                    predicate_counts[pred] += 1
    
    # Sort predicates by count and select top_k
    sorted_preds = sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)
    selected_predicates = [(pred, arity) for pred, arity in predicates if pred in dict(sorted_preds[:top_k])]
    
    return selected_predicates



def write_bias_file(
    output_path: str,
    chebi_id: int,
    head_predicate: str,
    body_predicates: list[tuple[str, int]]
):
    """
    Write a bias.pl file with the selected predicates.
    
    Args:
        output_path: Path to write the bias file.
        chebi_id: ChEBI ID for the target class.
        head_predicate: Name of the head predicate (e.g., "chebi_24062").
        body_predicates: List of (predicate_name, arity) tuples to include.
    """
    lines = [
        f"%% CHEBI:{chebi_id} (bias file with pre-selected predicates)",
        f"",
        f"%% max_vars(TODO).",
        f"%% max_body(TODO).",
        f"%% max_clauses(TODO).",
        f"",
        f"head_pred({head_predicate}, 1).",
    ]
    
    for pred, arity in body_predicates:
        lines.append(f"body_pred({pred},{arity}).")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def select_predicates_for_class(
    chebi_id: int,
    chebi_graph: nx.DiGraph,
    problem_dir: str,
    predicate_set: Literal["atoms", "chembl_fgs"] = "atoms",
    selection_mode: Literal["claude", "random", "top_k"] = "claude",
    top_k: int = 10,
) -> str:
    """
    Select predicates for a single ChEBI class using Claude.
    
    Args:
        chebi_id: ChEBI ID of the target class.
        chebi_graph: ChEBI ontology graph built by ``build_chebi_graph``.
        problem_dir: Base directory for ILP problems.
        predicate_set: Which predicate set to use ("atoms" or "chembl_fgs").
        selection_mode: How to select predicates ("claude", "random", or "top_k").
        top_k: Number of predicates to select (used in "top_k" mode).
    Returns:
        Path to the output bias file.
    """
    bias_path_before = get_bias_path(chebi_id, base_dir=problem_dir, predicate_set=predicate_set)
    if not os.path.exists(bias_path_before):
        raise FileNotFoundError(f"Bias file not found: {bias_path_before}. Try to build the ILP problem for this class first to generate the bias file with all predicates.")
    
    # Load predicates from bias.pl
    predicates = load_bias_predicates(bias_path_before)
    print(f"Loaded {len(predicates)} predicates from {bias_path_before}")
    
    # Get ChEBI class info
    class_info = get_chebi_class_info(chebi_graph, chebi_id)
    chebi_name = class_info.get("name", f"CHEBI:{chebi_id}")
    chebi_definition = class_info.get("definition")
    
    print(f"Processing CHEBI:{chebi_id} - {chebi_name}")
    
    if selection_mode == "random":
        import random
        selected_predicates = random.sample(predicates, min(top_k, len(predicates)))
    elif selection_mode == "top_k":
        bk_path = get_bk_path(chebi_id, base_dir=problem_dir, predicate_set=predicate_set)
        selected_predicates =select_most_common_predicates(bk_path, predicates, top_k)
    elif selection_mode == "claude":
        # Ask Claude for predicate selection
        selected_predicates = ask_claude_for_predicates(
            chebi_id=chebi_id,
            chebi_name=chebi_name,
            chebi_definition=chebi_definition,
            predicates=predicates,
            top_k=top_k,
        )
    
    print(f"{selection_mode} selected {len(selected_predicates)} predicates out of {len(predicates)}")
    
    # Write output bias file
    output_path = get_bias_path(chebi_id, base_dir=problem_dir, predicate_set=predicate_set, selection_mode=selection_mode)
    write_bias_file(
        output_path=output_path,
        chebi_id=chebi_id,
        head_predicate=f"chebi_{chebi_id}",
        body_predicates=selected_predicates,
    )
    
    print(f"Wrote selected predicates to {output_path}")
    return output_path


def select_predicates_for_classes(
    chebi_ids: list[int],
    chebi_version: int = 244,
    problem_dir: str | None = None,
    predicate_set: Literal["atoms", "chembl_fgs"] = "atoms",
    selection_mode: Literal["claude", "random", "top_k"] = "claude",
    top_k: int = 10,
) -> dict[int, str]:
    """
    Select predicates for multiple ChEBI classes.
    
    Args:
        chebi_ids: List of ChEBI IDs to process.
        chebi_version: ChEBI version to use.
        problem_dir: Base directory for ILP problems (default: ilp/chebi_v{version}).
        predicate_set: Which predicate set to use ("atoms" or "chembl_fgs").
        selection_mode: How to select predicates ("claude", "random", or "top_k").
        top_k: Number of predicates to select.
    Returns:
        Dictionary mapping ChEBI IDs to output bias file paths.
    """
    if problem_dir is None:
        problem_dir = os.path.join("ilp", f"chebi_v{chebi_version}")
    
    # Load ChEBI data
    print(f"Loading ChEBI data (version {chebi_version})...")
    data_dir = os.path.join("data", f"chebi_v{chebi_version}")
    os.makedirs(data_dir, exist_ok=True)
    obo_path = os.path.join(data_dir, "chebi.obo")
    if not os.path.exists(obo_path):
        download_chebi_obo(chebi_version, dest_dir=data_dir)
    chebi_graph = build_chebi_graph(obo_path)
    
    results = {}
    for chebi_id in chebi_ids:
        try:
            output_path = select_predicates_for_class(
                chebi_id=chebi_id,
                chebi_graph=chebi_graph,
                problem_dir=problem_dir,
                predicate_set=predicate_set,
                selection_mode=selection_mode,
                top_k=top_k
            )
            results[chebi_id] = output_path
        except Exception as e:
            print(f"Error processing CHEBI:{chebi_id}: {e}")
            results[chebi_id] = None
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select predicates for ChEBI classes using Claude."
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help="Path to file with ChEBI IDs (one per line)."
    )
    parser.add_argument(
        "--chebi_version",
        type=int,
        default=244,
        help="ChEBI version to use."
    )
    parser.add_argument(
        "--problem_dir",
        type=str,
        default=None,
        help="Base directory for ILP problems."
    )
    parser.add_argument(
        "--predicate_set",
        type=str,
        default="atoms",
        choices=["atoms", "chembl_fgs"],
        help="Which predicate set to use."
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="claude",
        choices=["claude", "random", "top_k"],
        help="How to select predicates."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of predicates to select."
    )
    
    args = parser.parse_args()
    
    # Load ChEBI IDs from labels file
    with open(args.labels_file, "r") as f:
        chebi_ids = [int(line.strip()) for line in f if line.strip()]
    
    print(f"Processing {len(chebi_ids)} ChEBI classes...")
    
    results = select_predicates_for_classes(
        chebi_ids=chebi_ids,
        chebi_version=args.chebi_version,
        problem_dir=args.problem_dir,
        predicate_set=args.predicate_set,
        selection_mode=args.selection_mode,
        top_k=args.top_k,
    )
    
    # Print summary
    successful = sum(1 for v in results.values() if v is not None)
    print(f"\nCompleted: {successful}/{len(chebi_ids)} classes processed successfully")
