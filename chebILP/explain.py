import json
import re
from typing import Optional

import pandas as pd
from rdkit import Chem
from xclingo import XclingoControl

from chebILP.mol2ilp import build_background_chemlog
from chebILP.utils import split_prolog_literals
from chebILP.rule_to_nl import _ELEMENT_NAMES, _num_word, _sort_clause_literals


def _trace_for_predicate(pred_name: str, arity: int) -> Optional[str]:
    """Return a %!trace annotation line for a predicate, or None if not applicable."""
    if arity == 1:
        A = "A"
        call = f"{pred_name}({A})"
        if pred_name.lower() in _ELEMENT_NAMES:
            name = _ELEMENT_NAMES[pred_name.lower()]
            article = "an" if name[0] in "aeiou" else "a"
            return f'%!trace {{"% is {article} {name} atom", {A}}} {call}.'
        if pred_name == "charge0":
            return f'%!trace {{"% has no formal charge", {A}}} {call}.'
        m = re.fullmatch(r"charge_?([pm])(\d*)", pred_name)
        if m:
            sign = "+" if m.group(1) == "p" else "-"
            val = m.group(2) or "1"
            return f'%!trace {{"% has formal charge {sign}{val}", {A}}} {call}.'
        m = re.fullmatch(r"has_(\d+)_hs", pred_name)
        if m:
            n = int(m.group(1))
            noun = "hydrogen atom" if n == 1 else "hydrogen atoms"
            return f'%!trace {{"% has {_num_word(n)} {noun}", {A}}} {call}.'
        m = re.fullmatch(r"has_at_least_(\d+)_hs", pred_name)
        if m:
            n = int(m.group(1))
            noun = "hydrogen atom" if n == 1 else "hydrogen atoms"
            return f'%!trace {{"% has at least {_num_word(n)} {noun}", {A}}} {call}.'
        if pred_name.startswith("cip_code_"):
            code = pred_name.replace("cip_code_", "").upper()
            return f'%!trace {{"% has CIP code {code}", {A}}} {call}.'
        if pred_name.startswith("steroid_"):
            num = pred_name.replace("steroid_", "")
            return f'%!trace {{"% is at steroid position {num}", {A}}} {call}.'
        if pred_name == "aromatic":
            return f'%!trace {{"% is aromatic", {A}}} {call}.'
        if pred_name == "aliphatic":
            return f'%!trace {{"% is aliphatic", {A}}} {call}.'
        if pred_name == "net_charge_positive":
            return f'%!trace {{"% has a positive net charge", {A}}} {call}.'
        if pred_name == "net_charge_negative":
            return f'%!trace {{"% has a negative net charge", {A}}} {call}.'
        if pred_name == "net_charge_neutral":
            return f'%!trace {{"% has no net charge", {A}}} {call}.'
        return f'%!trace {{"% satisfies {pred_name}", {A}}} {call}.'

    if arity == 2:
        A1, A2 = "A1", "A2"
        call = f"{pred_name}({A1},{A2})"
        if pred_name == "has_atom":
            return f'%!trace {{"% has atom %", {A1}, {A2}}} {call}.'
        if pred_name == "has_bond_to":
            return f'%!trace {{"% has a bond to %", {A1}, {A2}}} {call}.'
        if re.fullmatch(r"b[A-Z]+", pred_name):
            bond_type = pred_name[1:].lower()
            article = "an" if bond_type[0] in "aeiou" else "a"
            return f'%!trace {{"% has {article} {bond_type} bond to %", {A1}, {A2}}} {call}.'
        return f'%!trace {{"% is related to % via {pred_name}", {A1}, {A2}}} {call}.'

    return None


def _parse_rule(rule: str) -> tuple[Optional[str], set[tuple[str, int]]]:
    """Parse rule clause(s) and return (head_pred_name, set of (body_pred, arity))."""
    head_pred = None
    body_predicates: set[tuple[str, int]] = set()

    for line in rule.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("%") or ":-" not in line:
            continue
        line = line.rstrip(".")
        head_str, body_str = line.split(":-", 1)
        head_str = head_str.strip()
        body_str = body_str.strip()

        head_pred = head_str.split("(")[0].strip() if "(" in head_str else head_str

        for lit in split_prolog_literals(body_str):
            lit = lit.strip()
            if not lit or "(" not in lit:
                continue
            pname = lit.split("(")[0].strip()
            args_str = lit[lit.index("(") + 1: lit.rindex(")")]
            n_args = len(split_prolog_literals(args_str)) if args_str.strip() else 0
            body_predicates.add((pname, n_args))

    return head_pred, body_predicates


def _sort_rule_literals(rule: str) -> str:
    """Sort body literals in each clause using the same BFS order as rule_to_nl."""
    sorted_clauses = []
    for line in rule.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("%") or ":-" not in line:
            sorted_clauses.append(line)
            continue
        line = line.rstrip(".")
        head_str, body_str = line.split(":-", 1)
        head_str = head_str.strip()
        mol_var = "V0"
        if "(" in head_str:
            args_str = head_str[head_str.index("(") + 1: head_str.rindex(")")]
            mol_var = args_str.strip().split(",")[0].strip()
        literals = split_prolog_literals(body_str.strip())
        sorted_lits = _sort_clause_literals(literals, mol_var)
        sorted_clauses.append(f"{head_str} :- {', '.join(sorted_lits)}.")
    return "\n".join(sorted_clauses)


def _build_xclingo_program(
    bk_lines: list[str],
    rule: str,
    head_pred: str,
    body_predicates: set[tuple[str, int]],
) -> str:
    all_predicates = set(body_predicates) | {("has_atom", 2)}

    trace_lines = []
    for pname, arity in sorted(all_predicates):
        ann = _trace_for_predicate(pname, arity)
        if ann:
            trace_lines.append(ann)
    trace_lines.append(f'%!trace {{"% satisfies {head_pred}", M}} {head_pred}(M).')

    return "\n".join([
        "% Background knowledge",
        *bk_lines,
        "",
        "% Rule(s)",
        *[l for l in _sort_rule_literals(rule).split("\n") if l.strip()],
        "",
        "% Trace annotations",
        *trace_lines,
        "",
        f"%!show_trace {head_pred}(M).",
    ])


def _run_xclingo(program: str) -> list:
    """Run xclingo (1 solution, 1 explanation) and return the Explanation objects found."""
    xcontrol = XclingoControl(n_solutions="1", n_explanations="1")
    xcontrol.add("base", [], program)
    xcontrol.ground()

    explanations = []
    for answer_explanations in xcontrol.explain():
        for expl in answer_explanations:
            explanations.append(expl)
    return explanations


def _process_explanations(
    explanations: list, mol_id: str
) -> tuple[bool, list[str], set[int]]:
    """
    Extract satisfies flag, readable conditions, and highlighted atom indices.

    Skips 'has atom' and 'satisfies' meta-relations; replaces atom IDs like
    'amol1_3' with 'atom 3'. Conditions are deduplicated across explanations.
    """
    if not explanations:
        return False, [], set()

    atom_re = re.compile(re.escape("a" + str(mol_id) + "_") + r"(\d+)")
    highlight_atoms: set[int] = set()
    conditions: list[str] = []
    seen: set[str] = set()

    for expl in explanations:
        for node, _ in expl.preorder_iterator():
            if not hasattr(node, "labels"):  # ExplanationRoot has no labels
                continue
            for label in node.labels:
                # Collect atom indices for highlighting before any text replacement
                for m in atom_re.finditer(label):
                    highlight_atoms.add(int(m.group(1)) - 1)

                if "has atom" in label or "satisfies" in label:
                    continue

                clean = atom_re.sub(lambda m: f"atom {m.group(1)}", label)
                clean = clean.replace(mol_id, "the molecule")
                if clean not in seen:
                    seen.add(clean)
                    conditions.append(clean)

    return True, conditions, highlight_atoms


def _draw_molecule(
    mol: Chem.Mol,
    highlight_atoms: set[int],
    output_path: Optional[str] = None,
):
    """Draw molecule with 1-based atom numbers matching ILP IDs, highlighted atoms in orange."""
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    import io

    mol_labeled = Chem.RWMol(mol)
    for atom in mol_labeled.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    hl_list = list(highlight_atoms)
    hl_colors = {idx: (1.0, 0.6, 0.2) for idx in highlight_atoms}

    drawer = rdMolDraw2D.MolDraw2DCairo(700, 450)
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(
        mol_labeled,
        highlightAtoms=hl_list,
        highlightAtomColors=hl_colors,
        highlightBonds=[],
        highlightBondColors={},
    )
    drawer.FinishDrawing()

    img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    if output_path:
        img.save(output_path)
        print(f"Image saved to: {output_path}")
    return img


def explain_molecule(
    smiles: str,
    rule: str,
    label_parents_json: str,
    mol_id: str = "mol1",
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> tuple[bool, str, "PIL.Image.Image"]:
    """
    Explain why a molecule satisfies (or fails to satisfy) a learned ILP rule.

    Args:
        smiles:       SMILES string of the molecule.
        rule:         One or more ILP rule clauses as a string.
        mol_id:       Identifier for the molecule in the logic program (default: "mol1").
        output_path:  Path to save the visualization image (PNG). Optional.
        verbose:      If True, print the assembled xclingo program before running.

    Returns:
        satisfies:   True if the molecule satisfies the rule.
        conditions:  List of human-readable condition strings (empty if not satisfying).
                     Atom IDs are replaced with readable labels (e.g. "atom 3").
        image:       PIL Image with 1-based numbered atoms; atoms in the explanation
                     are highlighted in orange.
    """
    label_parents = json.load(open(label_parents_json))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    mol_df = pd.DataFrame([{"mol": mol}], index=[mol_id])
    bk_lines, _ = build_background_chemlog(mol_df)

    head_pred, body_predicates = _parse_rule(rule)
    if head_pred is None:
        raise ValueError("Could not parse rule: no clause with ':-' found.")

    program = _build_xclingo_program(bk_lines, rule, head_pred, body_predicates)

    if verbose:
        print("=== xclingo program ===")
        print(program)
        print("=" * 23)

    explanations = _run_xclingo(program)
    satisfies, conditions, highlight_atoms = _process_explanations(explanations, mol_id)

    image = _draw_molecule(mol, highlight_atoms, output_path)

    parents = label_parents.get(head_pred.split("_")[1], [])
    p_conditions = [f"the molecule is a {p} (CHEBI:{id}, not verified)" for p, id in zip(parents["parent_names"], parents["parent_ids"])]
    conditions = p_conditions + conditions

    if not satisfies:
        full_text = f"The molecule is not a {parents['name']} (CHEBI:{head_pred.split('_')[1]})"
    else:
        full_text = f"The molecule is a {parents['name']} (CHEBI:{head_pred.split('_')[1]}) because it fulfills the following conditions:\n" + "\n".join(f"- {c}" for c in conditions)
    return satisfies, full_text, image
