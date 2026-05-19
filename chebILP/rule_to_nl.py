"""Translate chebILP ILP rules to natural language descriptions."""
import json
import re
from typing import Optional

from rdkit import Chem

_pt = Chem.GetPeriodicTable()
_ELEMENT_NAMES: dict[str, str] = {
    _pt.GetElementSymbol(z).lower(): _pt.GetElementName(z).lower()
    for z in range(1, 119)
}

_NUM_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _num_word(n: int) -> str:
    return _NUM_WORDS.get(n, str(n))


def _split_literals(body: str) -> list[str]:
    """Split a comma-separated body string into literals, respecting parenthesis depth."""
    literals, current, depth = [], [], 0
    for ch in body:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            literals.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        literals.append(''.join(current).strip())
    return [l for l in literals if l]


def _parse_literal(lit: str) -> tuple[str, list[str]]:
    pred = lit.split('(')[0].strip()
    args_str = lit[lit.index('(') + 1: lit.rindex(')')]
    args = [a.strip() for a in _split_literals(args_str)] if args_str.strip() else []
    return pred, args


def _parse_rule(rule: str) -> tuple[Optional[str], Optional[str], list[list[str]]]:
    """Parse rule text into (head_pred, mol_var, clause_bodies)."""
    head_pred = None
    mol_var = None
    clauses: list[list[str]] = []

    for line in rule.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('%') or ':-' not in line:
            continue
        line = line.rstrip('.')
        head_str, body_str = line.split(':-', 1)
        head_str = head_str.strip()

        if '(' in head_str:
            head_pred = head_str.split('(')[0].strip()
            args_str = head_str[head_str.index('(') + 1: head_str.rindex(')')]
            mol_var = args_str.strip().split(',')[0].strip()
        else:
            head_pred = head_str

        clauses.append(_split_literals(body_str.strip()))

    return head_pred, mol_var, clauses


def _collect_atom_vars(literals: list[str], mol_var: str) -> list[str]:
    """Return unique atom variables in order of first appearance (all non-molecule vars)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for lit in literals:
        if '(' not in lit:
            continue
        _, args = _parse_literal(lit)
        for arg in args:
            if re.fullmatch(r'[A-Z][A-Za-z0-9]*', arg) and arg != mol_var and arg not in seen:
                seen.add(arg)
                ordered.append(arg)
    return ordered


def _make_var_map(atom_vars: list[str]) -> dict[str, str]:
    """Map VN-style Popper variables to AN; leave human-readable vars unchanged."""
    if all(re.fullmatch(r'V\d+', v) for v in atom_vars):
        return {v: f"A{v[1:]}" for v in atom_vars}
    return {v: v for v in atom_vars}


def _apply_map(text: str, var_map: dict[str, str]) -> str:
    for old, new in var_map.items():
        text = re.sub(rf'\b{re.escape(old)}\b', new, text)
    return text


def _literal_to_nl(pred: str, args: list[str]) -> Optional[str]:
    """Translate a single predicate to a natural language phrase, or None to skip."""
    if pred == 'has_atom':
        return None

    if pred in _ELEMENT_NAMES:
        name = _ELEMENT_NAMES[pred]
        article = 'an' if name[0] in 'aeiou' else 'a'
        return f"{args[0]} is {article} {name} atom"

    if re.fullmatch(r'b[A-Z]+', pred):
        bond_type = pred[1:].lower()
        article = 'an' if bond_type[0] in 'aeiou' else 'a'
        return f"{args[0]} has {article} {bond_type} bond to {args[1]}"

    if pred == 'has_bond_to':
        return f"{args[0]} has a bond to {args[1]}"

    m = re.fullmatch(r'has_(\d+)_hs', pred)
    if m:
        n = int(m.group(1))
        noun = "hydrogen atom" if n == 1 else "hydrogen atoms"
        return f"{args[0]} has {_num_word(n)} {noun}"

    m = re.fullmatch(r'has_at_least_(\d+)_hs', pred)
    if m:
        n = int(m.group(1))
        noun = "hydrogen atom" if n == 1 else "hydrogen atoms"
        return f"{args[0]} has at least {_num_word(n)} {noun}"

    if pred == 'charge0':
        return f"{args[0]} has no formal charge"
    m = re.fullmatch(r'charge_?([pm])(\d*)', pred)
    if m:
        sign = '+' if m.group(1) == 'p' else '-'
        val = m.group(2) or '1'
        return f"{args[0]} has formal charge {sign}{val}"

    if pred.startswith('cip_code_'):
        code = pred.replace('cip_code_', '').upper()
        return f"{args[0]} has CIP code {code}"

    if pred.startswith('steroid_'):
        num = pred.replace('steroid_', '')
        return f"{args[0]} is at steroid position {num}"

    if pred == 'aromatic':
        return "the molecule is aromatic"
    if pred == 'aliphatic':
        return "the molecule is aliphatic"
    if pred == 'net_charge_positive':
        return "the molecule has a positive net charge"
    if pred == 'net_charge_negative':
        return "the molecule has a negative net charge"
    if pred == 'net_charge_neutral':
        return "the molecule has no net charge"

    return f"{pred}({', '.join(args)})"


def _clause_to_nl(literals: list[str], mol_var: str) -> str:
    """Translate a single clause body to a natural language sentence."""
    atom_vars = _collect_atom_vars(literals, mol_var)
    var_map = _make_var_map(atom_vars)
    mapped_atom_vars = sorted(
        [var_map[v] for v in atom_vars],
        key=lambda a: int(a[1:]) if a[1:].isdigit() else (0, a),
    )

    parts: list[str] = []
    if len(mapped_atom_vars) == 1:
        parts.append(f"it has an atom {mapped_atom_vars[0]}")
    elif len(mapped_atom_vars) > 1:
        listed = ", ".join(mapped_atom_vars[:-1]) + f" and {mapped_atom_vars[-1]}"
        parts.append(f"it has atoms {listed}")

    for lit in literals:
        if '(' not in lit:
            continue
        pred, args = _parse_literal(lit)
        mapped_args = [_apply_map(a, var_map) for a in args]
        phrase = _literal_to_nl(pred, mapped_args)
        if phrase:
            parts.append(phrase)

    return ". ".join(parts) + "."


def translate_rule(rule: str, class_parents: Optional[dict] = None) -> str:
    """
    Translate one or more ILP rule clauses to a natural language description.

    Args:
        rule:          Rule string (one or more clauses sharing the same head predicate).
        class_parents: Dict loaded from data/class_parents.json. Used to look up the
                       target class name and parent class(es) automatically.

    Returns:
        Multi-line natural language string.
    """
    head_pred, mol_var, clauses = _parse_rule(rule)

    chebi_id: Optional[str] = None
    if head_pred and head_pred.startswith('chebi_'):
        chebi_id = head_pred.split('_', 1)[1]

    chebi_name: Optional[str] = None
    parent_ids: list[str] = []
    parent_names: list[str] = []
    if class_parents and chebi_id and chebi_id in class_parents:
        entry = class_parents[chebi_id]
        chebi_name = entry.get("name")
        parent_ids = entry.get("parent_ids", [])
        parent_names = entry.get("parent_names", [])

    if chebi_name and chebi_id:
        header = f"A compound is a {chebi_name} (CHEBI:{chebi_id}) if it"
    elif chebi_id:
        header = f"A compound is a CHEBI:{chebi_id} entity if it"
    else:
        header = "A compound satisfies the rule if it"

    lines = [header]
    label = ord('a')

    for pid, pname in zip(parent_ids, parent_names):
        lines.append(f"({chr(label)}) is a {pname} (CHEBI:{pid}) AND")
        label += 1

    if not clauses:
        lines.append(f"({chr(label)}) (no conditions found)")
    elif len(clauses) == 1:
        lines.append(f"({chr(label)}) {_clause_to_nl(clauses[0], mol_var)}")
    else:
        lines.append(f"({chr(label)}) satisfies one of the following conditions:")
        for clause_lits in clauses:
            lines.append(f"    - {_clause_to_nl(clause_lits, mol_var)}")

    return "\n".join(lines)


def load_class_parents(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
