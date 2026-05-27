"""Translate chebILP ILP rules to natural language descriptions."""
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


def _sort_clause_literals(literals: list[str], mol_var: str) -> list[str]:
    """Sort clause body literals for readable NL output.

    Order:
      1. has_atom literals (define variables; produce no NL output)
      2. Molecule-level predicates (only V0: aromatic, net_charge_*, etc.)
      3. Atom predicates in BFS order from has_atom-introduced variables:
         - unary predicates for the current variable first (arity 1)
         - then binary predicates, which may introduce new variables into the BFS
    """
    if not literals:
        return literals

    def atom_args(lit: str) -> list[str]:
        if '(' not in lit:
            return []
        _, args = _parse_literal(lit)
        return [a for a in args if re.fullmatch(r'[A-Z][A-Za-z0-9]*', a) and a != mol_var]

    has_atom_lits, mol_lits, atom_lits = [], [], []
    for lit in literals:
        if '(' not in lit:
            atom_lits.append(lit)
            continue
        pred, _ = _parse_literal(lit)
        if pred == 'has_atom':
            has_atom_lits.append(lit)
        elif not atom_args(lit):
            mol_lits.append(lit)
        else:
            atom_lits.append(lit)

    if not atom_lits:
        return has_atom_lits + mol_lits

    # Root vars: variables introduced by has_atom(V0, VX), in order of appearance
    root_vars: list[str] = []
    seen_root: set[str] = set()
    for lit in has_atom_lits:
        for a in atom_args(lit):
            if a not in seen_root:
                root_vars.append(a)
                seen_root.add(a)
    if not root_vars:
        for lit in atom_lits:
            args = atom_args(lit)
            if args:
                root_vars = [args[0]]
                break

    result: list[str] = has_atom_lits + mol_lits
    placed: set[int] = set()
    seen_vars: set[str] = set(root_vars) | {mol_var}
    queue: list[str] = list(root_vars)

    while queue:
        var = queue.pop(0)
        unary, binary = [], []
        for i, lit in enumerate(atom_lits):
            if i in placed or var not in atom_args(lit):
                continue
            (unary if len(atom_args(lit)) == 1 else binary).append(i)

        for i in unary:
            result.append(atom_lits[i])
            placed.add(i)

        for i in binary:
            result.append(atom_lits[i])
            placed.add(i)
            for v in atom_args(atom_lits[i]):
                if v not in seen_vars:
                    seen_vars.add(v)
                    queue.append(v)

        if not queue:
            # Start next disconnected component
            for i, lit in enumerate(atom_lits):
                if i not in placed:
                    args = atom_args(lit)
                    if args and args[0] not in seen_vars:
                        seen_vars.add(args[0])
                        queue.append(args[0])
                        break

    for i, lit in enumerate(atom_lits):
        if i not in placed:
            result.append(lit)

    return result


def _make_clause_var_maps(clauses: list[list[str]], mol_var: str) -> list[dict[str, str]]:
    """Build per-clause variable maps with a global atom counter across all clauses.

    Variables in different clauses are unrelated, so each gets a unique label
    (A1, A2, A3, ...) rather than restarting from A1 in every clause.
    """
    per_clause_vars = [_collect_atom_vars(lits, mol_var) for lits in clauses]
    all_vars = [v for vs in per_clause_vars for v in vs]
    use_global = bool(all_vars) and all(re.fullmatch(r'V\d+', v) for v in all_vars)

    maps: list[dict[str, str]] = []
    counter = 1
    for atom_vars in per_clause_vars:
        if use_global:
            var_map = {v: f"A{counter + i}" for i, v in enumerate(atom_vars)}
            counter += len(atom_vars)
        else:
            var_map = {v: v for v in atom_vars}
        maps.append(var_map)
    return maps


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


def _clause_to_nl(literals: list[str], mol_var: str, var_map: dict[str, str]) -> str:
    """Translate a single clause body to a natural language sentence."""
    atom_vars = _collect_atom_vars(literals, mol_var)
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


def translate_rule(rule: str, chebi_graph=None) -> str:
    """
    Translate one or more ILP rule clauses to a natural language description.

    Args:
        rule:        Rule string (one or more clauses sharing the same head predicate).
        chebi_graph: networkx DiGraph loaded from chebi_graph.pkl. Used to look up the
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
    if chebi_graph is not None and chebi_id:
        node_data = dict(chebi_graph.nodes.get(chebi_id, {}))
        chebi_name = node_data.get("name")
        for pid in chebi_graph.successors(chebi_id):
            pdata = dict(chebi_graph.nodes.get(pid, {}))
            parent_ids.append(str(pid))
            parent_names.append(pdata.get("name", f"CHEBI:{pid}"))

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

    sorted_clauses = [_sort_clause_literals(lits, mol_var) for lits in clauses]
    clause_var_maps = _make_clause_var_maps(sorted_clauses, mol_var)

    if not clauses:
        lines.append(f"({chr(label)}) (no conditions found)")
    elif len(clauses) == 1:
        lines.append(f"({chr(label)}) {_clause_to_nl(sorted_clauses[0], mol_var, clause_var_maps[0])}")
    else:
        lines.append(f"({chr(label)}) satisfies one of the following conditions:")
        for i, (clause_lits, var_map) in enumerate(zip(sorted_clauses, clause_var_maps)):
            suffix = "" if i == len(clauses) - 1 else " OR"
            lines.append(f"    - {_clause_to_nl(clause_lits, mol_var, var_map)}{suffix}")

    return "\n".join(lines)


