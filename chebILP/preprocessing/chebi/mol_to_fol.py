"""Conversion of RDKit molecules to FOL facts.

``build_background_chemlog`` is the atom-level ChEBI encoding (``has_atom`` plus the atom,
bond and ring predicates of ``chebi_utils.extract_properties.mol_to_fol_atoms``).
``mol_to_fol_fgs`` is the FARM functional-group level conversion.
"""

from rdkit import Chem

from chebi_utils.extract_properties import get_numerical_facts, mol_to_fol_atoms

from chebILP.utils import get_atom_id


def _skip_predicate(predicate: str, indices) -> bool:
    return (
        predicate in {"EQ", "atom", "*", "r", "r#"}
        or (
            predicate.startswith("r")
            and predicate[1:].isdigit()
            and int(predicate[1:]) > 0
        )
        or not indices
    )


def build_background_chemlog(rows) -> tuple[list[str], list[tuple[str, int]]]:
    """Atom-level BK facts for the molecules in ``rows`` (index = molecule id, column ``mol``).

    Returns the fact lines and the body predicates as ``(name, arity)``.
    """
    lines_by_predicate = {"has_atom": []}
    arities = {"has_atom": 2}

    for row in rows.itertuples():
        for atom in row.mol.GetAtoms():
            atom_id = get_atom_id(atom.GetIdx(), row.Index)
            lines_by_predicate["has_atom"].append(f"has_atom({row.Index},{atom_id}).")

        atom_extensions, mol_extensions = mol_to_fol_atoms(row.mol)

        for predicate, indices in atom_extensions.items():
            if predicate.startswith("cip_code_"):
                predicate = "cip_code_" + predicate[-1].upper()
            if _skip_predicate(predicate, indices):
                continue
            is_tuple = isinstance(indices[0], tuple)
            lines = lines_by_predicate.setdefault(predicate, [])
            arities.setdefault(predicate, len(indices[0]) if is_tuple else 1)
            if is_tuple:
                for args in indices:
                    arg_str = ",".join(get_atom_id(a, row.Index) for a in args)
                    lines.append(f"{predicate}({arg_str}).")
            else:
                for idx in indices:
                    lines.append(f"{predicate}({get_atom_id(idx, row.Index)}).")

        for predicate in mol_extensions:
            lines_by_predicate.setdefault(predicate, []).append(
                f"{predicate}({row.Index})."
            )
            arities.setdefault(predicate, 1)

    return [line for lines in lines_by_predicate.values() for line in lines], list(
        arities.items()
    )


def build_computed_facts(rows) -> list[str]:
    """``mol_weight(M,W)`` and ``ring_size(M,S)`` facts; used to ground LLM rules, never written to ``bk.pl``."""
    lines = []
    for row in rows.itertuples():
        for pred, values in get_numerical_facts(row.mol).items():
            for value in values:
                lines.append(f"{pred}({row.Index},{value}).")
    return lines


def mol_to_fol_fgs(mol: Chem.Mol, add_fg_atom_predicates: bool = False):
    # extract functional groups with the FARM algorithm

    from chebai_graph.preprocessing.reader.augmented_reader import (
        AtomFGReader_WithFGEdges_NoGraphNode,
    )

    reader = AtomFGReader_WithFGEdges_NoGraphNode()

    try:
        augmented_graph = reader._augment_graph_structure(mol)
    except Exception as e:
        print(f"Error occurred while augmenting graph structure: {e}")
        return {}

    nodes = augmented_graph["node_info"]["fg_nodes"]

    fg_extensions = {}

    for node_id, node in nodes.items():
        fg_type = node["FG"].lower()
        if not fg_type[0].isalpha():
            fg_type = f"fg_{fg_type}"
        fg_extensions.setdefault(fg_type, []).append(node_id)
        # ring_size = node["RING"]
        # if ring_size != 0:
        #    fg_extensions.setdefault(f"fg_ring{ring_size}", []).append(node_id)

    for edge in augmented_graph["edge_info"]["within_fg_lvl"]:
        left, right = edge.split("_")
        left = int(left)
        right = int(right)
        fg_extensions.setdefault("is_fg_neighbor", []).append((left, right))
        fg_extensions.setdefault("is_fg_neighbor", []).append((right, left))

    if add_fg_atom_predicates:
        for fg_id, atoms in augmented_graph["graph_meta_info"][
            "fg_to_atoms_map"
        ].items():
            for atom in atoms["atom"]:
                fg_extensions.setdefault("in_fg", []).append((int(atom), int(fg_id)))

    return fg_extensions
