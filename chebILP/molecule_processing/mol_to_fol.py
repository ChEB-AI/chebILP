"""Functional-group level conversion of RDKit molecules to FOL predicate extensions.

The atom-level conversion (``mol_to_fol_atoms``) lives in
``chebi_utils.extract_properties``.
"""

from rdkit import Chem


def mol_to_fol_fgs(mol: Chem.Mol, add_fg_atom_predicates: bool = False):
    # extract functional groups with the FARM algorithm

    from chebai_graph.preprocessing.reader.augmented_reader import AtomFGReader_WithFGEdges_NoGraphNode
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
        #ring_size = node["RING"]
        #if ring_size != 0:
        #    fg_extensions.setdefault(f"fg_ring{ring_size}", []).append(node_id)

    for edge in augmented_graph["edge_info"]["within_fg_lvl"]:
        left, right = edge.split("_")
        left = int(left)
        right = int(right)
        fg_extensions.setdefault("is_fg_neighbor", []).append((left, right))
        fg_extensions.setdefault("is_fg_neighbor", []).append((right, left))

    if add_fg_atom_predicates:
        for fg_id, atoms in augmented_graph["graph_meta_info"]["fg_to_atoms_map"].items():
            for atom in atoms["atom"]:
                fg_extensions.setdefault("in_fg", []).append((int(atom), int(fg_id)))


    return fg_extensions


if __name__ == "__main__":

    from chebai_graph.preprocessing.reader.augmented_reader import AtomFGReader_WithFGEdges_NoGraphNode
    reader = AtomFGReader_WithFGEdges_NoGraphNode()

    mol = Chem.MolFromSmiles("CCO")
    try:
        augmented_graph = reader._augment_graph_structure(mol)
    except Exception as e:
        print(f"Error occurred while augmenting graph structure: {e}")

    print(augmented_graph)

