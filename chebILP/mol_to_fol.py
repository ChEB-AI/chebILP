"""Convert RDKit molecules to first-order logic predicate extensions.

Ported from chemlog.preprocessing.mol_to_fol — only the atom-level
conversion (``mol_to_fol_atoms``) is included here.
"""

import logging

from rdkit import Chem

# Largest ring size that gets its own size-specific ring{N}/in_ring{N}
# predicates. Larger rings only contribute to the generic unary in_ring.
MAX_RING_SIZE = 8

# Gonane core SMARTS (C1–C17, IUPAC steroid numbering) compiled once at import.
# [#6] matches any carbon; ~ matches any bond — handles unsaturated steroids
# (Δ4, Δ5), ketones, and estrogens (aromatic ring A) without modification.
# C18/C19 methyls and side chains are intentionally excluded so the pattern
# matches all steroid sub-classes, not just fully-saturated ones.
_GONANE_PATTERN = Chem.MolFromSmarts(
    "[#6:13]12~[#6:12]~[#6:11]~[#6:9]3~[#6:10]4~"
    "[#6:1]~[#6:2]~[#6:3]~[#6:4]~[#6:5]4~"
    "[#6:6]~[#6:7]~[#6:8]3~[#6:14]2~"
    "[#6:15]~[#6:16]~[#6:17]1"
)
_GONANE_IDX_TO_IUPAC: dict[int, int] = {
    atom.GetIdx(): atom.GetAtomMapNum()
    for atom in _GONANE_PATTERN.GetAtoms()
    if atom.GetAtomMapNum() > 0
}


def mol_to_fol_atoms(mol: Chem.Mol):
    """Convert an RDKit ``Mol`` into a first-order logic model at the atom level.

    Returns ``(atom_extensions, mol_extensions)`` where:
    - ``atom_extensions`` is a ``dict[str, list]``: unary predicates map to
      ``list[int]`` of atom indices; binary predicates map to
      ``list[tuple[int, int]]`` of (left, right) index pairs.
    - ``mol_extensions`` is a ``set[str]`` of molecule-level predicate names
      that hold for this molecule (e.g. ``net_charge_positive``).
    """
    atom_extensions: dict[str, list] = {}

    try:
        Chem.rdCIPLabeler.AssignCIPLabels(mol)
    except Exception as e:
        logging.error(
            "Failed to assign CIP labels to molecule, skipping "
            "chirality-related extensions: %s",
            e,
        )

    # For each atom: element symbol, charge, hydrogen counts, chirality
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol().lower()
        atom_extensions.setdefault(atom_symbol, []).append(atom_idx)

        charge = atom.GetFormalCharge()
        if charge != 0:
            for pred in [
                f"charge_{'n' if charge < 0 else 'p'}",
                f"charge{'_m' + str(-charge) if charge < 0 else str(charge)}",
            ]:
                atom_extensions.setdefault(pred, []).append(atom_idx)
        else:
            atom_extensions.setdefault("charge0", []).append(atom_idx)

        num_hs = atom.GetTotalNumHs(includeNeighbors=True)
        for pred in [f"has_{num_hs}_hs"] + [f"has_at_least_{n}_hs" for n in range(1, num_hs + 1)]:
            atom_extensions.setdefault(pred, []).append(atom_idx)

        if atom.HasProp("_CIPCode"):
            chiral_code = f'cip_code_{atom.GetProp("_CIPCode")}'
            atom_extensions.setdefault(chiral_code, []).append(atom_idx)

    # Bond predicates (symmetric)
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()

        bond_pred = f"b{bond.GetBondType()}"
        atom_extensions.setdefault(bond_pred, []).extend([(left, right), (right, left)])
        atom_extensions.setdefault("has_bond_to", []).extend([(left, right), (right, left)])

        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            stereo_pred = f"b{bond.GetStereo().name}"
            atom_extensions.setdefault(stereo_pred, []).extend([(left, right), (right, left)])

    # Rings.
    #   ring{N}(A1, …, AN) – A1…AN form an N-membered ring, listed in cyclic
    #                        order. Exactly one fact per ring (no rotations or
    #                        reflections). Only for N <= MAX_RING_SIZE.
    #   in_ring{N}(A)      – A belongs to some N-membered ring (N <= MAX_RING_SIZE).
    #   in_ring(A)         – A belongs to some ring of any size.
    in_ring_atoms: set[int] = set()
    in_ringN_atoms: dict[int, set[int]] = {}
    for ring in mol.GetRingInfo().AtomRings():
        n = len(ring)
        in_ring_atoms.update(ring)
        if n <= MAX_RING_SIZE:
            atom_extensions.setdefault(f"ring{n}", []).append(tuple(ring))
            in_ringN_atoms.setdefault(n, set()).update(ring)
    if in_ring_atoms:
        atom_extensions["in_ring"] = sorted(in_ring_atoms)
    for n, atoms in in_ringN_atoms.items():
        atom_extensions[f"in_ring{n}"] = sorted(atoms)

    # Steroid nucleus positions (steroid_1 … steroid_17)
    steroid_match = mol.GetSubstructMatch(_GONANE_PATTERN, useChirality=False)
    if steroid_match:
        for pat_idx, atom_idx in enumerate(steroid_match):
            iupac = _GONANE_IDX_TO_IUPAC.get(pat_idx)
            if iupac is not None:
                atom_extensions.setdefault(f"steroid_{iupac}", []).append(atom_idx)

    # Molecule-level (global) properties
    mol_extensions: set[str] = set()
    net_charge = Chem.GetFormalCharge(mol)
    if net_charge > 0:
        mol_extensions.add("net_charge_positive")
    elif net_charge < 0:
        mol_extensions.add("net_charge_negative")
    else:
        mol_extensions.add("net_charge_neutral")
    # aliphatic vs aromatic (defined as having at least one aromatic atom)
    if len(list(mol.GetAromaticAtoms())) > 0:
        mol_extensions.add("aromatic")
    else:        
        mol_extensions.add("aliphatic")

    return atom_extensions, mol_extensions

def mol_to_fol_fgs(mol: Chem.Mol):
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
        fg_extensions.setdefault(fg_type, []).append(node_id)
        #ring_size = node["RING"]
        #if ring_size != 0:
        #    fg_extensions.setdefault(f"fg_ring{ring_size}", []).append(node_id)

    for edge in augmented_graph["edge_info"]["atom_fg_lvl"]:
        left, right = edge.split("_")
        left = int(left)
        right = int(right)
        fg_extensions.setdefault("is_fg_neighbor", []).append((left, right))
        fg_extensions.setdefault("is_fg_neighbor", []).append((right, left))

    return fg_extensions

