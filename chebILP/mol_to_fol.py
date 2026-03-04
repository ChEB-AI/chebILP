"""Convert RDKit molecules to first-order logic predicate extensions.

Ported from chemlog.preprocessing.mol_to_fol — only the atom-level
conversion (``mol_to_fol_atoms``) is included here.
"""

import logging

import numpy as np
from rdkit import Chem


def mol_to_fol_atoms(mol: Chem.Mol):
    """Convert an RDKit ``Mol`` into a first-order logic model at the atom level.

    Returns ``(universe_size, extensions)`` where *universe_size* is the number
    of atoms plus one (for a *global* pseudo-element) and *extensions* is a
    ``dict[str, np.ndarray]``.  Unary predicates are stored as 1-D boolean
    arrays; binary predicates as 2-D boolean arrays.
    """
    universe = mol.GetNumAtoms() + 1
    extensions: dict[str, np.ndarray] = {
        "EQ": np.array(
            [[i == j for i in range(universe)] for j in range(universe)]
        ),
        "atom": np.ones(universe, dtype=np.bool_),
    }
    extensions["atom"][-1] = False  # last position is global, not an atom

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
        if atom_symbol not in extensions:
            extensions[atom_symbol] = np.zeros(universe, dtype=np.bool_)
        extensions[atom_symbol][atom_idx] = True

        charge = atom.GetFormalCharge()
        if charge != 0:
            for predicate_symbol_charge in [
                f"charge_{'n' if charge < 0 else 'p'}",
                f"charge{'_m' + str(-1 * charge) if charge < 0 else str(charge)}",
            ]:
                if predicate_symbol_charge not in extensions:
                    extensions[predicate_symbol_charge] = np.zeros(
                        universe, dtype=np.bool_
                    )
                extensions[predicate_symbol_charge][atom_idx] = True
        else:
            predicate_symbol_charge = "charge0"
            if predicate_symbol_charge not in extensions:
                extensions[predicate_symbol_charge] = np.zeros(
                    universe, dtype=np.bool_
                )
            extensions[predicate_symbol_charge][atom_idx] = True

        # Hydrogen count predicates
        if universe != 1 or atom.GetAtomicNum() != 1:
            num_hs = atom.GetTotalNumHs(includeNeighbors=True)
            predicate_symbols = [f"has_{num_hs}_hs"] + [
                f"has_at_least_{n}_hs" for n in range(1, num_hs + 1)
            ]
            for predicate_symbol in predicate_symbols:
                if predicate_symbol not in extensions:
                    extensions[predicate_symbol] = np.zeros(
                        universe, dtype=np.bool_
                    )
                extensions[predicate_symbol][atom_idx] = True

        # CIP chirality
        if atom.HasProp("_CIPCode"):
            chiral_code = f'cip_code_{atom.GetProp("_CIPCode")}'
            if chiral_code not in extensions:
                extensions[chiral_code] = np.zeros(universe, dtype=np.bool_)
            extensions[chiral_code][atom_idx] = True

    # Bond predicates (symmetric)
    for bond in mol.GetBonds():
        predicate_symbol = f"b{bond.GetBondType()}"
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()

        if predicate_symbol not in extensions:
            extensions[predicate_symbol] = np.zeros(
                (universe, universe), dtype=np.bool_
            )
        extensions[predicate_symbol][left][right] = True
        extensions[predicate_symbol][right][left] = True

        # generic has_bond_to
        predicate_symbol = "has_bond_to"
        if predicate_symbol not in extensions:
            extensions[predicate_symbol] = np.zeros(
                (universe, universe), dtype=np.bool_
            )
        extensions[predicate_symbol][left][right] = True
        extensions[predicate_symbol][right][left] = True

        # stereo
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            stereo_pred = f"b{bond.GetStereo().name}"
            if stereo_pred not in extensions:
                extensions[stereo_pred] = np.zeros(
                    (universe, universe), dtype=np.bool_
                )
            extensions[stereo_pred][left][right] = True
            extensions[stereo_pred][right][left] = True

    # Global properties (last element in universe)
    extensions["net_charge_positive"] = np.zeros(universe, dtype=np.bool_)
    extensions["net_charge_negative"] = np.zeros(universe, dtype=np.bool_)
    extensions["net_charge_neutral"] = np.zeros(universe, dtype=np.bool_)
    extensions["global"] = np.zeros(universe, dtype=np.bool_)

    extensions["net_charge_positive"][-1] = Chem.GetFormalCharge(mol) > 0
    extensions["net_charge_negative"][-1] = Chem.GetFormalCharge(mol) < 0
    extensions["net_charge_neutral"][-1] = Chem.GetFormalCharge(mol) == 0
    extensions["global"][-1] = True

    return universe, extensions
