import os

from rdkit import Chem


def mol_to_smarts(mol: Chem.Mol) -> str:
    """Convert a molecule to a SMARTS string."""
    return Chem.MolToSmarts(mol)

def build_fowl_smarts(molecules_df, target_labels, output_path):
    """ 
    Build predicates directly from SMILES strings with wildcards. Will skip target labels that don't have a molecule with wildcard (*/R) attached.
    Outputs a csv file containing chebi_id, SMARTS.
    """
    smarts = dict()
    for chebi_id in target_labels:
        mol = molecules_df.loc[chebi_id, "mol"]
        if mol is None:
            continue
        if not any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            continue
        smarts[chebi_id] = mol_to_smarts(mol)

    with open(output_path, "w") as f:
        f.write("chebi_id,SMARTS\n")
        for chebi_id, smarts_str in smarts.items():
            f.write(f"{chebi_id},{smarts_str}\n")


def build_fowl_predicate(smarts: str, chebi_id: str):
    # predicate has arity number of wildcard atoms in the SMARTS
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise ValueError(f"Invalid SMARTS string: {smarts}")
    num_wildcards = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    predicate_name = f"fowl_{chebi_id}"
    return predicate_name, num_wildcards 

def calculate_fowl_predicate(smarts: str, mol: Chem.Mol):
    # check if molecule matches the SMARTS pattern
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        raise ValueError(f"Invalid SMARTS string: {smarts}")
    extensions = []
    wildcard_positions = [atom.GetIdx() for atom in pattern.GetAtoms() if atom.GetAtomicNum() == 0]
    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        extensions.append(tuple(match[idx] for idx in wildcard_positions))
    return extensions


def main(argv=None):
    """CLI: build the fowl SMARTS table used by the 'fowl' predicate set.

    Reads molecules from a pickle and a list of target ChEBI IDs, then writes one
    ``chebi_id,SMARTS`` row per class whose molecule carries a ``*``/R wildcard
    (classes without one are skipped). The output CSV is what
    ``ilp_problem_builder.load_fowl_smarts`` consumes.
    """
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Build the fowl SMARTS table (chebi_id,SMARTS) for the 'fowl' "
                    "predicate set.",
    )
    parser.add_argument("--molecules_path", type=str, required=True,
                        help="Path to molecules.pkl (index = ChEBI IDs, 'mol' column).")
    parser.add_argument("--labels_file", type=str, required=True,
                        help="File with one target ChEBI ID per line.")
    parser.add_argument("--output_path", type=str,
                        default=os.path.join("data", "fowl_smarts.csv"),
                        help="Destination CSV (default: data/fowl_smarts.csv).")
    args = parser.parse_args(argv)

    molecules_df = pd.read_pickle(args.molecules_path)
    with open(args.labels_file, "r") as f:
        target_labels = [line.strip() for line in f if line.strip()]

    # Labels are read as strings, but molecules.pkl may be indexed by int; match to
    # the index dtype and drop labels with no molecule row so .loc can't KeyError.
    index_set = set(molecules_df.index)
    matched = []
    for label in target_labels:
        if label in index_set:
            matched.append(label)
        elif label.isdigit() and int(label) in index_set:
            matched.append(int(label))
    missing = len(target_labels) - len(matched)
    if missing:
        print(f"{missing} of {len(target_labels)} label(s) not found in {args.molecules_path}; skipped.")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    build_fowl_smarts(molecules_df, matched, args.output_path)

    with open(args.output_path) as f:
        n_written = sum(1 for _ in f) - 1  # minus header
    print(f"Wrote {n_written} fowl SMARTS pattern(s) to {args.output_path} "
          f"(from {len(matched)} class(es) with a molecule).")


if __name__ == "__main__":
    main()

