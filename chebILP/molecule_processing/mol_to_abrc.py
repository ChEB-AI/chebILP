"""
Based on LLM-generated molecule formalisation. Includes atoms, bonds, rings and components.
"""

import os

from chebILP.evaluation.clingo_eval import ground_extensions
from chebILP.predicate_generation.auxiliary_rules import DEFAULT_GROUNDING_TIMEOUT
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


_BOND_TYPES = {
    Chem.BondType.SINGLE: "single",
    Chem.BondType.DOUBLE: "double",
    Chem.BondType.TRIPLE: "triple",
    Chem.BondType.AROMATIC: "aromatic",
    Chem.BondType.DATIVE: "dative",
}

# Predicates whose argument is a constant (element, charge, count, size, ...). Popper cannot bind
# constants, so they stay in the BK for grounding the unary unfoldings in abrc_core.pl but are not
# offered as body predicates.
_CONSTANT_ARG_PREDICATES = {
    "element", "charge", "hcount", "degree", "isotope", "radical", "cip", "bond", "bond_cip",
    "ring", "steroid_pos", "net_charge", "elem_count", "heavy_atoms", "num_rings", "mol_weight",
    "num_components", "heavy_degree", "c_neighbors", "hetero_neighbors", "amine_class",
    "ring_size_atom", "ring_has_el", "has_element", "carbon_count", "n_carboxylic_acid",
    "n_carboxylate", "n_carboxy", "ring_system_size",
}
# Element lookup tables of abrc_core.pl; needed for grounding only, so neither written nor offered.
_LOOKUP_PREDICATES = {"halogen_el", "chalcogen_el", "pnictogen_el", "metal_el", "alkali_el"}

_GONANE = Chem.MolFromSmarts(
    "[#6:13]12~[#6:12]~[#6:11]~[#6:9]3~[#6:10]4~"
    "[#6:1]~[#6:2]~[#6:3]~[#6:4]~[#6:5]~4~"
    "[#6:6]~[#6:7]~[#6:8]~3~[#6:14]~2~"
    "[#6:15]~[#6:16]~[#6:17]~1"
)
_GONANE_POS = {a.GetIdx(): a.GetAtomMapNum() for a in _GONANE.GetAtoms()}


def _elem(atom):
    if atom.GetAtomicNum() == 0:
        return "r"
    return atom.GetSymbol().lower()


def mol_to_abrc_facts(mol_id, mol):
    """Return the ASP facts (list of strings) describing ``mol`` under id ``mol_id``."""
    M = str(mol_id)
    aid = lambda i: f"a{M}_{i + 1}"
    facts = [f"mol({M})."]
    try:
        # freeze each heavy atom's H count first: RemoveHs(sanitize=False) does not always transfer
        # removed H neighbours to the heavy atom (e.g. [H][PH-] became P with 0 H)
        mol = Chem.Mol(mol)
        mol.UpdatePropertyCache(strict=False)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                atom.SetNumExplicitHs(atom.GetTotalNumHs())
                atom.SetNoImplicit(True)
    except Exception:
        pass
    try:
        # normalise: explicit (non-isotopic, non-essential) H atoms become implicit
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        pass
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        mol.GetRingInfo().NumRings()
    except Exception:  # ring info not initialised (unsanitisable molecule)
        Chem.GetSymmSSSR(mol)
    try:
        Chem.rdCIPLabeler.AssignCIPLabels(mol)
    except Exception:
        pass

    elem_counts = {}
    total_h = 0
    heavy = 0
    for atom in mol.GetAtoms():
        a = aid(atom.GetIdx())
        e = _elem(atom)
        facts.append(f"has_atom({M},{a}).")
        facts.append(f"element({a},{e}).")
        facts.append(f"charge({a},{atom.GetFormalCharge()}).")
        nh = atom.GetTotalNumHs(includeNeighbors=True)
        facts.append(f"hcount({a},{nh}).")
        facts.append(f"degree({a},{atom.GetDegree()}).")
        if atom.GetIsAromatic():
            facts.append(f"aromatic({a}).")
        if atom.GetIsotope():
            facts.append(f"isotope({a},{atom.GetIsotope()}).")
        if atom.GetNumRadicalElectrons():
            facts.append(f"radical({a},{atom.GetNumRadicalElectrons()}).")
        if atom.HasProp("_CIPCode"):
            code = atom.GetProp("_CIPCode")
            lab = {"R": "r", "S": "s", "r": "pr", "s": "ps"}.get(code)
            if lab:
                facts.append(f"cip({a},{lab}).")
        if e == "h":
            total_h += 1  # explicit H atom (counted once here; not in neighbours' implicit Hs)
        else:
            heavy += 1
            total_h += atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
            elem_counts[e] = elem_counts.get(e, 0) + 1

    # double-bond E/Z: the stored mols carry bond directions but no perceived bond stereo, and the
    # new CIP labeler sets no bond labels then; perceive it on a copy (atom stereo stays untouched)
    bond_ez = {}
    try:
        m2 = Chem.Mol(mol)
        Chem.AssignStereochemistry(m2, cleanIt=True, force=True)
        for b in m2.GetBonds():
            st = b.GetStereo()
            if st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
                bond_ez[b.GetIdx()] = "e" if st == Chem.BondStereo.STEREOE else "z"
    except Exception:
        pass

    for bond in mol.GetBonds():
        i, j = aid(bond.GetBeginAtomIdx()), aid(bond.GetEndAtomIdx())
        t = _BOND_TYPES.get(bond.GetBondType(), "other")
        facts.append(f"bond({i},{j},{t}).")
        facts.append(f"bond({j},{i},{t}).")
        if bond.IsInRing():
            facts.append(f"ring_bond({i},{j}).")
            facts.append(f"ring_bond({j},{i}).")
        code = bond.GetProp("_CIPCode").lower() if bond.HasProp("_CIPCode") else bond_ez.get(bond.GetIdx())
        if code:
            if code in ("e", "z"):
                facts.append(f"bond_cip({i},{j},{code}).")
                facts.append(f"bond_cip({j},{i},{code}).")

    ri = mol.GetRingInfo()
    in_ring = set()
    for k, ring in enumerate(ri.AtomRings()):
        r = f"r{M}_{k + 1}"
        facts.append(f"ring_of({M},{r}).")
        facts.append(f"ring({r},{len(ring)}).")
        for i in ring:
            facts.append(f"ring_atom({r},{aid(i)}).")
        in_ring.update(ring)
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            facts.append(f"aromatic_ring({r}).")
    for i in sorted(in_ring):
        facts.append(f"in_ring({aid(i)}).")

    match = mol.GetSubstructMatch(_GONANE)
    for p, i in enumerate(match):
        facts.append(f"steroid_pos({aid(i)},{_GONANE_POS[p]}).")

    facts.append(f"net_charge({M},{Chem.GetFormalCharge(mol)}).")
    if total_h:
        elem_counts["h"] = total_h
    for e, n in sorted(elem_counts.items()):
        facts.append(f"elem_count({M},{e},{n}).")
    facts.append(f"heavy_atoms({M},{heavy}).")
    facts.append(f"num_rings({M},{ri.NumRings()}).")
    try:
        mw = round(Chem.Descriptors.MolWt(mol))
    except Exception:
        mw = 0
    facts.append(f"mol_weight({M},{mw}).")
    frags = Chem.GetMolFrags(mol)
    facts.append(f"num_components({M},{len(frags)}).")
    for k, frag in enumerate(frags):
        c = f"c{M}_{k + 1}"
        facts.append(f"component({M},{c}).")
        for i in frag:
            facts.append(f"in_component({aid(i)},{c}).")

    body_predicates = dict()
    for fact in facts:
        pred = fact.split("(")[0]
        arity = fact.count(",") + 1
        body_predicates[pred] = arity
    return facts, body_predicates


def mol_to_abrc_core(mol_id, mol):
    """Add predicates from core.pl to the basic facts from ``mol_to_abrc_facts``."""
    from chebILP.evaluation.clingo_eval import ground_extensions_isolated

    facts, body_predicates = mol_to_abrc_facts(mol_id, mol)

    rules = []
    with open (os.path.join(os.path.dirname(__file__), "abrc_core.pl"), "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("%"):
                rules.append(line)
    rules_joined = "\n".join(rules)
    rules_split = ["%s." % f.strip() for f in rules_joined.split(".") if f.strip()]
    for rule in rules_split:
        pred = rule.split("(")[0]
        head = rule.split(":-")[0] if ":-" in rule else rule
        arity = head.count(",") + 1
        body_predicates[pred] = arity

    grounded_rules = ground_extensions(rules_split, facts, list(body_predicates.keys()), 
                                                timeout=DEFAULT_GROUNDING_TIMEOUT)
    grounding_str = []
    for name, extensions in grounded_rules.items():
        if name in _LOOKUP_PREDICATES:
            continue
        for ext in extensions:
            grounding_str.append(f"{name}({','.join(ext)}).")

    body_predicates = {p: a for p, a in body_predicates.items()
                       if p not in _CONSTANT_ARG_PREDICATES and p not in _LOOKUP_PREDICATES}
    return facts + grounding_str, body_predicates


def build_background_abrc(rows) -> tuple[list[str], set[(str, int)]]:
    lines, predicates = [], dict()
    for row in rows.itertuples():
        mol_id, mol = row.Index, row.mol
        lines_mol, predicates_mol = mol_to_abrc_core(mol_id, Chem.Mol(mol))
        lines += lines_mol
        predicates.update(predicates_mol)
    # sort lines alphabetically to ensure facts are grouped by predicate
    lines.sort()
    return lines, [(pred, predicates[pred]) for pred in predicates.keys()]

if __name__ == "__main__":
    from rdkit import Chem

    smiles = "CC1=CC(=O)C2=C(C1=O)C=CC=C2"
    mol = Chem.MolFromSmiles(smiles)
    facts, body_predicates = mol_to_abrc_core("test", mol)
    for f in facts:
        print(f)
    #print("Body predicates:", body_predicates)