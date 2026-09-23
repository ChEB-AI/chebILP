# Pipeline: for each ChEBI class, ask an LLM to choose auxiliary predicates that help
# ILP tell the class apart from other molecules.
# Each auxiliary predicate is a small
# self-contained Python program mapping an RDKit Mol to a predicate extension (see
# chebILP.predicate_generation.auxiliary_predicates for the contract, and chebILP.predicate_generation.auxiliary_generation for the
# shared generate -> validate -> store loop).
#
# Predicates are kept in ONE shared library. For each class we first RETRIEVE the most 
# relevant existing predicates (hybrid
# BM25 + dense, see chebILP.predicate_generation.predicate_retrieval) and offer them to the LLM, which may
# reuse any of them and/or invent new ones. Reused + new predicates are recorded for the
# class in class_map.json; new programs are added to the library. `build_bk
# --predicate_set llm_generated_fgs` then loads each class's chosen programs.
#
# Input to the LLM: the ChEBI class name/definition, its superclasses, the predicates
# already available from mol_to_fol_atoms (so it proposes complementary properties), and
# the retrieved reuse candidates from the shared library.
#
# Usage:
#   python -m chebILP.predicate_generation.generate_auxiliary_predicates \
#       --labels_file labels.txt \
#       --chebi_version 248 \
#       --n_predicates 8

import argparse
import os
import re
from typing import Literal

from dotenv import load_dotenv
from rdkit import Chem

from chebILP.predicate_generation.auxiliary_generation import (
    OUTPUT_CONTRACT,
    AuxiliaryGenerator,
    NewItem,
    Selection,
    format_candidates,
    format_parents,
    one_line,
)
from chebILP.predicate_generation.auxiliary_predicates import add_program_to_library, load_program_source
from chebILP.predicate_generation.predicate_retrieval import HybridPredicateRetriever


# Reuse the exact predicate reference the enhancement pipeline shows the model,
# so it knows what mol_to_fol_atoms already provides and avoids duplicating it.
_EXISTING_PREDICATES = """\
- has_atom(Molecule, Atom): Molecule contains Atom
- c(Atom), n(Atom), o(Atom), s(Atom), p(Atom), cl(Atom), br(Atom), f(Atom), i(Atom), se(Atom), ...: Atom type (element symbol, lowercase)
- charge0/charge_p/charge_n/charge1/charge_m1(Atom): formal charge of Atom
- has_X_hs(Atom) / has_at_least_X_hs(Atom): attached-hydrogen counts
- cip_code_R(Atom), cip_code_S(Atom): R/S CIP stereochemistry
- bSINGLE/bDOUBLE/bTRIPLE/bAROMATIC(Atom1, Atom2): bond type
- bSTEREOCIS/bSTEREOTRANS(Atom1, Atom2): cis/trans bond stereochemistry
- has_bond_to(Atom1, Atom2): any bond between two atoms
- in_ring(Atom), in_ringN(Atom): ring membership / membership of an N-membered ring, for
  every N that occurs. There is no N-ary ringN(A1..AN) naming a ring's atoms in order.
- net_charge_positive/negative/neutral(Molecule), aromatic/aliphatic(Molecule): molecule-level
- steroid_1..steroid_17(Atom): atom at a steroid-nucleus position\
"""

_SYSTEM_PROMPT = """\
You are an expert in cheminformatics, RDKit, and Inductive Logic Programming (ILP).
Your task is to select AUXILIARY PREDICATES that help an ILP system distinguish a
ChEBI chemical class from other molecules.

You will be shown some potentially relevant pre-existing auxiliary predicates for the target class. 
Prefer to REUSE those that fit rather than writing new ones; only invent NEW predicates for
properties the reuse candidates do not already cover.

An auxiliary predicate is a short, self-contained Python program that maps an RDKit
Mol object to a predicate extension. Two kinds are useful:
  1. SHORTCUTS: functional groups / substructures that recur in this class
     (e.g. carboxylic acid, glycosidic bond) which would otherwise need many
     atom/bond literals to express.
  2. CONCEPTS: properties that are hard or impossible to express with the existing
     atom/bond predicates (e.g. "molecule has exactly 40 carbon atoms",
     "molecular weight above 500", "contains at least 3 rings").

Each NEW predicate has four parts, which you supply as separate fields:

    name:        snake_case_name
    description: <one short line describing what the predicate captures>
    kind:        one of "atom_unary", "atom_binary", "molecule"
    program:     def extension(mol):
                     # mol is an RDKit Mol object (already sanitized).
                     # kind == "molecule"    -> return a bool (does the property hold?)
                     # kind == "atom_unary"  -> return a list of 0-based atom indices
                     # kind == "atom_binary" -> return a list of (i, j) 0-based index pairs
                     ...

"program" holds ONLY the function. Do not restate the name, description or kind inside it.

Two worked examples — a molecule-level predicate and an atom_binary one:

    name:        carbon_chain_23_27
    description: Molecule has a carbon chain of length 23 to 27.
    kind:        molecule
    program:
        def extension(mol):
            max_len = 0
            for atom in mol.GetAtoms():
                queue = [(atom, 0, {atom.GetIdx()})]  # (atom, length so far, visited)
                while queue:
                    cur, length, visited = queue.pop(0)
                    for nb in cur.GetNeighbors():
                        if nb.GetAtomicNum() == 6 and nb.GetIdx() not in visited:
                            max_len = max(max_len, length + 1)
                            if max_len > 27:
                                return False
                            queue.append((nb, length + 1, visited | {nb.GetIdx()}))
            return 23 <= max_len <= 27

    name:        amide_bond
    description: Pairs of atoms joined by an amide bond: (carbonyl carbon, amide nitrogen).
    kind:        atom_binary
    program:
        def extension(mol):
            pairs = []
            patt = Chem.MolFromSmarts("[CX3](=O)[NX3]")
            for c_idx, o_idx, n_idx in mol.GetSubstructMatches(patt):
                pairs.append((c_idx, n_idx))  # return the C-N pair, not the =O oxygen
            return pairs

Rules:
- Use only the RDKit API. `Chem`, `rdMolDescriptors`, and `Descriptors` are already
  in scope; you may also `from rdkit ... import` inside the function if needed.
- Return atom INDICES (mol.GetAtomWithIdx(i)), never RDKit atom objects.
- Never raise; return an empty list / False when the property does not apply.
- Keep each program to a single predicate. Do not print or read files.
- Prefer predicates that are TRUE for many molecules of the target class and
  FALSE for other molecules.\
"""


class NewPredicate(NewItem):
    """A Python auxiliary predicate; ``kind`` decides how its extension is read."""

    kind: Literal["atom_unary", "atom_binary", "molecule"]


class PredicateSelection(Selection):
    new: list[NewPredicate]


# A few structurally diverse molecules used to smoke-test generated programs.
_TEST_SMILES = [
    "CCO",  # ethanol
    "c1ccccc1",  # benzene
    "CC(=O)O",  # acetic acid
    "OC(=O)CCC(N)C(=O)O",  # glutamic acid-ish
    "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",  # a steroid
]
_TEST_MOLS = [m for m in (Chem.MolFromSmiles(s) for s in _TEST_SMILES) if m is not None]

# Header lines the model may restate inside "program"; the pipeline synthesizes them.
_HEADER_RE = re.compile(r"^\s*(PREDICATE_NAME|KIND)\s*=", re.IGNORECASE)
_DOCSTRING_RE = re.compile(r'^(?:"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\')\s*')


def _validate(source: str, source_label: str) -> tuple[bool, str]:
    """Compile a program and run it against the smoke-test molecules."""
    pred = load_program_source(source, source_file=source_label)
    if pred is None:
        return False, "does not satisfy the program contract"
    for mol in _TEST_MOLS:
        try:
            result = pred.fn(mol)
        except Exception as e:
            return False, f"crashed on a test molecule ({e})"
        if pred.kind == "molecule":
            bool(result)
        else:
            try:
                list(result)
            except TypeError:
                return False, f"{pred.kind} did not return an iterable"
    return True, "validated"


class PredicateGenerator(AuxiliaryGenerator):
    """Auxiliary predicates written as Python programs over an RDKit Mol."""

    noun = "predicate"
    selection_model = PredicateSelection

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_retriever(self):
        return HybridPredicateRetriever.from_library(base_dir=self.library_dir)

    def build_user_prompt(self, chebi_id, info, candidates, ctx) -> str:
        definition_str = f"Definition: {info['definition']}\n" if info["definition"] else ""
        return f"""\
Target class: {info['name']} (CHEBI:{chebi_id})
{definition_str}{format_parents(info)}
Predicates ALREADY available (do NOT duplicate these):
{_EXISTING_PREDICATES}

{format_candidates(candidates, with_kind=True)}
Choose up to {self.n_predicates} auxiliary predicates that would help distinguish
"{info['name']}" from other molecules. REUSE the candidates above wherever they fit;
only write NEW programs for properties they do not already cover. Favour a mix of
shortcut substructures and hard-to-express global concepts grounded in the definition.

{OUTPUT_CONTRACT}
  "kind" is one of "atom_unary", "atom_binary", "molecule".
  "program" is the `def extension(mol):` function only.
"""

    def to_source(self, item) -> str:
        body = "\n".join(l for l in item.program.splitlines() if not _HEADER_RE.match(l)).strip()
        body = _DOCSTRING_RE.sub("", body)
        description = one_line(item.description).replace('"""', "'''")
        return (
            f'"""{description}"""\n'
            f'PREDICATE_NAME = "{item.name}"\n'
            f'KIND = "{item.kind}"\n\n'
            f"{body}\n"
        )

    def accept(self, source, label, ctx) -> tuple[bool, str]:
        return _validate(source, label)

    def add_to_library(self, source, chebi_id):
        return add_program_to_library(source, problem_dir=self.library_dir, chebi_id=chebi_id)

    def retriever_entry(self, stem, saved) -> dict:
        return {"name": saved.name, "description": saved.description, "kind": saved.kind, "stem": stem}

    def describe(self, saved, stem, reason) -> str:
        return f"{saved.name} ({saved.kind}) -> library/{stem}.py: {saved.description}"


def main():
    parser = argparse.ArgumentParser(description="Generate LLM auxiliary predicates for ChEBI classes.")
    parser.add_argument("--labels_file", required=True, help="File with one ChEBI ID per line.")
    parser.add_argument("--chebi_version", type=int, default=248)
    parser.add_argument("--predicate_dir", default=os.path.join("data", "llm_generated_predicates"),
                        help="Directory for storing generated predicates.")
    parser.add_argument("--model", default="claude-haiku-4-5",
                        help="A bare Claude id (claude-opus-5, claude-haiku-4-5) runs through the "
                             "logged-in `claude` CLI and bills your Claude subscription (set "
                             "CHEBILP_CLAUDE_CLI if it is not on PATH). A 'provider/name' id "
                             "(openai/gpt-4o) runs through an OpenAI-compatible endpoint set by "
                             "OPENAI_API_BASE and OPENAI_API_KEY.")
    parser.add_argument("--n_predicates", type=int, default=4, help="Target number of predicates per class.")
    parser.add_argument("--top_k", type=int, default=16,
                        help="Reuse candidates retrieved from the shared library per class.")
    args = parser.parse_args()

    load_dotenv()

    import pickle as _pickle

    data_dir = os.path.join("data", f"chebi_v{args.chebi_version}")
    with open(os.path.join(data_dir, "chebi_graph.pkl"), "rb") as _f:
        chebi_graph = _pickle.load(_f)

    with open(args.labels_file) as f:
        chebi_ids = [line.strip() for line in f if line.strip()]

    PredicateGenerator(
        args.predicate_dir, args.model, args.n_predicates, args.top_k,
    ).run(chebi_graph, chebi_ids)


if __name__ == "__main__":
    main()
