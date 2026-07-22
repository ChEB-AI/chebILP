import os
from typing import Literal
import networkx as nx

import tqdm
from chebILP.molecule_processing.data_preparation import ChEBIDataset
from chebILP.molecule_processing.mol_to_fol import mol_to_fol_atoms, mol_to_fol_fgs
from chebILP.predicate_generation.auxiliary_predicates import load_auxiliary_predicates, compute_auxiliary_extensions, DEFAULT_AUX_TIMEOUT
from chebILP.predicate_generation.auxiliary_rules import derive_rule_extensions, load_class_rules
from chebILP.molecule_processing.fg_matching import get_chembl_fgs, get_chebi_fgs
from chebILP.molecule_processing.fowl_predicates import build_fowl_predicate, calculate_fowl_predicate
import pandas as pd
from chebILP.utils import AVAILABLE_PREDICATE_SETS, get_atom_id
from chebILP.ilp_path_manager import get_bk_path, get_bias_path, get_exs_path
from chebILP.evaluation.clingo_eval import evaluate_with_clingo


CHEBI_FG_RULES_PATH = os.path.join("data", "chebi_fg_rules_from_smiles.pl")
CHEBI_FG_LEARNED_RULES_PATH = os.path.join("data", "chebi_fg_learned_rules.pl")
# SMARTS patterns (one per ChEBI class that has a wildcard-bearing molecule) used
# by the "fowl" predicate set, produced by fowl_predicates.build_fowl_smarts.
FOWL_SMARTS_PATH = os.path.join("data", "fowl_smarts.csv")


def load_fowl_smarts(path=FOWL_SMARTS_PATH) -> dict[str, str]:
    """Load the fowl SMARTS CSV (``chebi_id,SMARTS``) into ``{chebi_id: smarts}``.

    Returns an empty dict if the file is missing so ``build_bk`` degrades to the
    plain atom predicates for classes without a fowl pattern. Only a subset of
    classes have an entry (those whose molecule carries a ``*``/R wildcard).
    """
    if not os.path.exists(path):
        return {}
    mapping: dict[str, str] = {}
    with open(path, "r") as f:
        next(f, None)  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            chebi_id, smarts = line.split(",", 1)
            mapping[chebi_id.strip()] = smarts.strip()
    return mapping

class ILPProblemBuilder:

    def __init__(self, chebi_version: int, three_star_only: bool = True, base_dir: str = "data", min_pos_samples: int = 25, predicate_set: AVAILABLE_PREDICATE_SETS = "atoms", aux_timeout: float = DEFAULT_AUX_TIMEOUT, aux_library_dir: str | None = None, computed_facts: bool = False):
        self.predicate_set = predicate_set
        self.problem_dir = os.path.join(base_dir, "ilp_problems")
        os.makedirs(self.problem_dir, exist_ok=True)
        # Per-call wall-clock budget for LLM-generated auxiliary predicates.
        self.aux_timeout = aux_timeout
        self.aux_library_dir = aux_library_dir
        # When set, molecular-weight and ring-size facts are computed and made
        # available to llm_generated_rules during rule evaluation, but never
        # written to bk.pl (only the derived aux_* extensions are).
        self.computed_facts = computed_facts

        # --- Load pre-built ChEBI data -------------------------------------
        self.dataset = ChEBIDataset(chebi_version=chebi_version, three_star_only=three_star_only, base_dir=base_dir, min_pos_samples=min_pos_samples)
        self.hierarchy_graph = nx.transitive_closure_dag(self.dataset.chebi_graph)
        self.undirected_graph = self.dataset.chebi_graph.to_undirected()
        self.splits = self.dataset.load_splits_from_csv() 
            
            
    def build_examples(self, target_ids: list[str], min_pos_samples=25, max_pos_samples=200, min_neg_samples=25, max_neg_samples=200):
        min_n_pos = max_pos_samples + 1
        min_n_pos_id = None
        min_n_neg = max_neg_samples + 1
        min_n_neg_id = None
        for target_id in tqdm.tqdm(target_ids, desc="Building examples for ChEBI classes"):
            n_pos, n_neg = self.gather_samples_for_chebi_cls(target_id, min_pos_samples, max_pos_samples, min_neg_samples, max_neg_samples)
            if n_pos < min_n_pos:
                min_n_pos = n_pos
                min_n_pos_id = target_id
            if n_neg < min_n_neg:
                min_n_neg = n_neg
                min_n_neg_id = target_id
        print(f"Label with least positive samples: ChEBI:{min_n_pos_id} with {min_n_pos} samples")
        print(f"Label with least negative samples: ChEBI:{min_n_neg_id} with {min_n_neg} samples")


    def build_bk(self, target_ids):
        """
        Build ILP background knowledge.

        Args:
            """

        rules, rule_predicates = [], []
        if self.predicate_set in ["chebi_fg_rules", "chebi_fg_learned_rules"]:
            prolog_lines_rules, body_predicates_rules = build_background_chebi_fg_rules(CHEBI_FG_RULES_PATH if self.predicate_set == "chebi_fg_rules" else CHEBI_FG_LEARNED_RULES_PATH)
            rules = prolog_lines_rules
            rule_predicates = body_predicates_rules
        
        for target_id in tqdm.tqdm(target_ids, desc="Building background knowledge for ChEBI classes"):
            print(f"Building background knowledge for ChEBI:{target_id}...")

            # LLM-generated auxiliary predicates are specific to the target class,
            # so they are loaded once per target and merged into the atom-level BK.
            aux_predicates = None
            if self.predicate_set == "llm_generated_fgs":
                aux_predicates = load_auxiliary_predicates(target_id, library_dir=self.aux_library_dir)
                print(f"  Loaded {len(aux_predicates)} auxiliary predicate(s) for ChEBI:{target_id}")

            # llm_generated_rules: the class's auxiliary predicates are ASP rules,
            # evaluated (below) against the atom facts plus optional computed facts.
            # Only the derived aux_* extensions are written to bk.pl.
            rule_programs = None
            if self.predicate_set == "llm_generated_rules":
                rule_programs = load_class_rules(target_id, library_dir=self.aux_library_dir)
                print(f"  Loaded {len(rule_programs)} auxiliary rule(s) for ChEBI:{target_id}")

            # The fowl set adds a single class-specific predicate, fowl_<target_id>,
            # derived from a SMARTS pattern, on top of the atom predicates. Not every
            # class has a pattern; those fall back to the plain atom predicates.
            fowl_smarts = None
            if self.predicate_set == "fowl":
                if not hasattr(self, "_fowl_smarts"):
                    self._fowl_smarts = load_fowl_smarts()
                smarts = self._fowl_smarts.get(target_id)
                if smarts is None:
                    print(f"  No fowl SMARTS for ChEBI:{target_id}; falling back to plain atom predicates.")
                else:
                    print(f"  Loaded fowl SMARTS for ChEBI:{target_id}: {smarts}")
                    fowl_smarts = {target_id: smarts}

            selected_ids_by_split = dict()
            prolog_lines_by_split = dict()
            computed_lines_by_split = dict()
            body_predicates = set()
            for split in ["train", "validation", "test"]:
                exs_path = get_exs_path(target_id, base_dir=self.problem_dir, split=split)
                with open(exs_path, "r") as f:
                    # for each line get id between inner parentheses (e.g. pos(chebi_123(456)). -> 456) and select corresponding rows from samples_df
                    selected_ids = [line.strip().split("(")[-1].split(")")[0] for line in f.readlines() if line.strip() and not line.startswith("%")]
                selected_rows = self.dataset.molecules[[id in selected_ids for id in self.dataset.molecules.index]]
                selected_ids_by_split[split] = selected_ids

                # standard bk is always added
                prolog_lines = []
                prolog_lines_atoms, body_predicates_atoms = build_background_chemlog(selected_rows, aux_predicates=aux_predicates, aux_timeout=self.aux_timeout, predicate_set=self.predicate_set, fowl_smarts=fowl_smarts)
                prolog_lines += prolog_lines_atoms
                body_predicates.update(body_predicates_atoms)
                if self.predicate_set in ["chembl_fgs", "chebi_fgs"]:
                    # add fgs as samples
                    if not hasattr(self, "_fg_data"):
                        if self.predicate_set == "chembl_fgs":
                            self._fg_data = get_chembl_fgs(self.dataset.molecules)
                        else:
                            self._fg_data = get_chebi_fgs(self.dataset.molecules)
                    prolog_lines_fgs, body_predicates_fgs = build_background_fg_data(self._fg_data, selected_rows, source=self.predicate_set)
                    prolog_lines += prolog_lines_fgs
                    body_predicates.update(body_predicates_fgs)
                prolog_lines_by_split[split] = prolog_lines

                # Computed facts (molecular weight, ring size) feed rule evaluation
                # only; they are intentionally kept out of prolog_lines (bk.pl).
                if self.predicate_set == "llm_generated_rules" and self.computed_facts:
                    computed_lines_by_split[split] = build_computed_facts(selected_rows)

            # for evaluating rules, merge alls splits, separate results afterwards
            if self.predicate_set in ["chebi_fg_rules", "chebi_fg_learned_rules"]:
                all_selected_ids = [id for split in ["train", "validation", "test"] for id in selected_ids_by_split[split]]
                all_prolog_lines = [line for split in ["train", "validation", "test"] for line in prolog_lines_by_split[split]]
                positives = evaluate_with_clingo(rules, all_prolog_lines, rule_predicates, all_selected_ids, list(body_predicates))
                for positive_extension in positives:
                    pred = positive_extension
                    in_split = {"train": False, "validation": False, "test": False}
                    for example in positives[positive_extension]:
                        for split in ["train", "validation", "test"]:
                            if example in selected_ids_by_split[split]:
                                if not in_split[split]:
                                    body_predicates.add((pred, 1))
                                    in_split[split] = True
                                prolog_lines_by_split[split].append(f"{pred}({example}).")

            # llm_generated_rules: ground each class rule against the atom facts plus
            # computed facts (all splits merged), then write only the derived aux_*
            # extensions back into each split. The computed facts are never written.
            if self.predicate_set == "llm_generated_rules" and rule_programs:
                all_selected_ids = [id for split in ["train", "validation", "test"] for id in selected_ids_by_split[split]]
                eval_facts = [line for split in ["train", "validation", "test"] for line in prolog_lines_by_split[split]]
                if self.computed_facts:
                    eval_facts += [line for split in ["train", "validation", "test"] for line in computed_lines_by_split.get(split, [])]
                # The class's rules are grounded as one program, so a rule may use a predicate
                # another of its rules defines. The head may be of any arity; each derived
                # atom is written to the split of the molecule it belongs to.
                extensions = derive_rule_extensions(rule_programs, eval_facts, all_selected_ids)
                for rp in rule_programs:
                    emitted = {split: set() for split in ["train", "validation", "test"]}
                    for example, arg_tuples in extensions.get(rp.name, {}).items():
                        for split in ["train", "validation", "test"]:
                            if example not in selected_ids_by_split[split]:
                                continue
                            for args in arg_tuples:
                                line = f"{rp.name}({','.join(args)})."
                                if line in emitted[split]:
                                    continue
                                emitted[split].add(line)
                                body_predicates.add((rp.name, len(args)))
                                prolog_lines_by_split[split].append(line)

            for split in ["train", "validation", "test"]:
                prolog_lines = prolog_lines_by_split[split]
                bk_path = get_bk_path(target_id, base_dir=self.problem_dir, predicate_set=self.predicate_set, split=split)

                with open(bk_path, "w+") as f:
                    f.write("\n".join(prolog_lines) + "\n")

            # create bias file template based on bk predicates
            plain_bias_path = get_bias_path(target_id, split="train", base_dir=self.problem_dir, predicate_set=self.predicate_set) # bias file path for settings-specific bias file (created in build_bias)
            bias_lines = [
                f"%% CHEBI:{target_id} (bias file without settings)",
                f"",
                f"%% max_vars(TODO).",
                f"%% max_body(TODO).",
                f"%% max_clauses(TODO).",
                f"",
                f"head_pred(chebi_{target_id}, 1)."] + [
                f"body_pred({pred},{arity})." for pred, arity in body_predicates
            ]
            # bias without settings (as template)
            with open(plain_bias_path, "w+") as f:
                f.write("\n".join(bias_lines) + "\n")


    def get_closest_negatives(self, samples: pd.DataFrame, target_id: str, min_samples=25, max_samples=None, direct_only=False):
        # goal: reach min_samples, but continue collecting samples (until max_samples) if they are siblings.
        # if direct_only=True, only collect from direct neighbors of target_id (first BFS ring) regardless of count.
        import queue
        q = queue.Queue()
        q.put(target_id)
        visited = set() # visit closest labels
        selected = set() # select samples that are subclasses of closest labels until we have enough samples
        samples_index = list(str(id) for id in samples.index)
        siblings = True
        while not q.empty():
            current = q.get()
            for neighbor in self.undirected_graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.put(neighbor)
                    for neighbor_sub in self.hierarchy_graph.predecessors(neighbor):
                        if str(neighbor_sub) in samples_index:
                            selected.add(str(neighbor_sub))
                        if (max_samples and len(selected) >= max_samples) or (len(selected) >= min_samples and not siblings):
                            return self.dataset.molecules.loc[[id in selected for id in self.dataset.molecules.index]]
            if len(selected) >= min_samples or direct_only:
                break
            siblings = False

        return self.dataset.molecules.loc[[str(id) in selected for id in self.dataset.molecules.index]]


    def build_negative_mix(self, neg_pool: pd.DataFrame, sibling_ids: set, max_samples: int, random_state: int = 42) -> pd.DataFrame:
        """50:50 mix of direct-sibling negatives and random negatives from ``neg_pool``.

        Up to half of ``max_samples`` are the target's direct siblings (near-misses); the rest
        are drawn uniformly at random from the non-sibling remainder. When a class has fewer
        siblings than half, the random draw takes up the slack rather than the set shrinking, so
        the objective is global classification instead of separation from the superclass alone.
        """
        half = max_samples // 2
        sibling_negs = neg_pool[neg_pool.index.astype(str).isin(sibling_ids)]
        if len(sibling_negs) > half:
            sibling_negs = sibling_negs.sample(half, random_state=random_state)
        random_pool = neg_pool[~neg_pool.index.astype(str).isin(sibling_ids)]
        n_random = min(max_samples - len(sibling_negs), len(random_pool))
        random_negs = random_pool.sample(n_random, random_state=random_state) if n_random > 0 else random_pool.iloc[:0]
        return pd.concat([sibling_negs, random_negs])

    def gather_samples_for_chebi_cls(self, target_id: str, min_pos_samples=25, max_pos_samples=200, min_neg_samples=25, max_neg_samples=200):
        descendants = list(self.hierarchy_graph.predecessors(target_id)) + [target_id]
        # not all descendants are molecules (i.e., have a SMILES annotation) -> only take the ones that are in the samples_df (i.e. have a SMILES annotation and are in the 3_STAR subset)

        df_pos = self.dataset.molecules[[id in descendants for id in self.dataset.molecules.index]]
        df_neg = self.dataset.molecules[[id not in df_pos.index for id in self.dataset.molecules.index]]
        if len(df_pos) < min_pos_samples:
            print(f"ChEBI class {target_id} does not have enough positive samples (found {len(df_pos)}, required are at least {min_pos_samples}). Got samples {df_pos.index.tolist()}")
        if len(df_neg) < min_neg_samples:
            print(f"ChEBI class {target_id} does not have enough negative samples (found {len(df_neg)}, required are at least {min_neg_samples}). Got samples {df_neg.index.tolist()}")
        
        # Direct-sibling molecules: subclasses shared with the target's parents. They form the
        # near-miss half of every split's negatives; the other half is drawn uniformly at random
        # from the full negative pool. The objective is therefore global classification, not
        # separating the target from its superclass only.
        pos_ids, sibling_neg_ids = get_direct_neighbors(target_id, self.dataset.chebi_graph, self.hierarchy_graph, self.dataset.molecules)
        sibling_neg_ids = set(sibling_neg_ids)

        samples_by_split = dict()
        pos_train_samples = df_pos[df_pos.index.astype(str).isin(self.splits["train"])]
        samples_by_split[("pos", "train")] = pos_train_samples.sample(min(max_pos_samples, len(pos_train_samples)), random_state=42) # if there are more positives than max_pos_samples, sample randomly
        neg_train_samples = df_neg[df_neg.index.astype(str).isin(self.splits["train"])]
        samples_by_split[("neg", "train")] = self.build_negative_mix(neg_train_samples, sibling_neg_ids, max_neg_samples)
        
        samples_by_split[("pos", "validation")] = df_pos[df_pos.index.astype(str).isin(self.splits["validation"]) & df_pos.index.astype(str).isin(pos_ids)]
        neg_val_samples = df_neg[df_neg.index.astype(str).isin(self.splits["validation"])]
        samples_by_split[("neg", "validation")] = self.build_negative_mix(neg_val_samples, sibling_neg_ids, max_neg_samples)
        samples_by_split[("pos", "test")] = df_pos[df_pos.index.astype(str).isin(self.splits["test"]) & df_pos.index.astype(str).isin(pos_ids)]
        neg_test_samples = df_neg[df_neg.index.astype(str).isin(self.splits["test"])]
        samples_by_split[("neg", "test")] = self.build_negative_mix(neg_test_samples, sibling_neg_ids, max_neg_samples)
        
        for (posneg, split), df in samples_by_split.items():
            exs_path = get_exs_path(target_id, base_dir=self.problem_dir, split=split)
            with open(exs_path, "w+" if posneg == "pos" else "a") as f:
                for sample in df.index:
                    f.write(f"{posneg}(chebi_{target_id}({sample})).\n")

        # sum up all positive and negative samples across splits
        return sum(len(v) for k, v in samples_by_split.items() if k[0] == "pos"), sum(len(v) for k, v in samples_by_split.items() if k[0] == "neg")
    

def get_direct_neighbors(
    target_id: str,
    chebi_graph,
    hierarchy_graph,
    molecules_df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Return the positive and negative samples for a target class, only considering descendants of its direct parents.

    Returns:
        pos_ids: list of positive validation molecule IDs
        neg_ids: list of negative validation molecule IDs
                 (empty when target has no siblings)
    """
    mol_index = set(str(idx) for idx in molecules_df.index)
    pos_ids = [
        str(d)
        for d in hierarchy_graph.predecessors(target_id)
        if str(d) in mol_index
    ]

    sample_space_by_parent = dict()
    for parent in chebi_graph.successors(target_id):
        sample_space_by_parent[parent] = set()
        for desc in hierarchy_graph.predecessors(parent):
            s = str(desc)
            if s in mol_index:
                sample_space_by_parent[parent].add(s)
    if len(sample_space_by_parent) == 0:
        return pos_ids, []
    sample_space = set.intersection(*sample_space_by_parent.values())
    neg_ids = list(sample_space - set(pos_ids))
    return pos_ids, neg_ids





def build_background_chemlog(rows, aux_predicates=None, aux_timeout=DEFAULT_AUX_TIMEOUT, aux_failures=None, predicate_set="atoms", fowl_smarts=None):
    comments = []
    lines_by_predicate, arities = {}, {}
    if "farm_fgs" in predicate_set:
        lines_by_predicate["has_fg"] = []
        arities["has_fg"] = 2
    if "atoms" in predicate_set or "farm_fgs" not in predicate_set:
        lines_by_predicate["has_atom"] = []
        arities["has_atom"] = 2 

    aux_ext_by_mol = {}
    if aux_predicates:
        aux_ext_by_mol = compute_auxiliary_extensions(
            aux_predicates,
            [(row.Index, row.mol) for row in rows.itertuples()],
            timeout=aux_timeout,
            failures=aux_failures,
        )

    for row in rows.itertuples():
        atom_extensions, fg_extensions, mol_extensions = {}, {}, set()
        if "farm_fgs" in predicate_set:
            # Functional-group level model: entities are FARM functional-group
            # nodes rather than atoms. has_fg links the molecule to its FG nodes.
            fg_extensions = mol_to_fol_fgs(row.mol, add_fg_atom_predicates="atoms" in predicate_set)
            node_ids = sorted({id for ids in fg_extensions.values()  for nid in ids for id in (nid if isinstance(nid, tuple) else (nid,))})  # flatten tuples
            for node_id in node_ids:
                 if node_id >= row.mol.GetNumAtoms():
                    fg_id = get_atom_id(node_id, row.Index)
                    lines_by_predicate["has_fg"].append(
                        f"has_fg({row.Index},{fg_id}).")
        if "atoms" in predicate_set or "farm_fgs" not in predicate_set:
            for atom in row.mol.GetAtoms():
                atom_id = get_atom_id(atom.GetIdx(), row.Index)
                lines_by_predicate["has_atom"].append(f"has_atom({row.Index},{atom_id}).")

            atom_extensions, mol_extensions = mol_to_fol_atoms(row.mol)

        # Merge LLM-generated auxiliary predicates. Their names are ``aux_``-prefixed,
        # so they never collide with the built-in extensions produced above.
        if aux_predicates:
            aux_atom_ext, aux_mol_ext = aux_ext_by_mol.get(row.Index, ({}, set()))
            atom_extensions.update(aux_atom_ext)
            mol_extensions.update(aux_mol_ext)

        # fowl: class-specific SMARTS-match predicates (fowl_<chebi_id>) added on
        # top of the atom predicates. Each match binds the pattern's wildcard
        # atoms, so the arity equals the number of wildcards; the tuples are
        # emitted as atom-id arguments by the extension loop below.
        if fowl_smarts:
            for cls_id, smarts in fowl_smarts.items():
                predicate_name, _ = build_fowl_predicate(smarts, cls_id)
                try:
                    matches = calculate_fowl_predicate(smarts, row.mol)
                except Exception as e:
                    print(f"Warning: failed to compute {predicate_name} for CHEBI:{row.Index}: {e}")
                    continue
                if matches:
                    atom_extensions.setdefault(predicate_name, []).extend(matches)

        for predicate, indices in {**atom_extensions, **fg_extensions}.items():
            if predicate.startswith("cip_code_"):
                predicate = "cip_code_" + predicate[-1].upper()
            if (predicate in {"EQ", "atom", "*", "r", "r#"} or (predicate.startswith("r") and predicate[1:].isdigit() and int(predicate[1:]) > 0) or not indices):
                continue
        
            is_tuple = isinstance(indices[0], tuple)
            if predicate not in lines_by_predicate:
                lines_by_predicate[predicate] = []
            if predicate not in arities:
                arities[predicate] = len(indices[0]) if is_tuple else 1
            if is_tuple:
                for args in indices:
                    arg_str = ",".join(get_atom_id(a, row.Index) for a in args)
                    lines_by_predicate[predicate].append(f"{predicate}({arg_str}).")
            else:
                for idx in indices:
                    lines_by_predicate[predicate].append(f"{predicate}({get_atom_id(idx, row.Index)}).")

        for predicate in mol_extensions:
            if predicate not in lines_by_predicate:
                lines_by_predicate[predicate] = []
            if predicate not in arities:
                arities[predicate] = 1
            lines_by_predicate[predicate].append(f"{predicate}({row.Index}).")

    return comments + [line for lines in lines_by_predicate.values() for line in lines], [(pred, arities[pred]) for pred in arities.keys()]


def build_computed_facts(rows):
    """Molecular-weight and ring-size facts used only to evaluate llm_generated_rules.

    Emits ``mol_weight(Mol, Wint)`` (rounded ``Descriptors.MolWt``) and one
    ``ring_size(Mol, Size)`` per SSSR ring. These facts are fed to Clingo when a
    class's auxiliary rules are grounded, but are never written to ``bk.pl`` — only
    the derived ``aux_*`` extensions are persisted. Returns a flat list of Prolog lines.
    """
    from rdkit.Chem import Descriptors

    lines = []
    for row in rows.itertuples():
        lines.append(f"mol_weight({row.Index},{round(Descriptors.MolWt(row.mol))}).")
        for ring in row.mol.GetRingInfo().AtomRings():
            lines.append(f"ring_size({row.Index},{len(ring)}).")
    return lines


def build_full_background(
    rows: pd.DataFrame,
    predicate_set: AVAILABLE_PREDICATE_SETS = "atoms",
    aux_predicates=None,
    aux_timeout: float = DEFAULT_AUX_TIMEOUT,
    aux_failures=None,
    fowl_smarts=None,
    rule_programs=None,
    computed_facts: bool = False,
) -> list[str]:
    """Build one flat background-knowledge fact list for the molecules in ``rows``.

    Mirrors :meth:`ILPProblemBuilder.build_bk` so prediction tensors are evaluated
    against exactly the same BK the programs were learned on, rather than always the
    plain ``atoms`` set. For the ``chebi_fg_rules`` / ``chebi_fg_learned_rules`` sets the
    functional-group rule clauses are added to the BK directly (rather than pre-evaluated
    into facts): the caller grounds and solves the combined program, which derives them.

    All work is scoped to ``rows``, so this can be called per molecule (e.g. to bound
    Clingo grounding memory). ``aux_predicates`` (for ``llm_generated_fgs``) are the
    name-deduplicated predicates gathered across all classes; their extensions are
    evaluated on ``rows`` here.
    """
    prolog_lines, _ = build_background_chemlog(
        rows, aux_predicates=aux_predicates, aux_timeout=aux_timeout, aux_failures=aux_failures,
        predicate_set=predicate_set, fowl_smarts=fowl_smarts,
    )
    prolog_lines = list(prolog_lines)

    if predicate_set in ("chembl_fgs", "chebi_fgs"):
        fg_data = get_chembl_fgs(rows) if predicate_set == "chembl_fgs" else get_chebi_fgs(rows)
        fg_lines, _ = build_background_fg_data(fg_data, rows, source=predicate_set)
        prolog_lines += fg_lines

    if predicate_set in ("chebi_fg_rules", "chebi_fg_learned_rules"):
        rule_lines, _ = build_background_chebi_fg_rules(
            CHEBI_FG_RULES_PATH if predicate_set == "chebi_fg_rules" else CHEBI_FG_LEARNED_RULES_PATH
        )
        prolog_lines += rule_lines

    # llm_generated_rules: recompute the class's aux_* extensions exactly as build_bk
    # does (ground each rule over atom + computed facts) and append them as facts, so a
    # learned program's aux_* body literals resolve. Computed facts stay local to the
    # grounding and are not added to the returned BK.
    if predicate_set == "llm_generated_rules" and rule_programs:
        eval_facts = list(prolog_lines)
        if computed_facts:
            eval_facts += build_computed_facts(rows)
        mol_ids = [str(i) for i in rows.index]
        extensions = derive_rule_extensions(rule_programs, eval_facts, mol_ids)
        for rp in rule_programs:
            emitted = set()
            for arg_tuples in extensions.get(rp.name, {}).values():
                for args in arg_tuples:
                    line = f"{rp.name}({','.join(args)})."
                    if line not in emitted:
                        emitted.add(line)
                        prolog_lines.append(line)

    return prolog_lines


def build_background_chebi_fg_rules(rules_path=None):
    """Load ChEBI functional group rules from a Prolog file and return them as BK lines and body predicates.
    
    Each rule defines a chebi_XXXXX(M) predicate in terms of atom-level predicates.
    These are added as Prolog rules to the BK and as body_pred entries (arity 1) in the bias.
    """
    if rules_path is None:
        rules_path = CHEBI_FG_RULES_PATH
    
    prolog_lines = [f"% ChEBI FG rules from {os.path.basename(rules_path)}"]
    body_predicates = []
    seen_predicates = set()
    
    with open(rules_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            prolog_lines.append(line)
            # Extract predicate name from head: chebi_XXXXX(M) :- ...
            pred_name = line.split("(")[0].strip()
            if pred_name and pred_name not in seen_predicates:
                seen_predicates.add(pred_name)
                body_predicates.append(pred_name)
    
    print(f"Loaded {len(body_predicates)} ChEBI FG rule predicates from {rules_path}")
    return prolog_lines, body_predicates


def build_background_fg_data(fg_data: dict[int, list[str]], rows, source: Literal["chembl_fgs", "chebi_fgs"]):
    lines_by_predicate = dict()

    for row in rows.itertuples():
        if row.Index not in fg_data:
            print(f"Warning: No functional group data found for CHEBI:{row.Index} in source {source}. This molecule will only have atom and bond predicates in the background knowledge.")
            continue
        for fg in fg_data[row.Index]:
            if fg not in lines_by_predicate:
                lines_by_predicate[fg] = []
            lines_by_predicate[fg].append(f"{fg}({row.Index}).")
    total_lines = [line for lines in lines_by_predicate.values() for line in lines]
    return total_lines, [(pred, 1) for pred in lines_by_predicate.keys()]


if __name__ == "__main__":
    builder = ILPProblemBuilder(
        chebi_version=251,
        predicate_set="atoms",
    )
    target_ids = ["134362"]
    builder.build_examples(target_ids)
