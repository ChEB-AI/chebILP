import os
from typing import Literal
import networkx as nx

import tqdm
from chebILP.molecule_processing.data_preparation import ChEBIDataset
from chebILP.molecule_processing.mol_to_fol import mol_to_fol_fgs
from chebi_utils.extract_properties import mol_to_fol_atoms, get_numerical_facts
from chebILP.predicate_generation.auxiliary_predicates import load_auxiliary_predicates, compute_auxiliary_extensions, DEFAULT_AUX_TIMEOUT
from chebILP.predicate_generation.auxiliary_rules import derive_rule_extensions, load_class_rules, resolve_rule_dependencies
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

    def __init__(self, chebi_version: int, three_star_only: bool = True, base_dir: str = "data", min_pos_samples: int = 25, predicate_set: AVAILABLE_PREDICATE_SETS = "atoms", aux_timeout: float = DEFAULT_AUX_TIMEOUT, aux_library_dir: str | None = None, computed_facts: bool = True):
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
        self.splits = self.dataset.load_splits_from_csv()

        # Invariant across target classes, so built once rather than per class.
        self._mol_index = set(self.dataset.molecules.index)
        self._split_ids = {split: set(self.splits[self.splits["split"] == split]["id"].astype(str))
                           for split in ["train", "validation", "test"]}
        # Size of the molecule graph, used to prefer small molecules when a split is
        # capped. This is the same count that drives the has_atom facts in bk.pl, so it
        # includes explicit hydrogens where a molecule carries them.
        self._atom_counts = self.dataset.molecules["mol"].map(lambda m: m.GetNumAtoms())
            
            
    def build_examples(self, target_ids: list[str], min_pos_samples=25, max_pos_samples=200, min_neg_samples=25, max_neg_samples=200):
        # Counts are summed over the three splits, so they are not comparable against the
        # per-split max_*_samples caps; take the minimum over what was actually written.
        counts = {}
        for target_id in tqdm.tqdm(target_ids, desc="Building examples for ChEBI classes"):
            counts[target_id] = self.gather_samples_for_chebi_cls(target_id, min_pos_samples, max_pos_samples, min_neg_samples, max_neg_samples)
        if not counts:
            return
        min_n_pos_id = min(counts, key=lambda c: counts[c][0])
        min_n_neg_id = min(counts, key=lambda c: counts[c][1])
        print(f"Label with least positive samples: ChEBI:{min_n_pos_id} with {counts[min_n_pos_id][0]} samples across all splits")
        print(f"Label with least negative samples: ChEBI:{min_n_neg_id} with {counts[min_n_neg_id][1]} samples across all splits")


    def build_bk(self, target_ids):
        """
        Build ILP background knowledge.

        Args:
            """

        rules, rule_predicates = [], []
        failed_rule_classes: list[str] = []
        if self.predicate_set in ["chebi_fg_rules", "chebi_fg_learned_rules"]:
            prolog_lines_rules, body_predicates_rules = build_background_chebi_fg_rules(CHEBI_FG_RULES_PATH if self.predicate_set == "chebi_fg_rules" else CHEBI_FG_LEARNED_RULES_PATH)
            rules = prolog_lines_rules
            rule_predicates = body_predicates_rules
        
        pbar = tqdm.tqdm(target_ids, desc="Building background knowledge")
        for target_id in pbar:
            # The per-class status goes into the bar itself; printing it would redraw the bar
            # on every iteration. Only warnings and failures are written as their own lines.
            pbar.set_description(f"Building background knowledge for ChEBI:{target_id}")
            pbar.set_postfix_str("")

            # LLM-generated auxiliary predicates are specific to the target class,
            # so they are loaded once per target and merged into the atom-level BK.
            aux_predicates = None
            if self.predicate_set == "llm_generated_fgs":
                aux_predicates = load_auxiliary_predicates(target_id, library_dir=self.aux_library_dir)
                pbar.set_postfix_str(f"{len(aux_predicates)} aux predicate(s)")

            # llm_generated_rules: the class's auxiliary predicates are ASP rules,
            # evaluated (below) against the atom facts plus optional computed facts.
            # Only the derived aux_* extensions are written to bk.pl.
            rule_programs, dependency_programs = None, []
            if self.predicate_set == "llm_generated_rules":
                rule_programs = load_class_rules(target_id, library_dir=self.aux_library_dir)
                # class_map.json records only the predicates the class chose, not the ones
                # they build on, so the dependencies have to be pulled in from the library
                # or the rules ground against an empty body and derive nothing.
                dependency_programs = resolve_rule_dependencies(rule_programs, self.aux_library_dir)
                pbar.set_postfix_str(f"{len(rule_programs)} rule(s)"
                                     + (f" +{len(dependency_programs)} dep(s)" if dependency_programs else ""))

            # The fowl set adds a single class-specific predicate, fowl_<target_id>,
            # derived from a SMARTS pattern, on top of the atom predicates. Not every
            # class has a pattern; those fall back to the plain atom predicates.
            fowl_smarts = None
            if self.predicate_set == "fowl":
                if not hasattr(self, "_fowl_smarts"):
                    self._fowl_smarts = load_fowl_smarts()
                smarts = self._fowl_smarts.get(target_id)
                if smarts is None:
                    pbar.set_postfix_str("no fowl SMARTS, plain atom predicates")
                else:
                    pbar.set_postfix_str(f"fowl SMARTS {smarts}")
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
                # atom is written to the split of the molecule it belongs to. Dependencies
                # take part in the grounding but never reach bk.pl.
                try:
                    extensions = derive_rule_extensions(
                        rule_programs + dependency_programs, eval_facts, all_selected_ids
                    )
                except (RuntimeError, MemoryError) as e:
                    # One class's rules must not end a run that is hours long. The class keeps
                    # its atom-level bk.pl and simply goes without its aux_* extensions.
                    pbar.write(f"  Grounding failed for ChEBI:{target_id} ({e}); "
                               f"continuing without its auxiliary extensions. "
                               f"Rules: {', '.join(rp.name for rp in rule_programs)}")
                    failed_rule_classes.append(target_id)
                    extensions = {}
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

        if failed_rule_classes:
            print(f"\n{len(failed_rule_classes)} class(es) built without their auxiliary rule "
                  f"extensions because grounding failed: {', '.join(failed_rule_classes)}")


    def _take_smallest(self, df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
        """The ``max_samples`` smallest molecules of ``df``, by atom count.

        Ties resolve by the frame's own order, so the pick is deterministic without a seed.
        """
        if len(df) <= max_samples:
            return df
        return df.loc[self._atom_counts[df.index].nsmallest(max_samples, keep="first").index]

    def _direct_neighbors(self, target_id: str) -> tuple[set[str], set[str]]:
        """Molecule ids at or below ``target_id``, and those shared by all its direct parents.

        The second set is the near-miss pool: descendants of every parent that are not
        descendants of the target itself. Like
        ``chebi_utils.sample_filters.get_direct_neighbors``, but reuses ``hierarchy_graph``
        instead of rebuilding the transitive closure once per class, and counts a target
        that is itself a molecule as its own positive. That also keeps it out of the
        negative pool: it is a descendant of each of its parents, so leaving it out of
        ``pos_ids`` would make it a negative of itself (364 of the 1763 v251 labels are
        molecules).
        """
        pos_ids = ({str(d) for d in self.hierarchy_graph.predecessors(target_id)} | {str(target_id)}) & self._mol_index
        parent_spaces = [
            {str(d) for d in self.hierarchy_graph.predecessors(parent)} & self._mol_index
            for parent in self.dataset.chebi_graph.successors(target_id)
        ]
        if not parent_spaces:
            return pos_ids, set()
        return pos_ids, set.intersection(*parent_spaces) - pos_ids

    def build_negatives(self, neg_pool: pd.DataFrame, max_samples: int, random_state: int = 42, prefer_smallest: bool = False) -> pd.DataFrame:
        """At most ``max_samples`` of ``neg_pool``, which holds direct siblings only.

        Every negative is a near-miss, so the objective is separating the target from its
        superclass rather than global classification -- the learned rule is only ever asked
        about molecules a classifier for the parent classes has already admitted.

        With ``prefer_smallest``, an over-full pool keeps the smallest molecules rather than
        a random draw, which shrinks the derived bk.pl. Only the training split sets it;
        validation and test stay random so their scores remain size-unbiased.
        """
        if len(neg_pool) <= max_samples:
            return neg_pool
        return self._take_smallest(neg_pool, max_samples) if prefer_smallest else neg_pool.sample(max_samples, random_state=random_state)

    def gather_samples_for_chebi_cls(self, target_id: str, min_pos_samples=25, max_pos_samples=200, min_neg_samples=25, max_neg_samples=200):
        # Positives are the molecules at or below the target; negatives are only its direct
        # siblings -- the molecules shared by all of its direct parents. Nothing outside the
        # parents' subtrees enters any split, so train, validation and test pose the same
        # near-miss problem and all three assume a classifier for the parent classes.
        # Not every descendant is a molecule, so both pools are intersected with the
        # molecules frame (a SMILES annotation, and the 3-star subset where selected).
        pos_ids, sibling_neg_ids = self._direct_neighbors(target_id)
        df_pos = self.dataset.molecules.loc[sorted(pos_ids)]
        df_neg = self.dataset.molecules.loc[sorted(sibling_neg_ids)]
        if len(df_pos) < min_pos_samples:
            print(f"ChEBI class {target_id} does not have enough positive samples (found {len(df_pos)}, required are at least {min_pos_samples}). Got samples {df_pos.index.tolist()}")
        if len(df_neg) < min_neg_samples:
            print(f"ChEBI class {target_id} does not have enough direct-sibling negatives (found {len(df_neg)}, required are at least {min_neg_samples}). Got samples {df_neg.index.tolist()}")

        split_ids = self._split_ids

        samples_by_split = dict()
        for split in ["train", "validation", "test"]:
            pos_split = df_pos[df_pos.index.astype(str).isin(split_ids[split])]
            neg_split = df_neg[df_neg.index.astype(str).isin(split_ids[split])]
            # Over the cap, training keeps the smallest molecules: they carry the class just
            # as well while keeping bk.pl small enough to ground cheaply. Validation and test
            # keep every positive, so their scores cover the whole held-out class.
            samples_by_split[("pos", split)] = self._take_smallest(pos_split, max_pos_samples) if split == "train" else pos_split
            samples_by_split[("neg", split)] = self.build_negatives(neg_split, max_neg_samples, prefer_smallest=(split == "train"))

        for (posneg, split), df in samples_by_split.items():
            exs_path = get_exs_path(target_id, base_dir=self.problem_dir, split=split)
            with open(exs_path, "w+" if posneg == "pos" else "a") as f:
                for sample in df.index:
                    f.write(f"{posneg}(chebi_{target_id}({sample})).\n")

        # sum up all positive and negative samples across splits
        return sum(len(v) for k, v in samples_by_split.items() if k[0] == "pos"), sum(len(v) for k, v in samples_by_split.items() if k[0] == "neg")



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

    Formats ``chebi_utils.get_numerical_facts`` per molecule as Prolog facts
    (``mol_weight(Mol, W)``, one ``ring_size(Mol, Size)`` per ring). These facts are
    fed to Clingo when a class's auxiliary rules are grounded, but are never written to
    ``bk.pl`` — only the derived ``aux_*`` extensions are persisted.
    """
    lines = []
    for row in rows.itertuples():
        for pred, values in get_numerical_facts(row.mol).items():
            for value in values:
                lines.append(f"{pred}({row.Index},{value}).")
    return lines


def build_full_background(
    rows: pd.DataFrame,
    predicate_set: AVAILABLE_PREDICATE_SETS = "atoms",
    aux_predicates=None,
    aux_timeout: float = DEFAULT_AUX_TIMEOUT,
    aux_failures=None,
    fowl_smarts=None,
    rule_programs=None,
    rule_dependencies=None,
    computed_facts: bool = True,
    aux_library_dir: str | None = None,
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

    ``rule_dependencies`` (``llm_generated_rules``) are the library programs ``rule_programs``
    build on. They are ground alongside but emit no facts of their own. Pass them when the
    caller has already resolved them — resolving here instead costs a full parse of the
    library per call, and needs ``aux_library_dir`` to point at the right one.
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
        if rule_dependencies is None:
            rule_dependencies = resolve_rule_dependencies(rule_programs, aux_library_dir)
        try:
            extensions = derive_rule_extensions(
                rule_programs + rule_dependencies, eval_facts, mol_ids,
            )
        except (RuntimeError, MemoryError) as e:
            print(f"Grounding failed ({e}); returning background knowledge without aux_* facts.")
            extensions = {}
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
