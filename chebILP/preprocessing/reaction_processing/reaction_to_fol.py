"""
First-order encoding of reactions: the ChEBI molecule encoding per participant, plus links.

A reaction ``r<id>`` links to its participants ``m<id>_<k>`` through ``has_reactant(R,M)`` /
``has_product(R,M)`` (Rhea's left-to-right orientation). Each participant is encoded exactly
like a ChEBI molecule (``has_atom(M,A)``, element/charge/bond/ring facts), so its atoms are
``am<id>_<k>_<idx>``. Ids are reaction-scoped: water in two reactions is two molecules, which
keeps every fact owned by exactly one example.
"""

import pandas as pd

from chebILP.ilp_problem_builder import build_background_chemlog, build_computed_facts

LINK_PREDICATES = [("has_reactant", 2), ("has_product", 2)]


def participant_rows(reactions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Participants of ``reactions`` as a molecule frame (index ``m<id>_<k>``, column ``mol``), and the link facts."""
    records, links = [], []
    for rid, row in reactions.iterrows():
        k = 0
        for side, link in (("reactants", "has_reactant"), ("products", "has_product")):
            for chebi_id, mol in row[side]:
                k += 1
                mid = f"m{rid[1:]}_{k}"
                records.append(
                    {"id": mid, "mol": mol, "chebi": chebi_id, "reaction": rid}
                )
                links.append(f"{link}({rid},{mid}).")
    frame = pd.DataFrame(records, columns=["id", "mol", "chebi", "reaction"]).set_index(
        "id"
    )
    frame.index.name = None
    return frame, links


def build_background_reaction(
    reactions: pd.DataFrame, **kwargs
) -> tuple[list[str], list[tuple[str, int]]]:
    """BK facts and body predicates for ``reactions``; ``kwargs`` go to ``build_background_chemlog``."""
    participants, links = participant_rows(reactions)
    lines, predicates = build_background_chemlog(participants, **kwargs)
    return links + list(lines), LINK_PREDICATES + list(predicates)


def build_computed_facts_reaction(reactions: pd.DataFrame) -> list[str]:
    """``mol_weight`` / ``ring_size`` per participant molecule."""
    return build_computed_facts(participant_rows(reactions)[0])
