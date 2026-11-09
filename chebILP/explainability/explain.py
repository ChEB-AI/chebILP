"""Explanations of learned ChEBI rules.

- ``explain_rule``: what a rule asks for (a picture of the pattern and one or two sentences).
- ``explain_molecule``: why a rule does or does not accept one molecule (the molecule with the match,
  or the closest match, highlighted, and one or two sentences).

Both depend only on the rule, its aux definitions and the class hierarchy, never on training data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from chebi_utils.read_molecule import parse_molblock, smiles_or_inchi_to_mol
from PIL import Image
from rdkit import Chem

from chebILP.explainability import mol_draw
from chebILP.explainability.global_figure import figure_image
from chebILP.explainability.global_model import figure_model
from chebILP.explainability.local_model import RuleContext, analyse
from chebILP.explainability.molecule import MOL_ID
from chebILP.explainability.pattern_draw import crop_to_content
from chebILP.explainability.style import MAX_W, WHITE, compose, fit, header, legend
from chebILP.explainability.text import global_text, local_text, plural

_AUX_RE = re.compile(r"\b(aux_[A-Za-z0-9_]+)\s*\(")
_SEED_RE = re.compile(r"\b((?:chembl_fg|efga|efg)_[A-Za-z0-9_]+)\s*\(")
_HEAD_RE = re.compile(r"^\s*([a-z][A-Za-z0-9_]*)\s*\(", re.M)


@dataclass
class Explanation:
    text: str
    image: Image.Image
    fires: bool | None = (
        None  # local explanations: whether the rule accepts the molecule
    )


# ── inputs ─────────────────────────────────────────────────────────────────


def read_molecule(molecule: str | Chem.Mol) -> Chem.Mol:
    """A molecule from SMILES, InChI or a molblock, parsed as the ChEBI dataset parses molecules."""
    if isinstance(molecule, Chem.Mol):
        return molecule
    mol = (
        parse_molblock(molecule)
        if "M  END" in molecule
        else smiles_or_inchi_to_mol(molecule.strip())
    )
    if mol is None:
        raise ValueError(f"Could not parse molecule: {molecule!r}")
    return mol


def load_aux_programs(rule: str, library_dir: str | None) -> list[dict]:
    """``{"name", "description", "source"}`` for the aux predicates in ``rule`` and those they build on."""
    names = sorted(set(_AUX_RE.findall(rule)))
    if not names:
        return []
    if library_dir is None:
        raise ValueError(
            f"The rule uses {', '.join(names)}; pass the aux library directory that defines them."
        )
    from chebILP.predicate_generation.auxiliary_rules import (
        aux_rule_path,
        parse_rule_program,
        resolve_rule_dependencies,
    )

    direct = []
    for name in names:
        with open(aux_rule_path(name, library_dir), encoding="utf-8") as f:
            direct.append(parse_rule_program(f.read(), source_file=name))
    deps = resolve_rule_dependencies(direct, library_dir)
    return [
        {"name": p.name, "description": p.description, "source": p.source}
        for p in direct + deps
    ]


def _seed_names(rule: str, aux_programs: list[dict]) -> set[str]:
    return set(
        _SEED_RE.findall(rule + "\n" + "\n".join(p["source"] for p in aux_programs))
    )


def _seed_descriptions(names: set[str]) -> dict:
    """Descriptions of seeded functional-group predicates, from their vocabularies."""
    if not names:
        return {}
    from chebILP.preprocessing.chebi.fg_matching import FG_SEED_SOURCES

    out = {}
    for src in FG_SEED_SOURCES.values():
        want = {n for n in names if n.startswith(src.prefix)}
        if not want:
            continue
        try:
            for name, desc in src.vocabulary():
                full = name if name.startswith(src.prefix) else src.prefix + name
                if full in want:
                    out[full] = desc
        except Exception:
            pass
    return out


def _seed_facts(names: set[str], mol: Chem.Mol) -> list[str]:
    """Facts of seeded functional-group predicates for one molecule."""
    if not names:
        return []
    from chebILP.preprocessing.chebi.fg_matching import (
        FG_SEED_SOURCES,
        fg_atom_fact_lines,
        fg_fact_lines,
    )

    rows = pd.DataFrame([{"mol": mol}], index=[MOL_ID])
    lines = []
    for src in FG_SEED_SOURCES.values():
        want = {n for n in names if n.startswith(src.prefix)}
        if want:
            emit = fg_atom_fact_lines if src.atom_level else fg_fact_lines
            lines += emit(src.matcher(rows), [MOL_ID], needed=want)
    return lines


def class_info(rule: str, chebi_graph=None) -> tuple[str | None, str | None, list[str]]:
    """``(class id, class name, parent names)`` of the rule's head ``chebi_<id>``."""
    m = _HEAD_RE.search(rule)
    head = m.group(1) if m else ""
    cid = head.split("_", 1)[1] if head.startswith("chebi_") else None
    if cid is None or chebi_graph is None or cid not in chebi_graph:
        return cid, None, []
    node = chebi_graph.nodes[cid]
    parents = [
        chebi_graph.nodes[p].get("name") or f"CHEBI:{p}"
        for p in chebi_graph.successors(cid)
    ]
    return cid, node.get("name"), parents


def _rule_context(
    rule: str, aux_library_dir, aux_programs
) -> tuple[RuleContext, set[str]]:
    progs = (
        aux_programs
        if aux_programs is not None
        else load_aux_programs(rule, aux_library_dir)
    )
    seeds = _seed_names(rule, progs)
    rc = RuleContext.build(rule, progs)
    rc.aux_desc.update(
        {k: v for k, v in _seed_descriptions(seeds).items() if k not in rc.aux_desc}
    )
    return rc, seeds


# ── explanations ───────────────────────────────────────────────────────────


def explain_rule(
    rule: str,
    chebi_graph=None,
    aux_library_dir: str | None = None,
    aux_programs: list[dict] | None = None,
) -> Explanation:
    """Picture and short text describing what ``rule`` asks of a molecule.

    ``aux_programs`` (``{"name", "description", "source"}``) overrides loading them from
    ``aux_library_dir``.
    """
    rc, _ = _rule_context(rule, aux_library_dir, aux_programs)
    cid, name, parents = class_info(rule, chebi_graph)
    fm = figure_model(rc.clauses, rc.defs, rc.aux_desc)
    text = global_text(fm, name, parents)
    title = name or (f"CHEBI:{cid}" if cid else "Rule")
    return Explanation(
        text, figure_image(fm, title, f"CHEBI:{cid}" if cid and name else "", text)
    )


def explain_molecule(
    molecule: str | Chem.Mol,
    rule: str,
    chebi_graph=None,
    aux_library_dir: str | None = None,
    aux_programs: list[dict] | None = None,
    name: str | None = None,
) -> Explanation:
    """Picture and short text on whether and why ``rule`` accepts ``molecule``.

    ``molecule`` is a SMILES, InChI, molblock or RDKit molecule; ``name`` is used in the title.
    """
    mol = read_molecule(molecule)
    rc, seeds = _rule_context(rule, aux_library_dir, aux_programs)
    rc.programs = rc.programs + _seed_facts(seeds, mol)
    loc = analyse(mol, rc)
    cid, cname, parents = class_info(rule, chebi_graph)
    text = local_text(loc, cname, parents)
    who = name or "this molecule"
    if cname or cid:
        members = f"the {plural(cname)}" if cname else f"CHEBI:{cid}"
        verdict = (
            f"belongs to {members}" if loc.fires else f"does not belong to {members}"
        )
    else:
        verdict = "satisfies the rule" if loc.fires else "does not satisfy the rule"
    title = f"Why {who} {verdict}"
    title = title[0].upper() + title[1:]
    pic = crop_to_content(
        mol_draw.draw_molecule(mol, loc.atom_cols, loc.bond_cols, loc.notes), WHITE, 16
    )
    width = max(min(MAX_W - 32, pic.width), 560)
    image = compose(
        header(title, f"CHEBI:{cid}" if cid and cname else "", text, width),
        fit(pic, width),
        legend(loc.legend, width),
        width,
    )
    return Explanation(text, image, loc.fires)
