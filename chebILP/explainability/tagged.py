"""Aux unfolding that keeps provenance: every body item carries the path of aux predicates it came from.

Same policy as ``rule_ir.unfold``; the tag of an item is ``()`` for literals written in the learned rule
and ``(p, q, ...)`` for items inlined from aux predicate ``p`` (and, nested inside it, ``q``).
"""

from __future__ import annotations

from dataclasses import dataclass

from chebILP.explainability.rule_ir import (
    Agg,
    AggElem,
    Clause,
    Lit,
    NegGroup,
    UnfoldPolicy,
    _Fresh,
    _rename_apart,
    _unify_head,
    recursive_preds,
    substitute,
)


@dataclass
class TClause:
    head: Lit
    body: list  # [(item, tag)]
    source: int  # index of the learned clause this one was unfolded from

    def clause(self) -> Clause:
        return Clause(self.head, [b for b, _ in self.body])

    @property
    def mol_var(self) -> str:
        return self.head.args[0]


def unfold_tagged(
    rule_clauses: list[Clause], defs: dict, policy: UnfoldPolicy | None = None
):
    """``(tclauses, kept)``: flat clauses with tagged items, and opaque predicates with the reason."""
    policy = policy or UnfoldPolicy()
    rec = recursive_preds(defs)
    kept: dict[str, str] = {}

    def opaque(pred):
        if pred in policy.keep:
            return "kept by policy"
        if pred in rec:
            return "recursive definition"
        return None

    def expand_body(body, fresh, depth=0):
        results = [[]]
        for pair in body:
            opts = expand_item(pair, fresh, depth)
            results = [r + o for r in results for o in opts]
            if len(results) > policy.max_clauses:
                return None
        return results

    def plain_alts(item, fresh, depth):
        """Alternative plain bodies of a defined literal (tags dropped), or None."""
        alts = []
        for d in defs[item.pred]:
            d = _rename_apart(d, fresh)
            u = _unify_head(d.head, Lit(item.pred, item.args))
            if u is None:
                continue
            sub, extra = u
            body = [(substitute(b, sub), ()) for b in d.body] + [(e, ()) for e in extra]
            sub_alts = expand_body(body, fresh, depth + 1)
            if sub_alts is None:
                return None
            alts.extend([[b for b, _ in a] for a in sub_alts])
        return alts

    def expand_item(pair, fresh, depth):
        item, tag = pair
        if depth > 12:
            return [[pair]]
        if isinstance(item, Lit) and item.pred in defs:
            reason = opaque(item.pred)
            if reason:
                kept.setdefault(item.pred, reason)
                return [[pair]]
            child_tag = tag + (item.pred,)
            if not item.neg:
                alts = []
                for d in defs[item.pred]:
                    d = _rename_apart(d, fresh)
                    u = _unify_head(d.head, Lit(item.pred, item.args))
                    if u is None:
                        continue
                    sub, extra = u
                    body = [(substitute(b, sub), child_tag) for b in d.body] + [
                        (e, child_tag) for e in extra
                    ]
                    sub_alts = expand_body(body, fresh, depth + 1)
                    if sub_alts is None:
                        kept.setdefault(item.pred, "too many alternatives")
                        return [[pair]]
                    alts.extend(sub_alts)
                if len(alts) > policy.max_clauses:
                    kept.setdefault(item.pred, "too many alternatives")
                    return [[pair]]
                return alts
            alts = plain_alts(item, fresh, depth)
            if (
                alts is None
                or not policy.inline_negation
                or len(alts) > policy.max_neg_alternatives
            ):
                kept.setdefault(item.pred, "negated, too many alternatives")
                return [[pair]]
            return [[(NegGroup(tuple(tuple(a) for a in alts), item.pred), child_tag)]]
        if isinstance(item, Agg):
            new_elems = []
            for e in item.elems:
                alts = expand_body([(c, ()) for c in e.cond], fresh, depth + 1)
                if alts is None or len(alts) != 1:
                    new_elems.append(e)
                else:
                    new_elems.append(AggElem(e.terms, tuple(b for b, _ in alts[0])))
            return [[(Agg(item.func, tuple(new_elems), item.lguard, item.rguard), tag)]]
        return [[pair]]

    out: list[TClause] = []
    for k, c in enumerate(rule_clauses):
        fresh = _Fresh(c.variables())
        alts = expand_body([(b, ()) for b in c.body], fresh)
        if alts is None or any(len(a) > policy.max_body for a in alts):
            out.append(TClause(c.head, [(b, ()) for b in c.body], k))
            continue
        out.extend(TClause(c.head, a, k) for a in alts)
    return out, kept
