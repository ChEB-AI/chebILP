"""Hybrid (BM25 + dense) retrieval over the shared auxiliary-predicate library.

Given a ChEBI class's ``name + definition``, rank the predicates already in the library so
the generator can offer the most relevant ones to the LLM for reuse instead of
regenerating near-duplicates (EXP-014). Two channels are fused with Reciprocal Rank
Fusion, matching the ``scripts/*_retrieval_demo.py`` experiments:

- BM25 (lexical) over ``"{name} {description}"`` documents.
- Dense cosine similarity over ``"{name}: {description}"`` sentence embeddings
  (``sentence-transformers``, all-MiniLM-L6-v2, L2-normalised so dot product = cosine).

Both libraries are heavy and only needed here, so they are imported lazily. If
``sentence-transformers`` is unavailable the retriever degrades to BM25-only.
"""

from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "its", "of", "on", "one", "or", "that", "the", "this", "to",
    "which", "with", "group", "groups", "atom", "atoms", "molecule",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop very short tokens and stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


class _IncrementalBM25:
    """Okapi BM25 that supports O(len(document)) incremental document addition.

    ``rank_bm25.BM25Okapi`` fixes the corpus (document frequencies, IDF, average length)
    in ``__init__``, so growing the corpus means rebuilding from scratch. Here the corpus
    statistics are kept as documents arrive and IDF is computed per query term at query
    time, so adding a document never invalidates a precomputed index. Scoring cost is the
    same O(N * |query|) as BM25Okapi. Uses the standard k1/b Okapi formula with the
    Lucene-style non-negative IDF ``log(1 + (N - df + 0.5) / (df + 0.5))`` (so very common
    terms simply contribute ~0 rather than needing rank_bm25's negative-IDF flooring).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[dict[str, int]] = []  # per-document term -> count
        self.doc_len: list[int] = []
        self.df: dict[str, int] = {}               # term -> number of documents containing it
        self._total_len = 0

    @property
    def corpus_size(self) -> int:
        return len(self.doc_len)

    @property
    def avgdl(self) -> float:
        return self._total_len / self.corpus_size if self.corpus_size else 0.0

    def add(self, tokens: list[str]) -> None:
        freqs: dict[str, int] = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        self.doc_freqs.append(freqs)
        self.doc_len.append(len(tokens))
        self._total_len += len(tokens)
        for t in freqs:  # each distinct term bumps its document frequency once
            self.df[t] = self.df.get(t, 0) + 1

    def get_scores(self, query_tokens: list[str]):
        import numpy as np

        n = self.corpus_size
        scores = np.zeros(n)
        if n == 0:
            return scores
        doc_len = np.array(self.doc_len, dtype=float)
        denom_len = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
        for q in set(query_tokens):
            df = self.df.get(q, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            q_freq = np.fromiter((d.get(q, 0) for d in self.doc_freqs), dtype=float, count=n)
            scores += idf * (q_freq * (self.k1 + 1) / (q_freq + denom_len))
        return scores


class HybridPredicateRetriever:
    def __init__(self, entries, model_name="all-MiniLM-L6-v2", rrf_k=60, use_dense=True):
        """``entries``: list of dicts with at least ``name`` and ``description`` keys.

        The initial ``entries`` are indexed in bulk (batch dense encoding); predicates
        added later with ``add_entry`` are folded in incrementally (see that method).
        """
        self.entries = list(entries)
        self.rrf_k = rrf_k
        self._bm25 = _IncrementalBM25()
        self._dense_model = None
        self._dense_embeddings = None   # materialised (n, dim) matrix aligned with entries
        self._dense_pending: list = []  # embedding rows added since the last retrieve()

        for e in self.entries:
            self._bm25.add(tokenize(self._doc_text_lexical(e)))
        if use_dense:
            self._load_dense_model(model_name)  # load the model even if the library is empty
            if self._dense_model is not None and self.entries:
                self._dense_embeddings = self._dense_model.encode(
                    [self._doc_text_dense(e) for e in self.entries], normalize_embeddings=True
                )

    @classmethod
    def from_library(cls, base_dir: str, **kwargs) -> "HybridPredicateRetriever":
        import os

        from chebILP.auxiliary_predicates import load_library_predicates

        entries = [
            {
                "name": p.name,
                "description": p.description,
                "kind": p.kind,
                # Library file stem (without .py); what class_map.json records for reuse.
                "stem": os.path.splitext(os.path.basename(p.source_file))[0],
            }
            for p in load_library_predicates(base_dir)
        ]
        return cls(entries, **kwargs)

    @classmethod
    def from_rule_library(cls, base_dir: str, **kwargs) -> "HybridPredicateRetriever":
        """Build a retriever over an ASP rule library (``.pl`` programs)."""
        import os

        from chebILP.auxiliary_rules import load_library_rules

        entries = [
            {
                "name": p.name,
                "description": p.description,
                "kind": "rule",
                # Library file stem (without .pl); what class_map.json records for reuse.
                "stem": os.path.splitext(os.path.basename(p.source_file))[0],
            }
            for p in load_library_rules(base_dir)
        ]
        return cls(entries, **kwargs)

    def __len__(self) -> int:
        return len(self.entries)

    def _doc_text_lexical(self, e) -> str:
        return f"{e['name'].replace('_', ' ')} {e.get('description', '')}"

    def _doc_text_dense(self, e) -> str:
        return f"{e['name'].replace('_', ' ')}: {e.get('description', '')}"

    def _load_dense_model(self, model_name):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; using BM25-only retrieval. "
                "Install the 'llm' extra for the dense channel."
            )
            return
        self._dense_model = SentenceTransformer(model_name)

    def add_entry(self, entry: dict):
        """Add one predicate incrementally — no full re-index of the corpus.

        BM25 corpus statistics are updated in O(len(document)). The dense embedding is
        computed once (a single forward pass for the new document) and buffered; buffered
        rows are concatenated onto the matrix in one shot on the next ``retrieve`` rather
        than copying the whole matrix on every add.
        """
        self.entries.append(entry)
        self._bm25.add(tokenize(self._doc_text_lexical(entry)))
        if self._dense_model is not None:
            self._dense_pending.append(
                self._dense_model.encode(self._doc_text_dense(entry), normalize_embeddings=True)
            )

    def _flush_dense(self):
        """Concatenate any buffered embedding rows onto the materialised matrix."""
        if not self._dense_pending:
            return
        import numpy as np

        new = np.asarray(self._dense_pending)
        self._dense_embeddings = (
            new if self._dense_embeddings is None else np.vstack([self._dense_embeddings, new])
        )
        self._dense_pending = []

    @staticmethod
    def _rank_positions(scores):
        import numpy as np

        order = np.argsort(scores)[::-1]
        positions = np.empty(len(scores), dtype=int)
        positions[order] = np.arange(1, len(scores) + 1)
        return positions

    def retrieve(self, query_text: str, top_k: int = 25) -> list[dict]:
        """Return the top-k library entries for ``query_text``, best first.

        Each returned dict is a copy of the library entry plus ``rrf`` and the
        per-channel ranks ``bm25_rank`` / ``dense_rank`` (``None`` if dense is off).
        """
        if not self.entries:
            return []

        import numpy as np

        rank_arrays = []
        bm25_ranks = self._rank_positions(self._bm25.get_scores(tokenize(query_text)))
        rank_arrays.append(bm25_ranks)

        dense_ranks = None
        if self._dense_model is not None:
            self._flush_dense()  # fold in any predicates added since the last retrieve
            q = self._dense_model.encode(query_text, normalize_embeddings=True)
            dense_ranks = self._rank_positions(np.dot(self._dense_embeddings, q))
            rank_arrays.append(dense_ranks)

        fused = np.zeros(len(self.entries), dtype=float)
        for positions in rank_arrays:
            fused += 1.0 / (self.rrf_k + positions)

        top_idx = np.argsort(fused)[::-1][:top_k]
        results = []
        for i in top_idx:
            e = dict(self.entries[i])
            e["rrf"] = float(fused[i])
            e["bm25_rank"] = int(bm25_ranks[i])
            e["dense_rank"] = int(dense_ranks[i]) if dense_ranks is not None else None
            results.append(e)
        return results
