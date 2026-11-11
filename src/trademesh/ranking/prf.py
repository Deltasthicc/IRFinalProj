"""Pseudo-relevance feedback, Rocchio style, on top of BM25 (proposal 9.5; Rocchio 1971).

    q_new = alpha * q0 + beta * (1/k) * sum_{d in top-k} d_hat

q0 is the unit-length query vector; d_hat is the L2-normalised BM25 term-weight vector of a feedback
passage. We keep the m strongest expansion terms (terms not in the query) and use q_new's
components as the term weights w_t of the BM25 sum, so an expansion term counts for less than an
original term. Defaults: k = 5, alpha = 1, beta = 0.5, m = 10 (proposal).
"""
from __future__ import annotations

import math
from collections import defaultdict

from ..types import Hit
from .bm25 import BM25Index


class PRFRetriever:
    def __init__(self, index: BM25Index, k_docs: int = 5, alpha: float = 1.0, beta: float = 0.5,
                 m_terms: int = 10, name: str = "bm25+prf"):
        self.index, self.k_docs, self.alpha, self.beta, self.m_terms, self.name = index, k_docs, alpha, beta, m_terms, name

    def expand(self, query: str) -> tuple[dict[str, float], list[str]]:
        """Return (term weights of the expanded query, the expansion terms added)."""
        ix = self.index
        q_terms = ix.query_terms(query)
        if not q_terms:
            return {}, []
        first = ix.search(query, k=self.k_docs)
        if not first:
            return {t: 1.0 for t in q_terms}, []
        centroid: dict[str, float] = defaultdict(float)
        for h in first:
            i = ix._pos[h.passage_id]
            vec = {t: ix.term_weight(t, i) for t in ix.doc_tf[i]}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for t, v in vec.items():
                centroid[t] += v / norm / len(first)
        expansion = sorted(((t, v) for t, v in centroid.items() if t not in set(q_terms)), key=lambda tv: (-tv[1], tv[0]))
        expansion = expansion[: self.m_terms]
        q0 = 1 / math.sqrt(len(q_terms))
        weights = {t: self.alpha * q0 + self.beta * centroid.get(t, 0.0) for t in q_terms}
        for t, v in expansion:
            weights[t] = self.beta * v
        # Rescale so that the original query terms keep weight about 1 (BM25 scores stay comparable).
        scale = 1 / q0
        return {t: w * scale for t, w in weights.items()}, [t for t, _ in expansion]

    def search(self, query: str, k: int = 30) -> list:
        weights, _ = self.expand(query)
        return self.index.search_weighted(weights, k) if weights else []
