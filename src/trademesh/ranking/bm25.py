"""BM25 from scratch over our own inverted index (proposal 9.3, System 2).

    score(q, d) = sum_{t in q} w_t * idf(t) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))
    idf(t)      = ln((N - df + 0.5) / (df + 0.5) + 1)

w_t is 1 for ordinary queries; pseudo-relevance feedback (prf.py) passes other weights.
Postings are NumPy arrays per term, so scoring a query touches only the passages that contain
a query term (the "term-at-a-time" evaluation of the lecture on efficient ranking).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..types import Hit, Passage
from .analyzer import Analyzer, AnalyzerConfig


class BM25Index:
    def __init__(self, passages: list[Passage], analyzer: Analyzer | None = None,
                 k1: float = 1.2, b: float = 0.75, name: str = "bm25"):
        self.name = name
        self.k1, self.b = k1, b
        self.analyzer = analyzer or Analyzer()
        self.ids = [p.id for p in passages]
        self._pos = {pid: i for i, pid in enumerate(self.ids)}
        self.n = len(passages)
        self.doc_tf: list[Counter] = []
        for p in passages:
            self.doc_tf.append(Counter(self.analyzer(self.analyzer.passage_text(p.ticker, p.company, p.text))))
        self.dl = np.array([sum(c.values()) for c in self.doc_tf], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        post: dict[str, tuple[list[int], list[int]]] = {}
        for i, c in enumerate(self.doc_tf):
            for t, f in c.items():
                docs, tfs = post.setdefault(t, ([], []))
                docs.append(i)
                tfs.append(f)
        self.df = {t: len(d) for t, (d, _) in post.items()}
        self.postings = {t: (np.array(d, dtype=np.int32), np.array(f, dtype=np.float64)) for t, (d, f) in post.items()}
        self._len_norm = (1 - self.b) + self.b * self.dl / (self.avgdl or 1.0)

    # --- weights ------------------------------------------------------------------------

    def set_params(self, k1: float, b: float) -> None:
        """Change k1 and b without rebuilding the index (used by the dev grid search)."""
        self.k1, self.b = k1, b
        self._len_norm = (1 - b) + b * self.dl / (self.avgdl or 1.0)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def term_weight(self, term: str, doc_index: int) -> float:
        """BM25 weight of `term` in one passage (idf times saturated, length-normalised tf)."""
        f = self.doc_tf[doc_index].get(term, 0)
        if not f:
            return 0.0
        return self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * self._len_norm[doc_index])

    # --- search -------------------------------------------------------------------------

    def query_terms(self, query: str) -> list[str]:
        return list(dict.fromkeys(self.analyzer(query)))      # unique, order kept: sum over t in q

    def score_vector(self, weights: dict[str, float]) -> np.ndarray:
        scores = np.zeros(self.n)
        for t, w in weights.items():
            entry = self.postings.get(t)
            if entry is None or w == 0:
                continue
            docs, tfs = entry
            scores[docs] += w * self.idf(t) * tfs * (self.k1 + 1) / (tfs + self.k1 * self._len_norm[docs])
        return scores

    def search_weighted(self, weights: dict[str, float], k: int = 30, mask: np.ndarray | None = None) -> list[Hit]:
        scores = self.score_vector(weights)
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        return self._top_k(scores, k)

    def search(self, query: str, k: int = 30) -> list[Hit]:
        return self.search_weighted({t: 1.0 for t in self.query_terms(query)}, k)

    def _top_k(self, scores: np.ndarray, k: int) -> list[Hit]:
        nz = np.flatnonzero(scores > 0)
        if nz.size == 0:
            return []
        if nz.size > k:
            part = nz[np.argpartition(-scores[nz], k - 1)[:k]]
        else:
            part = nz
        order = part[np.lexsort((part, -scores[part]))]      # ties broken by index, so runs are reproducible
        return [Hit(self.ids[i], float(scores[i])) for i in order]

    def explain(self, query: str, passage_id: str) -> list[tuple[str, float]]:
        """Per-term contribution to one passage's score (for the demo and the Viva)."""
        i = self._pos[passage_id]
        rows = [(t, self.term_weight(t, i)) for t in self.query_terms(query)]
        return sorted([r for r in rows if r[1] > 0], key=lambda r: -r[1])
