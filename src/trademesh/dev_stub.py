"""DEVELOPMENT STUB ONLY: a tiny in-memory BM25 retriever.

This exists so Role C's code can be run and tested before Ruchir's real BM25 lands.
It is not part of any reported result. Drop it once B's retriever is available
(see trademesh.retrievers.available_retrievers).

score(q, d) = sum_t idf(t) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avgdl))
idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)
"""
from __future__ import annotations

import math
from collections import Counter

from .text import tokenize
from .types import Hit, Passage


class Bm25Stub:
    name = "bm25-stub (dev only)"

    def __init__(self, passages: list[Passage], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = [p.id for p in passages]
        self.tf = [Counter(tokenize(p.text)) for p in passages]
        self.len = [sum(c.values()) for c in self.tf]
        self.avgdl = sum(self.len) / max(1, len(self.len))
        self.df: Counter = Counter()
        for c in self.tf:
            self.df.update(c.keys())
        self.n = len(passages)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, k: int = 30) -> list[Hit]:
        terms = tokenize(query)
        scores = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for t in terms:
                f = c.get(t, 0)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
                    )
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [Hit(self.ids[i], s) for s, i in scores[:k]]
