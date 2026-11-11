"""TF-IDF with cosine, heap top-K and champion lists (Role A, proposal 9.2, System 1).

    w(t, d) = (1 + log10 tf(t, d)) * log10(N / df(t))        tf > 0, else 0       (document: "ltc")
    cos(q, d) = sum_t w(t, q) w(t, d) / (||q|| ||d||)        query weighted the same way

Scoring is term-at-a-time over the postings with a score accumulator, and the K best are taken with a heap
(`heapq.nlargest`), as in the lecture on efficient scoring. Document vectors are length-normalised at build
time, so a query only needs the dot products.

Champion lists: for each term keep the r passages with the highest w(t, d). A query is scored only over the
union of its terms' champion lists; if that union has fewer than K passages we fall back to the full postings
(counted in `fallbacks`, so the speed claim cannot hide it).
"""
from __future__ import annotations

import heapq
import math
from collections import Counter

from ..ranking.analyzer import Analyzer
from ..types import Hit, Passage


class TfidfIndex:
    def __init__(self, passages: list[Passage], analyzer: Analyzer | None = None, champions: int | None = None,
                 name: str | None = None):
        self.analyzer = analyzer or Analyzer()
        self.champions = champions
        self.name = name or ("tfidf" if champions is None else f"tfidf-champ-{champions}")
        self.ids = [p.id for p in passages]
        self.n = len(passages)
        tfs = [Counter(self.analyzer(p.text)) for p in passages]
        df: Counter = Counter()
        for c in tfs:
            df.update(c.keys())
        self.df = dict(df)
        self.idf = {t: math.log10(self.n / d) for t, d in df.items()}
        self.postings: dict[str, dict[int, float]] = {t: {} for t in df}        # term -> {doc: normalised weight}
        for d, c in enumerate(tfs):
            raw = {t: (1 + math.log10(f)) * self.idf[t] for t, f in c.items()}
            norm = math.sqrt(sum(w * w for w in raw.values())) or 1.0
            for t, w in raw.items():
                self.postings[t][d] = w / norm
        self.champ: dict[str, list[int]] = {}
        if champions:
            for t, plist in self.postings.items():
                self.champ[t] = [d for d, _ in heapq.nlargest(champions, plist.items(), key=lambda kv: (kv[1], -kv[0]))]
        self.fallbacks = 0
        self.last_candidates = 0

    def _query_weights(self, query: str) -> dict[str, float]:
        c = Counter(t for t in self.analyzer(query) if t in self.idf)
        w = {t: (1 + math.log10(f)) * self.idf[t] for t, f in c.items()}
        norm = math.sqrt(sum(x * x for x in w.values())) or 1.0
        return {t: x / norm for t, x in w.items()}

    def search(self, query: str, k: int = 30) -> list[Hit]:
        qw = self._query_weights(query)
        if not qw:
            return []
        restrict = None
        if self.champions:
            restrict = set()
            for t in qw:
                restrict.update(self.champ[t])
            if len(restrict) < k:
                self.fallbacks += 1
                restrict = None
        scores: dict[int, float] = {}
        for t, wq in qw.items():
            plist = self.postings[t]
            if restrict is None:
                for d, wd in plist.items():
                    scores[d] = scores.get(d, 0.0) + wq * wd
            else:
                for d in restrict:
                    wd = plist.get(d)
                    if wd is not None:
                        scores[d] = scores.get(d, 0.0) + wq * wd
        self.last_candidates = len(scores)
        top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return [Hit(self.ids[d], s) for d, s in top]
