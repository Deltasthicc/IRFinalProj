"""Rank fusion for the hybrid retriever (proposal 9.4, System 4).

    RRF(d)     = sum_r 1 / (k + rank_r(d)),   k = 60            (Cormack et al., 2009)
    CombSUM(d) = sum_r norm(score_r(d))                         (Fox and Shaw, 1994)
    CombMNZ(d) = CombSUM(d) * number of systems that returned d

RRF uses ranks only, so it needs no score calibration between BM25 and cosine similarity.
CombSUM/CombMNZ with min-max normalisation are kept as comparison points for the fusion ablation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..types import Hit, Retriever


def rrf(rankings: Sequence[Sequence[Hit]], k: int = 60, weights: Sequence[float] | None = None) -> list[Hit]:
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for rank, h in enumerate(ranking, start=1):
            score[h.passage_id] += w / (k + rank)
    return _sorted(score)


def _minmax(hits: Sequence[Hit]) -> dict[str, float]:
    if not hits:
        return {}
    lo, hi = min(h.score for h in hits), max(h.score for h in hits)
    if hi == lo:
        return {h.passage_id: 1.0 for h in hits}
    return {h.passage_id: (h.score - lo) / (hi - lo) for h in hits}


def combsum(rankings: Sequence[Sequence[Hit]], mnz: bool = False) -> list[Hit]:
    score: dict[str, float] = defaultdict(float)
    seen: dict[str, int] = defaultdict(int)
    for ranking in rankings:
        for pid, s in _minmax(ranking).items():
            score[pid] += s
            seen[pid] += 1
    if mnz:
        for pid in score:
            score[pid] *= seen[pid]
    return _sorted(score)


def _sorted(score: dict[str, float]) -> list[Hit]:
    return [Hit(pid, s) for pid, s in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


class HybridRetriever:
    """Runs several retrievers to depth `depth` and fuses their rankings."""

    def __init__(self, retrievers: Sequence[Retriever], name: str = "hybrid", method: str = "rrf",
                 depth: int = 100, rrf_k: int = 60, weights: Sequence[float] | None = None):
        if method not in ("rrf", "combsum", "combmnz"):
            raise ValueError(f"unknown fusion method {method!r}")
        self.retrievers, self.name, self.method = list(retrievers), name, method
        self.depth, self.rrf_k, self.weights = depth, rrf_k, weights

    def search(self, query: str, k: int = 30) -> list[Hit]:
        rankings = [r.search(query, k=self.depth) for r in self.retrievers]
        if self.method == "rrf":
            fused = rrf(rankings, k=self.rrf_k, weights=self.weights)
        else:
            fused = combsum(rankings, mnz=self.method == "combmnz")
        return fused[:k]
