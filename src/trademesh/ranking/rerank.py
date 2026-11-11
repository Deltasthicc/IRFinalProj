"""Cross-encoder reranking of the top candidates (proposal 9.4, System 5).

A cross-encoder reads (query, passage) together, which is more accurate than comparing two
independent embeddings and too slow to run over the whole corpus, so it only reorders the top
`depth` hits from the first stage (Nogueira and Cho, 2019). The scorer is pluggable; production
uses `cross-encoder/ms-marco-MiniLM-L-6-v2`, loaded lazily. Scores are squashed with a sigmoid to
(0, 1) so the abstention threshold in rag/abstain.py can be compared across queries.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from ..types import Hit, Passage


class CrossEncoderScorer:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.name, self._model = model, None

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder      # optional dependency
            self._model = CrossEncoder(self.name)
        return [float(s) for s in self._model.predict(list(pairs), show_progress_bar=False)]


class CrossEncoderReranker:
    def __init__(self, passages: list[Passage], scorer: Callable[[Sequence[tuple[str, str]]], list[float]] | None = None,
                 depth: int = 30, calibrate: bool = True):
        self.by_id = {p.id: p for p in passages}
        self.scorer = scorer or CrossEncoderScorer()
        self.depth, self.calibrate = depth, calibrate

    def rerank(self, query: str, hits: list[Hit], k: int = 5) -> list[Hit]:
        head, tail = hits[: self.depth], hits[self.depth:]
        if not head:
            return []
        raw = self.scorer([(query, self.by_id[h.passage_id].text) for h in head])
        scores = [1 / (1 + math.exp(-s)) for s in raw] if self.calibrate else raw
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
        out = [Hit(head[i].passage_id, scores[i]) for i in order]
        return (out + tail)[:k]
