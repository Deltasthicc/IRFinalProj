"""Section-type static quality prior (proposal 9.6, Lecture 7 "static quality scores").

    final(q, d) = base(q, d) + lambda * g(section(d)),   g in [0, 1]

`base` is min-max normalised over the candidate list first, so lambda has the same meaning
for BM25, dense or hybrid scores. g is a hand-set prior per 10-K Item (how often that Item
holds the content analysts ask about); lambda is tuned on the 10 DEV queries only.
"""
from __future__ import annotations

import math

from ..types import Hit, Passage

# Hand-set prior per 10-K Item. Rationale: business description, risk factors, MD&A and the
# financial statements answer most questions; legal/mine-safety/exhibit boilerplate rarely does.
G = {
    "1": 0.8, "1A": 1.0, "1B": 0.1, "1C": 0.5, "2": 0.3, "3": 0.4, "4": 0.1, "5": 0.3, "6": 0.2,
    "7": 1.0, "7A": 0.7, "8": 0.8, "9": 0.1, "9A": 0.2, "9B": 0.1, "9C": 0.1,
    "10": 0.1, "11": 0.1, "12": 0.1, "13": 0.1, "14": 0.1, "15": 0.1, "16": 0.1,
}
DEFAULT_G = 0.3


def g(item: str) -> float:
    return G.get(item.upper(), DEFAULT_G)


def normalise(hits: list[Hit]) -> list[Hit]:
    if not hits:
        return []
    lo, hi = min(h.score for h in hits), max(h.score for h in hits)
    if hi == lo:
        return [Hit(h.passage_id, 1.0) for h in hits]
    return [Hit(h.passage_id, (h.score - lo) / (hi - lo)) for h in hits]


def apply_prior(hits: list[Hit], by_id: dict[str, Passage], lam: float) -> list[Hit]:
    """Re-score and re-sort. lam = 0 returns the hits unchanged (same order, same scores)."""
    if lam == 0 or not hits:
        return list(hits)
    out = [Hit(h.passage_id, n.score + lam * g(by_id[h.passage_id].item)) for h, n in zip(hits, normalise(hits))]
    return sorted(out, key=lambda h: -h.score)


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    """Binary-gain nDCG@k (Jarvelin & Kekalainen, 2002)."""
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def tune_lambda(search, by_id: dict[str, Passage], dev: list[tuple[str, set[str]]],
                grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), k: int = 10) -> tuple[float, dict[float, float]]:
    """Grid search lambda on dev queries by mean nDCG@k.

    `search(query)` -> list[Hit]; `dev` = [(query, set of relevant passage ids)] from the
    human judgments. Ties go to the smaller lambda. Returns (best lambda, {lambda: score}).
    """
    cache = [(search(q), rel) for q, rel in dev]
    scores = {}
    for lam in grid:
        vals = [ndcg_at_k([h.passage_id for h in apply_prior(hits, by_id, lam)], rel, k) for hits, rel in cache]
        scores[lam] = sum(vals) / len(vals) if vals else 0.0
    best = max(grid, key=lambda lam: (scores[lam], -lam))
    return best, scores
