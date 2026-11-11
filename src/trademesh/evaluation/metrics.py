"""Retrieval metrics, written from the definitions in the evaluation lecture (no ir_measures).

Inputs: `ranked` is a list of passage ids, best first; `rels` is {passage_id: grade} with grades
0 (not relevant), 1 (partly), 2 (answers the need). Passages absent from `rels` are UNJUDGED and
count as not relevant in the headline metrics (the standard pooled-evaluation assumption); bpref
and judged@k, below, quantify how much that assumption matters.

  * Binary metrics (P, R, F1, AP, MRR, success) treat grade >= `min_rel` as relevant (default 1).
  * nDCG uses the lecture's discount; the headline `nDCG@k` is binary-gain as in proposal 10.3,
    `gnDCG@k` is the graded 0/1/2 variant. Gain: DCG_k = sum_{i=1..k} rel_i / log2(i + 1); IDCG is the DCG of the
    judged grades sorted descending.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def _rel_set(rels: Mapping[str, int], min_rel: int) -> set[str]:
    return {d for d, g in rels.items() if g >= min_rel}


def precision_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int, min_rel: int = 1) -> float:
    rel = _rel_set(rels, min_rel)
    return sum(d in rel for d in ranked[:k]) / k


def recall_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int, min_rel: int = 1) -> float:
    rel = _rel_set(rels, min_rel)
    return sum(d in rel for d in ranked[:k]) / len(rel) if rel else 0.0


def f1_at_k(ranked, rels, k, min_rel=1) -> float:
    p, r = precision_at_k(ranked, rels, k, min_rel), recall_at_k(ranked, rels, k, min_rel)
    return 2 * p * r / (p + r) if p + r else 0.0


def success_at_k(ranked, rels, k, min_rel=1) -> float:
    rel = _rel_set(rels, min_rel)
    return float(any(d in rel for d in ranked[:k]))


def average_precision(ranked: Sequence[str], rels: Mapping[str, int], min_rel: int = 1) -> float:
    """Mean of precision at each relevant document retrieved; denominator is ALL judged relevant."""
    rel = _rel_set(rels, min_rel)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def reciprocal_rank(ranked, rels, min_rel=1) -> float:
    rel = _rel_set(rels, min_rel)
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return 1 / i
    return 0.0


def dcg(grades: Sequence[float], k: int | None = None) -> float:
    g = grades[:k] if k else grades
    return sum(x / math.log2(i + 1) for i, x in enumerate(g, start=1))


def ndcg_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int, binary: bool = False) -> float:
    """nDCG@k. binary=True uses gain 1 for grade >= 1 (the proposal's headline, section 10.3);
    binary=False uses the graded 0/1/2 values as gains."""
    grades = {d: (int(g >= 1) if binary else g) for d, g in rels.items()}
    ideal = dcg(sorted(grades.values(), reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg([grades.get(d, 0) for d in ranked[:k]], k) / ideal


def bpref(ranked: Sequence[str], rels: Mapping[str, int], min_rel: int = 1) -> float:
    """Buckley and Voorhees (2004): uses only JUDGED documents, so it is robust to missing judgments.

        bpref = (1/R) sum_{r relevant retrieved} (1 - min(n_above, R) / min(R, N))
    where n_above counts judged non-relevant documents ranked above r, R/N are the numbers of
    judged relevant/non-relevant documents.
    """
    rel = _rel_set(rels, min_rel)
    nonrel = {d for d, g in rels.items() if g < min_rel}
    R, N = len(rel), len(nonrel)
    if R == 0:
        return 0.0
    total, n_above = 0.0, 0
    for d in ranked:
        if d in rel:
            total += 1.0 if N == 0 else 1 - min(n_above, R) / min(R, N)
        elif d in nonrel:
            n_above += 1
    return total / R


def judged_at_k(ranked: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    """Share of the top k that has a judgment. Low values mean the pool did not cover the system."""
    return sum(d in rels for d in ranked[:k]) / k


def pr_points(ranked: Sequence[str], rels: Mapping[str, int], min_rel: int = 1) -> list[tuple[float, float]]:
    """(recall, precision) after each relevant document retrieved."""
    rel = _rel_set(rels, min_rel)
    pts, hits = [], 0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            pts.append((hits / len(rel), hits / i))
    return pts


def interpolated_11pt(ranked, rels, min_rel=1) -> list[float]:
    """Interpolated precision at recall 0.0, 0.1, ..., 1.0 (max precision at any recall >= level)."""
    pts = pr_points(ranked, rels, min_rel)
    return [max([p for r, p in pts if r >= lvl / 10 - 1e-12], default=0.0) for lvl in range(11)]


# --- rank correlation (RQ3, system-ranking agreement) ---------------------------------------

def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return _pearson(_ranks(x), _ranks(y))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    n, conc, disc = len(x), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            conc += s > 0
            disc += s < 0
    return (conc - disc) / (n * (n - 1) / 2) if n > 1 else 0.0


def _ranks(v: Sequence[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks, i = [0.0] * len(v), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1       # average rank for ties
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va and vb else 0.0


METRICS = {
    "P@5": lambda r, q: precision_at_k(r, q, 5),
    "P@10": lambda r, q: precision_at_k(r, q, 10),
    "R@10": lambda r, q: recall_at_k(r, q, 10),
    "F1@10": lambda r, q: f1_at_k(r, q, 10),
    "AP": average_precision,
    "MRR": reciprocal_rank,
    "R@5": lambda r, q: recall_at_k(r, q, 5),
    "F1@5": lambda r, q: f1_at_k(r, q, 5),
    "nDCG@5": lambda r, q: ndcg_at_k(r, q, 5, binary=True),
    "nDCG@10": lambda r, q: ndcg_at_k(r, q, 10, binary=True),
    "gnDCG@10": lambda r, q: ndcg_at_k(r, q, 10),
    "bpref": bpref,
    "S@5": lambda r, q: success_at_k(r, q, 5),
    "judged@10": lambda r, q: judged_at_k(r, q, 10),
}
