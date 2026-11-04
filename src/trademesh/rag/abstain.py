"""Abstention rule (proposal 9.8): abstain on INSUFFICIENT EVIDENCE, or when the best
score is below tau. tau is tuned on the DEV queries only (see eval/abstention_eval.py).

Until B's cross-encoder exists the score is `confidence` below: the best retrieval score
divided by the number of content words in the query (so long queries are not favoured).
When the reranker lands, pass its best score instead; the rule itself does not change.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..text import content_terms
from ..types import Hit
from .prompt import is_insufficient


def confidence(hits: list[Hit], query: str) -> float:
    if not hits:
        return 0.0
    return hits[0].score / max(1, len(set(content_terms(query))))


@dataclass
class Decision:
    abstain: bool
    reason: str = ""


def decide(answer: str | None, conf: float, tau: float) -> Decision:
    if conf < tau:
        return Decision(True, f"best score {conf:.3f} is below the threshold tau = {tau:.3f}")
    if answer is not None and is_insufficient(answer):
        return Decision(True, "the model answered INSUFFICIENT EVIDENCE")
    return Decision(False)


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int


def abstention_metrics(pairs: list[tuple[bool, bool]]) -> Metrics:
    """pairs = (should_abstain, did_abstain). Positive class = abstain."""
    tp = sum(1 for s, d in pairs if s and d)
    fp = sum(1 for s, d in pairs if not s and d)
    fn = sum(1 for s, d in pairs if s and not d)
    tn = sum(1 for s, d in pairs if not s and not d)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / len(pairs) if pairs else 0.0
    return Metrics(p, r, f1, acc, tp, fp, fn, tn)


def tune_tau(items: list[tuple[float, bool, bool]]) -> tuple[float, Metrics]:
    """Pick tau on dev data. items = (confidence, should_abstain, model_said_insufficient).

    Candidate taus are 0 and the midpoints between consecutive observed confidences.
    We maximise F1 and break ties towards the smaller tau (abstain less).
    """
    confs = sorted({c for c, _, _ in items})
    cands = [0.0] + [(a + b) / 2 for a, b in zip(confs, confs[1:])] + ([confs[-1] + 1e-6] if confs else [])
    best = None
    for tau in cands:
        m = abstention_metrics([(s, c < tau or llm) for c, s, llm in items])
        key = (m.f1, m.accuracy, -tau)
        if best is None or key > best[0]:
            best = (key, tau, m)
    return best[1], best[2]
