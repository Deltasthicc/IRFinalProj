"""Relevance judgments: file format, consolidation of two judges, agreement.

`data/judgments/qrels.csv` is the single source of truth, one row per (judge, query, passage):

    query_id, passage_id, rel, judge        rel in {0, 1, 2}

For queries judged by two people (the 10 double-judged needs), `consolidate` takes the adjudicated
grade from `data/judgments/adjudicated.csv` (query_id, passage_id, rel) when present, otherwise the
LOWER of the two grades (a conservative rule, stated in the report), and `agreement` returns
Cohen's kappa with linear weights over the doubly-judged pairs. Nothing here invents a judgment:
a missing file gives an empty collection, and the eval scripts refuse to report from it.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

JUDG = Path("data/judgments")


def read_rows(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_raw(path: Path = JUDG / "qrels.csv") -> dict[tuple[str, str], dict[str, int]]:
    """{(query_id, passage_id): {judge: rel}}; blank rel cells (not yet judged) are ignored."""
    raw: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for r in read_rows(path):
        if str(r.get("rel", "")).strip() == "":
            continue
        rel = int(r["rel"])
        if rel not in (0, 1, 2):
            raise ValueError(f"rel must be 0, 1 or 2, got {rel!r} for {r['query_id']}/{r['passage_id']}")
        raw[(r["query_id"], r["passage_id"])][r["judge"]] = rel
    return raw


def consolidate(raw: dict, adjudicated: dict[tuple[str, str], int] | None = None) -> dict[str, dict[str, int]]:
    """{query_id: {passage_id: rel}}"""
    adjudicated = adjudicated or {}
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for (q, d), by_judge in raw.items():
        out[q][d] = adjudicated.get((q, d), min(by_judge.values()))
    return dict(out)


def load_qrels(judg_dir: Path = JUDG) -> dict[str, dict[str, int]]:
    adj = {(r["query_id"], r["passage_id"]): int(r["rel"]) for r in read_rows(judg_dir / "adjudicated.csv")}
    return consolidate(load_raw(judg_dir / "qrels.csv"), adj)


def double_judged(raw: dict) -> list[tuple[int, int]]:
    return [tuple(v[j] for j in sorted(v)[:2]) for v in raw.values() if len(v) >= 2]


def weighted_kappa(pairs: list[tuple[int, int]], levels: int = 3) -> float:
    """Cohen's kappa with linear weights; 1 = perfect agreement, 0 = chance level."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    obs = [[0.0] * levels for _ in range(levels)]
    for a, b in pairs:
        obs[a][b] += 1 / n
    row = [sum(obs[i]) for i in range(levels)]
    col = [sum(obs[i][j] for i in range(levels)) for j in range(levels)]
    w = lambda i, j: abs(i - j) / (levels - 1)
    d_obs = sum(w(i, j) * obs[i][j] for i in range(levels) for j in range(levels))
    d_exp = sum(w(i, j) * row[i] * col[j] for i in range(levels) for j in range(levels))
    return 1.0 if d_exp == 0 else 1 - d_obs / d_exp
