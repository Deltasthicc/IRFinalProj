"""Late-interaction page scoring for the gated vision track (Role A, proposal 9.9).

    MaxSim(q, page) = sum_{i=1..n} max_{j=1..m} <q_i, p_j>

q_1..q_n are the query token vectors and p_1..p_m the patch vectors of a page image, as produced by a
ColPali-family model (Faysse et al. 2024). We implement the scoring ourselves; only the vectors come from the
pretrained model (`ColPaliEncoder`, loaded lazily, needs a GPU for indexing and `pip install colpali-engine`).
Brute force over about 300 pages is fast, so no vector database is used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


def maxsim(q: np.ndarray, page: np.ndarray) -> float:
    """q: (n, d) query vectors, page: (m, d) patch vectors."""
    if q.size == 0 or page.size == 0:
        return 0.0
    return float((q @ page.T).max(axis=1).sum())


def rank_pages(q: np.ndarray, pages: Mapping[str, np.ndarray], k: int = 5) -> list[tuple[str, float]]:
    scored = sorted(((pid, maxsim(q, p)) for pid, p in pages.items()), key=lambda x: (-x[1], x[0]))
    return scored[:k]


def write_results(path: str | Path, runs: list[dict]) -> None:
    """runs: [{"query": str, "results": [{"ticker", "page", "score", "image"}]}] -> the app's results.json."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"queries": runs}, indent=1))


class ColPaliEncoder:
    """Thin lazy wrapper; unused by tests. Fill in on the GPU machine (see colpali-engine docs)."""

    def __init__(self, model: str = "vidore/colpali-v1.3"):
        self.model_name = model

    def encode_images(self, images) -> list[np.ndarray]:           # pragma: no cover - needs GPU and model download
        raise NotImplementedError("Run on a GPU machine with colpali-engine installed; see docs/report_A/vision.md")

    def encode_queries(self, queries) -> list[np.ndarray]:         # pragma: no cover
        raise NotImplementedError("Run on a GPU machine with colpali-engine installed; see docs/report_A/vision.md")
