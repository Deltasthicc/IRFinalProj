"""Shared interfaces between Role A (data, index), Role B (ranking) and Role C (query, RAG, app).

If Tanish or Ruchir already have their own version of these, theirs wins and this file
should be replaced by a re-export. Keep field names identical so the JSONL files line up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Passage:
    id: str            # e.g. "AAPL-2025-1A-0007"
    ticker: str
    company: str
    fiscal_year: int
    item: str          # 10-K Item, e.g. "1A", "7"
    kind: str          # "text" | "table"
    text: str
    page: int | None = None


@dataclass
class Hit:
    passage_id: str
    score: float


class Retriever(Protocol):      # A: tfidf, boolean; B: bm25, dense, hybrid
    name: str

    def search(self, query: str, k: int = 30) -> list[Hit]: ...


class Reranker(Protocol):       # B
    def rerank(self, query: str, hits: list[Hit], k: int = 5) -> list[Hit]: ...
