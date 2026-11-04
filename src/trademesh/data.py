"""Loading the shared data files produced by Role A.

Expected: data/passages.jsonl (one Passage per line) and data/companies.json
(a list of {"ticker", "name", "aliases"}). If they are missing we fall back to the
tiny fixture under tests/fixtures/, and say so, so a fixture can never pass as real data.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .types import Passage

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIXTURES = ROOT / "tests" / "fixtures"


def load_passages(path: Path | str) -> list[Passage]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Passage(**json.loads(line)))
    return out


def save_passages(passages: list[Passage], path: Path | str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def load_companies(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_corpus() -> tuple[list[Passage], list[dict], bool]:
    """Return (passages, companies, is_fixture). Prefers A's real data."""
    real_p, real_c = DATA / "passages.jsonl", DATA / "companies.json"
    if real_p.exists() and real_c.exists():
        return load_passages(real_p), load_companies(real_c), False
    return (
        load_passages(FIXTURES / "passages_fixture.jsonl"),
        load_companies(FIXTURES / "companies_fixture.json"),
        True,
    )
