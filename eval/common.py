"""Shared helpers for Role C's evaluation scripts."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trademesh.data import load_corpus  # noqa: E402
from trademesh.query.entities import EntityMatcher  # noqa: E402
from trademesh.rag.llm import LLMClient  # noqa: E402
from trademesh.rag.pipeline import RagConfig, RagPipeline  # noqa: E402
from trademesh.retrievers import available_retrievers, get_reranker  # noqa: E402
from trademesh.text import tokenize  # noqa: E402

NEEDS = ROOT / "data" / "needs"
OUT = ROOT / "eval" / "out"
DRY_OUT = OUT / "dryrun"      # dry runs (fixture / stub) go here, which is git-ignored, never into reported results


def out_dir(dry: bool) -> Path:
    return DRY_OUT if dry else OUT


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(name: str, obj, dry: bool = False) -> Path:
    d = out_dir(dry)
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_pipeline(retriever_name: str | None, allow_stub: bool, use_reranker: bool,
                   config: RagConfig, llm: LLMClient | None = None, needs_llm: bool = True):
    """Build the pipeline over the real corpus. Refuses fixtures and the stub unless allowed,
    so a development run can never produce a number that ends up in the report."""
    passages, companies, is_fixture = load_corpus()
    retrievers, using_stub = available_retrievers(passages)
    if (is_fixture or using_stub) and not allow_stub:
        sys.exit("Refusing to run: the corpus is the test fixture or the retriever is the dev stub. "
                 "Wait for A's data and B's retrievers, or pass --allow-stub for a dry run (results are not reportable).")
    name = retriever_name or next(iter(retrievers))
    if name not in retrievers:
        sys.exit(f"unknown retriever {name!r}; available: {', '.join(retrievers)}")
    llm = llm or LLMClient()
    if needs_llm and llm.is_stub and not allow_stub:
        sys.exit("Refusing to run with LLM_PROVIDER=stub. Set a real provider in .env (or --allow-stub for a dry run).")
    vocab = {t for p in passages for t in tokenize(p.text)}
    matcher = EntityMatcher(companies, known_words=vocab)
    reranker = get_reranker(passages) if use_reranker else None
    pipe = RagPipeline(passages, retrievers[name], llm, matcher, reranker, config)
    return pipe, name, (is_fixture or using_stub or (needs_llm and llm.is_stub))
