"""Shared helpers for Role B's retrieval-evaluation scripts."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from eval.common import NEEDS, OUT, ROOT, out_dir, read_csv, write_json  # noqa: F401  (re-exported)

from trademesh.data import load_corpus  # noqa: E402
from trademesh.evaluation.qrels import load_qrels  # noqa: E402
from trademesh.retrievers import available_retrievers, get_reranker  # noqa: E402

JUDG = Path(os.environ.get("TRADEMESH_JUDGMENTS", ROOT / "data" / "judgments"))   # env override is for tests
FILTER_DEPTH = 300       # same depth Role C's pipeline uses when a ticker filter applies


def load_needs(split: str | None) -> list[dict]:
    rows = read_csv(NEEDS / "needs.csv")
    return [n for n in rows if split in (None, "all") or n["split"] == split]


def setup(allow_stub: bool, with_reranker: bool = False):
    """Return (passages, retrievers, reranker, dry). `dry` is True when nothing here is reportable."""
    passages, _companies, is_fixture = load_corpus()
    retrievers, using_stub = available_retrievers(passages)
    dry = is_fixture or using_stub
    if dry and not allow_stub:
        sys.exit("Refusing to run: the corpus is the test fixture or the retriever is the dev stub, so any number "
                 "would be meaningless. Use the real corpus, or pass --allow-stub for a dry run "
                 "(written to eval/out/dryrun/, never reportable).")
    return passages, retrievers, (get_reranker(passages) if with_reranker else None), dry


def tickers_of(need: dict) -> set[str]:
    return {t for t in need.get("tickers", "").split(";") if t}


def run_system(search, needs: list[dict], by_id: dict, k: int = 30, entity_filter: bool = True,
               drop_tables: bool = False) -> dict[str, list[str]]:
    """{need id: ranked passage ids}. `search(query, k)` returns Hits. The entity filter mirrors Role C's
    pipeline: for a need that names companies, keep only those companies' passages."""
    runs = {}
    for n in needs:
        tk = tickers_of(n) if entity_filter else set()
        depth = FILTER_DEPTH if (tk or drop_tables) else k
        ids = [h.passage_id for h in search(n["query"], depth)]
        if tk:
            ids = [i for i in ids if by_id[i].ticker in tk]
        if drop_tables:
            ids = [i for i in ids if by_id[i].kind != "table"]
        runs[n["id"]] = ids[:k]
    return runs


def require_qrels(needs: list[dict], dry: bool = False) -> dict[str, dict[str, int]]:
    qrels = load_qrels(JUDG)
    missing = [n["id"] for n in needs if not qrels.get(n["id"])]
    if missing and dry:          # dry runs on the tiny fixture have needs with no passages at all
        print(f"[dry run] no judgments for {', '.join(missing)}; scored as empty", file=sys.stderr)
        return {n["id"]: qrels.get(n["id"], {}) for n in needs}
    if missing:
        sys.exit(f"TODO(run): no relevance judgments yet for {', '.join(missing)}. "
                 "Run eval/make_pools.py, have the team judge the sheets, then `eval/make_pools.py --merge`.")
    return qrels


def claim_test_run(script: str, dry: bool, force: bool = False) -> None:
    """The test split is run once, at the freeze. A lock file makes a second run a deliberate act."""
    lock = OUT / f".testrun_{script}"
    if dry:
        return
    if lock.exists() and not force:
        sys.exit(f"{script} was already run on the test split ({lock.name}). The protocol is one run after the "
                 "freeze. If you must repeat it, pass --force and say so in the report.")
    OUT.mkdir(parents=True, exist_ok=True)
    lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def table(headers: list[str], rows: list[list]) -> str:
    cells = [[str(c) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r))
    return "\n".join([line(headers), line(["-" * x for x in w])] + [line(r) for r in cells])


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
