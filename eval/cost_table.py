"""Cost of each system (proposal 9.3-9.5): build time, memory/disk of the index, and query latency.
Pure measurement, no judgments needed. Latency is the median and 95th percentile over the 30 need queries,
after one warm-up pass.
    python eval/cost_table.py [--allow-stub]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_common import load_needs, setup, table, write_json  # noqa: E402

from trademesh.ranking import BM25Index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-stub", action="store_true")
    args = ap.parse_args()
    passages, retrievers, reranker, dry = setup(args.allow_stub, with_reranker=True)
    queries = [n["query"] for n in load_needs("all")]
    t0 = time.perf_counter(); BM25Index(passages); build = time.perf_counter() - t0
    rows, rec = [], {"bm25_build_s": build, "n_passages": len(passages)}
    systems = dict(retrievers)
    for name, r in systems.items():
        for q in queries:
            r.search(q, 30)                                  # warm-up
        lat = []
        for q in queries:
            t = time.perf_counter(); r.search(q, 30); lat.append((time.perf_counter() - t) * 1000)
        lat.sort()
        rows.append([name, f"{statistics.median(lat):.1f}", f"{lat[int(0.95 * len(lat)) - 1]:.1f}"])
        rec[name] = {"p50_ms": statistics.median(lat), "p95_ms": lat[int(0.95 * len(lat)) - 1]}
    if reranker is not None and "hybrid" in retrievers:
        lat = []
        for q in queries:
            hits = retrievers["hybrid"].search(q, 30)
            t = time.perf_counter(); reranker.rerank(q, hits, 5); lat.append((time.perf_counter() - t) * 1000)
        lat.sort()
        rows.append(["rerank(top30) only", f"{statistics.median(lat):.1f}", f"{lat[int(0.95 * len(lat)) - 1]:.1f}"])
    print(f"{len(passages)} passages; BM25 index build {build:.2f}s" + ("   [DRY RUN]" if dry else ""))
    print(table(["system", "p50 ms", "p95 ms"], rows))
    print(f"wrote {write_json('cost_table.json', rec, dry)}")


if __name__ == "__main__":
    main()
