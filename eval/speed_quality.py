"""Speed versus quality (H7, RQ2): latency against MAP and nDCG@10 for each retriever you name.
Built for Role A's champion-list sizes r, but it takes ANY retrievers by name, so it also answers
"what does reranking or PRF cost?".

    python eval/speed_quality.py --systems tfidf,tfidf-champ-50,tfidf-champ-200 --split dev
Latency = median and 95th percentile over the split's queries after one warm-up pass.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_common import claim_test_run, load_needs, mean, require_qrels, run_system, setup, table, write_json  # noqa: E402

from trademesh.evaluation.metrics import average_precision, ndcg_at_k  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="", help="comma-separated retriever names (default: all available)")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--allow-stub", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    passages, retrievers, _, dry = setup(args.allow_stub)
    names = [s for s in args.systems.split(",") if s] or list(retrievers)
    unknown = [s for s in names if s not in retrievers]
    if unknown:
        sys.exit(f"unknown retriever(s) {unknown}; available: {', '.join(retrievers)}")
    needs = load_needs(args.split)
    if args.split == "test":
        claim_test_run("speed_quality", dry, args.force)
    qrels = require_qrels(needs, dry)
    by_id = {p.id: p for p in passages}
    rows, rec = [], {}
    for name in names:
        r = retrievers[name]
        for n in needs:
            r.search(n["query"], 30)
        lat = []
        for n in needs:
            t = time.perf_counter(); r.search(n["query"], 30); lat.append((time.perf_counter() - t) * 1000)
        lat.sort()
        runs = run_system(r.search, needs, by_id, k=30)
        ap_ = mean(average_precision(runs[n["id"]], qrels[n["id"]]) for n in needs)
        nd = mean(ndcg_at_k(runs[n["id"]], qrels[n["id"]], 10, binary=True) for n in needs)
        p50, p95 = statistics.median(lat), lat[max(0, int(0.95 * len(lat)) - 1)]
        rows.append([name, f"{p50:.2f}", f"{p95:.2f}", f"{ap_:.3f}", f"{nd:.3f}"])
        rec[name] = {"p50_ms": p50, "p95_ms": p95, "AP": ap_, "nDCG@10": nd}
    print(table(["system", "p50 ms", "p95 ms", "MAP", "nDCG@10"], rows) + ("\n[DRY RUN]" if dry else ""))
    print(f"wrote {write_json(f'speed_quality_{args.split}.json', rec, dry)}")


if __name__ == "__main__":
    main()
