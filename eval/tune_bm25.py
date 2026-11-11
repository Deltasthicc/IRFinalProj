"""Tune BM25's k1 and b on the DEV split only (proposal 9.3). The test split is never touched here.

    python eval/tune_bm25.py            # grid search, writes data/judgments/bm25_params.json
Objective: mean nDCG@10 over the dev needs (ties broken toward the textbook 1.2 / 0.75). The grid
is printed in full so the report can show the whole surface and not only its maximum.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_common import JUDG, load_needs, mean, require_qrels, run_system, setup, table, write_json  # noqa: E402

from trademesh.evaluation.metrics import ndcg_at_k  # noqa: E402
from trademesh.ranking import BM25Index  # noqa: E402

K1 = [0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
B = [0.3, 0.5, 0.75, 0.9, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-stub", action="store_true")
    args = ap.parse_args()
    passages, _, _, dry = setup(args.allow_stub)
    needs = load_needs("dev")                                   # dev only, by construction
    qrels = require_qrels(needs, dry)
    by_id = {p.id: p for p in passages}
    ix = BM25Index(passages)
    grid = {}
    for k1, b in itertools.product(K1, B):
        ix.set_params(k1, b)
        runs = run_system(ix.search, needs, by_id, k=10)
        grid[(k1, b)] = mean(ndcg_at_k(runs[n["id"]], qrels[n["id"]], 10, binary=True) for n in needs)
    best = max(grid, key=lambda kb: (round(grid[kb], 6), -abs(kb[0] - 1.2) - abs(kb[1] - 0.75)))
    print(f"dev nDCG@10 over {len(needs)} needs" + ("  [DRY RUN]" if dry else ""))
    print(table(["k1 \\ b"] + [str(b) for b in B], [[k1] + [f"{grid[(k1, b)]:.3f}" for b in B] for k1 in K1]))
    print(f"\nbest: k1={best[0]}, b={best[1]}  (textbook 1.2/0.75 gives {grid[(1.2, 0.75)]:.3f}, best {grid[best]:.3f})")
    write_json("bm25_grid_dev.json", {f"{k1}|{b}": v for (k1, b), v in grid.items()}, dry)
    if not dry:
        (JUDG / "bm25_params.json").write_text(json.dumps({"k1": best[0], "b": best[1], "tuned_on": "dev", "objective": "nDCG@10"}, indent=1))
        print("wrote data/judgments/bm25_params.json (picked up automatically by trademesh.ranking)")


if __name__ == "__main__":
    main()
