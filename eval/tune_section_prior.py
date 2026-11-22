"""Tune the section-prior weight lambda on the DEV split only (proposal 9.6). The test split is never touched here.

    python eval/tune_section_prior.py            # writes eval/out/section_prior_dev.json
For each base retriever, candidates (company filter applied, as in the pipeline) are re-scored with
final = base + lambda * g(Item) and the mean binary-gain nDCG@10 over the dev needs is reported for each lambda.
Ties go to the smaller lambda, so lambda = 0 (no prior) wins unless the prior helps. Every reported test run used
lambda = 0; this script reports what the dev needs say, it does not change any locked result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_common import FILTER_DEPTH, load_needs, mean, require_qrels, setup, table, tickers_of, write_json  # noqa: E402

from trademesh.query.section_prior import apply_prior, ndcg_at_k, tune_lambda  # noqa: E402

GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-stub", action="store_true")
    args = ap.parse_args()
    passages, retrievers, _, dry = setup(args.allow_stub)
    needs = load_needs("dev")
    qrels = require_qrels(needs, dry)
    by_id = {p.id: p for p in passages}
    out = {"n_dev_needs": len(needs), "grid": list(GRID), "metric": "nDCG@10 (binary gain)", "systems": {}}
    rows = []
    for name in ("bm25", "hybrid"):
        if name not in retrievers:
            print(f"skipping {name}: not available (install requirements-ml.txt)")
            continue
        base = retrievers[name]
        by_query = {n["query"]: n for n in needs}

        def search(q, base=base, by_query=by_query):
            tk = tickers_of(by_query[q])
            hits = base.search(q, FILTER_DEPTH)
            return [h for h in hits if by_id[h.passage_id].ticker in tk] if tk else hits

        dev = [(n["query"], {d for d, g in qrels[n["id"]].items() if g > 0}) for n in needs]
        best, scores = tune_lambda(search, by_id, dev, GRID, 10)
        # per-need change at the largest lambda, so the report can say how many needs moved
        moved = {}
        for lam in GRID[1:]:
            w = l = 0
            for q, rel in dev:
                hits = search(q)
                a = ndcg_at_k([h.passage_id for h in hits], rel, 10)
                b = ndcg_at_k([h.passage_id for h in apply_prior(hits, by_id, lam)], rel, 10)
                w += b > a + 1e-9
                l += b < a - 1e-9
            moved[str(lam)] = {"better": w, "worse": l, "same": len(dev) - w - l}
        out["systems"][name] = {"best_lambda": best, "mean_ndcg10": {str(k): v for k, v in scores.items()}, "needs_moved": moved}
        rows.append([name] + [f"{scores[g]:.3f}" for g in GRID] + [best])
    print(f"dev nDCG@10 over {len(needs)} needs" + ("  [DRY RUN]" if dry else ""))
    print(table(["base"] + [f"lam={g}" for g in GRID] + ["best"], rows))
    write_json("section_prior_dev.json", out, dry)


if __name__ == "__main__":
    main()
