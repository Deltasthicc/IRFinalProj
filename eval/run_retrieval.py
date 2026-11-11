"""RQ1/RQ2: compare retrieval systems on the pooled test collection. Never invents a number.

    python eval/run_retrieval.py --split dev            # while building and tuning
    python eval/run_retrieval.py --split test           # ONCE, after the freeze (locked afterwards)

Per system: P@5, P@10, R@10, F1@10, MAP, MRR, nDCG@5, nDCG@10, bpref, S@5 and judged@10, overall and
by need type (entity / topical / terminology); the 11-point interpolated PR curve; a paired
bootstrap against the baseline (default bm25) with 95% CI and win/loss/tie. Needs judgments:
without data/judgments/qrels.csv it stops with TODO(run).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_common import (claim_test_run, load_needs, mean, require_qrels, run_system, setup, table,  # noqa: E402
                                   write_json)

from trademesh.evaluation.metrics import METRICS, interpolated_11pt  # noqa: E402
from trademesh.evaluation.stats import paired_bootstrap  # noqa: E402

SHOW = ["P@5", "P@10", "AP", "MRR", "nDCG@10", "bpref", "judged@10"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--baseline", default="bm25")
    ap.add_argument("--no-entity-filter", action="store_true")
    ap.add_argument("--allow-stub", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    passages, retrievers, reranker, dry = setup(args.allow_stub, with_reranker=True)
    needs = load_needs(args.split)
    if args.split == "test":
        claim_test_run("run_retrieval", dry, args.force)
    qrels = require_qrels(needs, dry)
    by_id = {p.id: p for p in passages}

    systems = {name: r.search for name, r in retrievers.items()}
    if reranker is not None:
        for base in ("bm25", "hybrid"):
            if base in retrievers:
                systems[f"{base}+rerank"] = (lambda b: lambda q, k: reranker.rerank(q, retrievers[b].search(q, 30), k))(base)
    runs = {s: run_system(fn, needs, by_id, k=30, entity_filter=not args.no_entity_filter) for s, fn in systems.items()}

    per_query = {s: {n["id"]: {m: f(runs[s][n["id"]], qrels[n["id"]]) for m, f in METRICS.items()} for n in needs} for s in runs}
    overall = {s: {m: mean(per_query[s][n["id"]][m] for n in needs) for m in METRICS} for s in runs}
    by_type = {s: {t: {m: mean(per_query[s][n["id"]][m] for n in needs if n["type"] == t) for m in ("AP", "nDCG@10", "P@5")}
                   for t in ("entity", "topical", "terminology")} for s in runs}
    curve = {s: [mean(x) for x in zip(*(interpolated_11pt(runs[s][n["id"]], qrels[n["id"]]) for n in needs))] for s in runs}
    base = args.baseline if args.baseline in runs else next(iter(runs))
    sig = {s: {m: paired_bootstrap([per_query[s][n["id"]][m] for n in needs], [per_query[base][n["id"]][m] for n in needs])
               for m in ("AP", "nDCG@10")} for s in runs if s != base}

    print(f"split={args.split}  needs={len(needs)}  baseline={base}" + ("   [DRY RUN, not reportable]" if dry else ""))
    print(table(["system"] + SHOW, [[s] + [f"{overall[s][m]:.3f}" for m in SHOW] for s in runs]))
    print("\nby need type (nDCG@10):")
    print(table(["system", "entity", "topical", "terminology"],
                [[s] + [f"{by_type[s][t]['nDCG@10']:.3f}" for t in ("entity", "topical", "terminology")] for s in runs]))
    print(f"\npaired bootstrap vs {base} (95% CI of mean difference, wins/losses/ties):")
    print(table(["system", "metric", "diff", "CI", "W/L/T", "p"],
                [[s, m, f"{r['mean_diff']:+.3f}", f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                  f"{r['wins']}/{r['losses']}/{r['ties']}", f"{r['p']:.3f}"] for s, d in sig.items() for m, r in d.items()]))
    low = [s for s in runs if overall[s]["judged@10"] < 0.8]
    if low:
        print(f"\nNOTE: judged@10 < 0.8 for {', '.join(low)}: part of their top 10 is unjudged, so their scores are "
              "pessimistic. Cite bpref alongside, and see eval/pool_bias.py.")
    path = write_json(f"retrieval_{args.split}.json", {"split": args.split, "baseline": base, "n_needs": len(needs),
                      "overall": overall, "by_type": by_type, "pr_11pt": curve, "per_query": per_query, "bootstrap": sig,
                      "runs_top10": {s: {q: ids[:10] for q, ids in r.items()} for s, r in runs.items()}}, dry)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
