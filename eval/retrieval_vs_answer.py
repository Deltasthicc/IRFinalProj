"""RQ3: does better retrieval give better answers? Spearman correlation per need between retrieval
quality (nDCG@5 of the retriever the RAG used) and the human answer rating (0-2, mean of raters).
Reads human work only: data/judgments/qrels.csv and eval/out/rating_sheet.csv + rating_key.json
(Role C's rag_sheets.py). Reports n; with n near 20 the CI is wide and the report says so.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_common import OUT, load_needs, mean, require_qrels, run_system, setup, write_json  # noqa: E402

from trademesh.evaluation.metrics import ndcg_at_k, spearman  # noqa: E402

CFG_TO_RETRIEVER = {"bm25_rag": "bm25", "best_rag": "hybrid"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--best", default="hybrid", help="retriever behind best_rag")
    ap.add_argument("--allow-stub", action="store_true")
    args = ap.parse_args()
    sheet, keyf = OUT / "rating_sheet.csv", OUT / "rating_key.json"
    if not (sheet.exists() and keyf.exists()):
        sys.exit("TODO(run): eval/out/rating_sheet.csv / rating_key.json missing (Role C's rag_sheets.py, then human ratings).")
    key = json.loads(keyf.read_text())
    ratings = defaultdict(list)
    for r in csv.DictReader(open(sheet, encoding="utf-8", newline="")):
        if r["rating"].strip():
            ratings[(r["question_id"], key[r["question_id"]][r["label"]])].append(int(r["rating"]))
    if not ratings:
        sys.exit("TODO(run): no answer ratings yet.")
    passages, retrievers, _, dry = setup(args.allow_stub)
    needs = load_needs(args.split)
    qrels = require_qrels(needs, dry)
    by_id = {p.id: p for p in passages}
    mapping = dict(CFG_TO_RETRIEVER, best_rag=args.best)
    xs, ys, out = [], [], {}
    for cfg, rname in mapping.items():
        if rname not in retrievers:
            continue
        run = run_system(retrievers[rname].search, needs, by_id, k=10)
        for n in needs:
            if (n["id"], cfg) in ratings:
                xs.append(ndcg_at_k(run[n["id"]], qrels[n["id"]], 5, binary=True))
                ys.append(mean(ratings[(n["id"], cfg)]))
    if len(xs) < 5:
        sys.exit("TODO(run): fewer than 5 (need, configuration) pairs have both judgments and ratings.")
    rho = spearman(xs, ys)
    print(f"Spearman rho = {rho:.3f} over {len(xs)} (need, configuration) pairs" + ("  [DRY RUN]" if dry else ""))
    out = {"spearman": rho, "n": len(xs), "pairs": list(zip(xs, ys))}
    print(f"wrote {write_json('retrieval_vs_answer.json', out, dry)}")


if __name__ == "__main__":
    main()
