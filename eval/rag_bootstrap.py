"""Paired bootstrap confidence intervals for the RAG study (answer rating, 0/1/2).

Reads eval/out/rag_runs_<split>.json, rating_sheet.csv and rating_key.json (the key is read here
only to join labels to systems; raters never see it). The per-question rating is the mean over raters.
Resamples QUESTIONS with replacement (10,000 times, seed 358) and reports, for each pair of
configurations, the mean difference and its 95% percentile interval. Also reports mean rating by
question type. With only ~20 test questions the intervals are wide; say so in the report.

Run:  python eval/rag_bootstrap.py --split test
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, out_dir, read_csv, write_json  # noqa: E402

CONFIGS = ["closed_book", "bm25_rag", "best_rag"]
PAIRS = [("bm25_rag", "closed_book"), ("best_rag", "closed_book"), ("best_rag", "bm25_rag")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()
    OUT = out_dir(args.dry)
    runs = json.loads((OUT / f"rag_runs_{args.split}.json").read_text(encoding="utf-8"))
    key = json.loads((OUT / "rating_key.json").read_text(encoding="utf-8"))
    qtype = {r["id"]: r["type"] for r in runs["runs"]}

    raw = defaultdict(lambda: defaultdict(list))               # cfg -> qid -> [ratings of all raters]
    for row in read_csv(OUT / "rating_sheet.csv"):
        if row["rating"].strip() != "":
            raw[key[row["question_id"]][row["label"]]][row["question_id"]].append(int(row["rating"]))
    per = {c: {q: sum(v) / len(v) for q, v in raw[c].items()} for c in CONFIGS}
    qids = sorted(per[CONFIGS[0]])
    assert all(sorted(per[c]) == qids for c in CONFIGS), "every configuration must have a rating for every question"

    rng = random.Random(358)
    report: dict = {"split": args.split, "n_questions": len(qids), "resamples": args.resamples, "pairs": {}, "by_type": {}}
    for a, b in PAIRS:
        d = [per[a][q] - per[b][q] for q in qids]
        means = []
        for _ in range(args.resamples):
            s = [d[rng.randrange(len(d))] for _ in d]
            means.append(sum(s) / len(s))
        means.sort()
        lo, hi = means[int(0.025 * args.resamples)], means[int(0.975 * args.resamples) - 1]
        report["pairs"][f"{a} - {b}"] = {"mean_diff": sum(d) / len(d), "ci95": [lo, hi],
                                          "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d),
                                          "tied": sum(x == 0 for x in d)}
    for t in sorted(set(qtype.values())):
        ids = [q for q in qids if qtype[q] == t]
        report["by_type"][t] = {"n": len(ids), **{c: sum(per[c][q] for q in ids) / len(ids) for c in CONFIGS}}
    path = write_json(f"rag_bootstrap_{args.split}.json", report, args.dry)
    for k, v in report["pairs"].items():
        print(f"{k}: diff {v['mean_diff']:+.3f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  "
              f"better {v['better']} / worse {v['worse']} / tied {v['tied']}")
    for t, v in report["by_type"].items():
        print(f"{t:12} n={v['n']:2d}  " + "  ".join(f"{c} {v[c]:.2f}" for c in CONFIGS))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
