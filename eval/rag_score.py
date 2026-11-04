"""Compute the RAG study scores from the human-filled sheets. It never invents a rating.

Inputs (eval/out/): rag_runs_<split>.json, rating_sheet.csv (rating filled by humans),
rating_key.json, citation_sheet.csv (supports filled by humans).

Reports per configuration:
  mean rating (0/1/2) and the share rated 2; if two raters filled the sheet (column `rater`),
  Cohen's kappa between them; citation precision (manual: supporting citations / all citations);
  number grounding (automatic: answers whose numbers all occur in cited passages / answers with numbers).
Empty ratings are counted and reported; the script exits with TODO(run) if there are none.

Run:  python eval/rag_score.py --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, out_dir, write_json  # noqa: E402

CONFIGS = ["closed_book", "bm25_rag", "best_rag"]


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Unweighted Cohen's kappa for two raters over the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dry", action="store_true", help="read/write eval/out/dryrun (for dry runs)")
    args = ap.parse_args()
    OUT = out_dir(args.dry)
    runs = json.loads((OUT / f"rag_runs_{args.split}.json").read_text(encoding="utf-8"))
    key = json.loads((OUT / "rating_key.json").read_text(encoding="utf-8"))

    ratings = defaultdict(lambda: defaultdict(dict))          # cfg -> qid -> rater -> rating
    for row in read(OUT / "rating_sheet.csv"):
        if row["rating"].strip() != "":
            cfg = key[row["question_id"]][row["label"]]
            ratings[cfg][row["question_id"]][row["rater"] or "r1"] = int(row["rating"])
    if not any(ratings.values()):
        print("TODO(run): rating_sheet.csv has no ratings yet. Humans must fill the `rating` column.")
        return

    report: dict = {"split": args.split, "configs": {}}
    for cfg in CONFIGS:
        per_q = {q: sum(r.values()) / len(r) for q, r in ratings[cfg].items()}
        vals = list(per_q.values())
        entry = {"n_rated": len(vals),
                 "mean_rating": sum(vals) / len(vals) if vals else None,
                 "share_rated_2": sum(v == 2 for v in vals) / len(vals) if vals else None}
        # citation precision (manual)
        sup = [int(r["supports"]) for r in read(OUT / "citation_sheet.csv")
               if r["supports"].strip() != "" and key[r["question_id"]][r["label"]] == cfg]
        entry["citation_precision"] = sum(sup) / len(sup) if sup else None
        entry["n_citations_judged"] = len(sup)
        # number grounding (automatic, from the verifier)
        with_nums = [r[cfg] for r in runs["runs"] if r[cfg].get("n_numbers")]
        entry["number_grounding"] = (sum(1 for r in with_nums if r["numbers_grounded"]) / len(with_nums)) if with_nums else None
        entry["n_answers_with_numbers"] = len(with_nums)
        entry["verifier_pass_rate"] = (lambda xs: sum(xs) / len(xs) if xs else None)(
            [r[cfg]["verifier_passed"] for r in runs["runs"] if r[cfg]["verifier_passed"] is not None])
        report["configs"][cfg] = entry

    # inter-rater agreement where two raters rated the same answer
    a, b = [], []
    for cfg in CONFIGS:
        for q, byr in ratings[cfg].items():
            if len(byr) >= 2:
                r1, r2 = list(byr.values())[:2]
                a.append(r1)
                b.append(r2)
    report["kappa"] = cohen_kappa(a, b) if a else None
    report["n_double_rated"] = len(a)

    path = write_json(f"rag_scores_{args.split}.json", report, args.dry)
    for cfg, e in report["configs"].items():
        print(cfg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()})
    print("kappa:", report["kappa"], "on", report["n_double_rated"], "double-rated answers")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
