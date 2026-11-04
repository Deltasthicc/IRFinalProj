"""Prepare the HUMAN rating sheets from eval/out/rag_runs_<split>.json. This script fills in no ratings.

Outputs (eval/out/):
  rating_sheet.csv      one row per (question, anonymous label A/B/C): the answer text and an empty
                        `rating` column to be filled 0/1/2 by two raters. System names are hidden and
                        the label order is shuffled separately for every question (presentation bias).
  rating_key.json       label -> configuration for each question. RATERS MUST NOT OPEN THIS FILE.
  citation_sheet.csv    one row per (answer, cited passage) for the manual citation-precision check,
                        with an empty `supports` column (1 if the passage supports the claim, else 0).
                        It does not show which configuration produced the answer.
  reference_drafts.csv  DRAFT reference answers for the humans to check or replace (first sentence of
                        the top passage of best_rag). Marked DRAFT; never used as truth unedited.

Rating scale: 2 = correct and complete, 1 = partly correct or incomplete, 0 = wrong, invented or no answer.
An abstention ("INSUFFICIENT EVIDENCE") on an answerable question is rated 0 unless the reference says
the corpus has no answer.

Run:  python eval/rag_sheets.py --split test --seed 358
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import ROOT, out_dir, write_json  # noqa: E402

from trademesh.data import load_corpus  # noqa: E402

CONFIGS = ["closed_book", "bm25_rag", "best_rag"]
LABELS = ["A", "B", "C"]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dry", action="store_true", help="read/write eval/out/dryrun (for dry runs)")
    ap.add_argument("--seed", type=int, default=358)
    args = ap.parse_args()
    OUT = out_dir(args.dry)
    runs = json.loads((OUT / f"rag_runs_{args.split}.json").read_text(encoding="utf-8"))
    if not runs.get("reportable", False):
        print("WARNING: the input run is marked NOT reportable (dry run). Sheets are for testing only.")
    passages = {p.id: p for p in load_corpus()[0]}
    rng = random.Random(args.seed)

    rating_rows, cite_rows, key, ref_rows = [], [], {}, []
    for r in runs["runs"]:
        order = CONFIGS[:]
        rng.shuffle(order)
        key[r["id"]] = dict(zip(LABELS, order))
        for label, cfg in zip(LABELS, order):
            rating_rows.append([r["id"], r["question"], label, r[cfg]["answer"], "", ""])
            for i in (r[cfg]["cited"] or []):
                ids = r[cfg]["passage_ids"]
                if 1 <= i <= len(ids):
                    cite_rows.append([r["id"], label, f"[P{i}]", r[cfg]["answer"], passages[ids[i - 1]].text, ""])
        top = r["best_rag"]["passage_ids"][:1]
        draft = passages[top[0]].text.split(". ")[0] if top and top[0] in passages else ""
        ref_rows.append([r["id"], r["question"], "DRAFT: " + draft, "", ""])

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "rating_sheet.csv", ["question_id", "question", "label", "answer", "rating", "rater"], rating_rows)
    write_csv(OUT / "citation_sheet.csv", ["question_id", "label", "citation", "answer", "passage_text", "supports"],
              [row for row in cite_rows])
    write_csv(OUT / "reference_drafts.csv", ["question_id", "question", "draft_answer", "final_reference", "author"], ref_rows)
    write_json("rating_key.json", key, args.dry)
    print(f"wrote {len(rating_rows)} rating rows, {len(cite_rows)} citation rows, {len(ref_rows)} reference drafts to eval/out/")


if __name__ == "__main__":
    main()
