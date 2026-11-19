"""Prepare the HUMAN rating package for the RAG study. It rates nothing and contains no model ratings.

    python eval/human_prepare.py                       # write eval/human/ (blind answers, key, retrieval sample)
    python eval/human_prepare.py --names Shashwat,Ruchir --seed 1119

What it writes (all under eval/human/):
  blind_answers.json         the 20 test questions with three anonymous answers each (labels A/B/C, shuffled with a NEW
                             seed, so the order differs from the model-rated sheets) and the passages each answer cites.
                             No system names, no scores, no model ratings.
  key_DO_NOT_OPEN.json       label -> system for every question. Raters must not open it before they have finished.
  judgments/pool_sheet_<name>.csv   blank relevance sheets (retrieval validation sample) for eval/judge_app.py.

Reference material shown to the rater (never a rating): for each question the passages that the pooled retrieval
judgments marked relevant (these judgments were made by a language model, so the app says so). The rater writes their own
reference answer first and judges against the filing, not against those hints.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import NEEDS, OUT, ROOT, read_csv  # noqa: E402

from trademesh.data import DATA, load_passages  # noqa: E402

HUMAN = ROOT / "eval" / "human"
CONFIGS = ["closed_book", "bm25_rag", "best_rag"]
LABELS = ["A", "B", "C"]
JUDG = DATA / "judgments"


def build_package(runs: dict, passages: dict, qrels: list[dict], seed: int) -> tuple[dict, dict]:
    """(blind, key). `runs` is rag_runs_test.json, `passages` maps id -> Passage, `qrels` the pooled judgments."""
    rng = random.Random(seed)
    helper: dict[str, list[str]] = {}
    for r in qrels:
        if r["rel"] == "2" and r["passage_id"] in passages:
            helper.setdefault(r["query_id"], [])
            if r["passage_id"] not in helper[r["query_id"]]:
                helper[r["query_id"]].append(r["passage_id"])
    blind_q, key = [], {}
    for run in runs["runs"]:
        order = CONFIGS[:]
        rng.shuffle(order)
        key[run["id"]] = dict(zip(LABELS, order))
        answers = {}
        for label, cfg in zip(LABELS, order):
            res = run[cfg]
            cites = []
            for i in res.get("cited") or []:
                ids = res["passage_ids"]
                if 1 <= i <= len(ids) and ids[i - 1] in passages:
                    p = passages[ids[i - 1]]
                    cites.append({"ref": f"P{i}", "ticker": p.ticker, "fiscal_year": p.fiscal_year, "item": p.item, "text": p.text})
            answers[label] = {"text": res["answer"], "cited": cites}
        hints = [{"ticker": passages[i].ticker, "fiscal_year": passages[i].fiscal_year, "item": passages[i].item,
                  "text": passages[i].text} for i in helper.get(run["id"], [])[:6]]
        blind_q.append({"id": run["id"], "question": run["question"], "type": run["type"], "answers": answers, "hints": hints})
    return {"seed": seed, "labels": LABELS, "questions": blind_q}, key


def retrieval_sample(needs: list[dict], seed: int = 358, per_type: dict | None = None) -> list[str]:
    """A small stratified sample of TEST needs for validating the model's relevance judgments."""
    per_type = per_type or {"entity": 3, "topical": 3, "terminology": 2}
    rng = random.Random(seed)
    out = []
    for t, n in per_type.items():
        ids = sorted(x["id"] for x in needs if x["split"] == "test" and x["type"] == t)
        out += sorted(rng.sample(ids, min(n, len(ids))))
    return out


def write_retrieval_sheets(names: list[str], need_ids: list[str] | None, out: Path) -> int:
    """Blank copies of the pool sheet (same passages, same blind order, grade column empty)."""
    src = JUDG / "pool_sheet_AI.csv"
    with open(src, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if need_ids is not None:
        rows = [r for r in rows if r["query_id"] in need_ids]
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        with open(out / f"pool_sheet_{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({**r, "rel": "", "judge": name})
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="Shashwat", help="comma-separated rater names")
    ap.add_argument("--seed", type=int, default=1119, help="shuffle seed for the A/B/C labels")
    ap.add_argument("--retrieval-needs", default="sample", help="'sample' (8 test needs), 'all' (all 20 test needs) or 'none'")
    ap.add_argument("--force", action="store_true", help="overwrite an existing package (this changes the labels!)")
    ap.add_argument("--sheets-only", action="store_true",
                    help="only (re)write the blank retrieval sheets, e.g. for one more rater; the answer package and labels are untouched")
    args = ap.parse_args()
    if args.sheets_only:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
        needs = read_csv(NEEDS / "needs.csv")
        ids = [n["id"] for n in needs if n["split"] == "test"] if args.retrieval_needs == "all" else retrieval_sample(needs)
        for name in names:
            if (HUMAN / "judgments" / f"pool_sheet_{name}.csv").exists():
                sys.exit(f"eval/human/judgments/pool_sheet_{name}.csv already exists; delete it first if you want a blank one.")
        n = write_retrieval_sheets(names, ids, HUMAN / "judgments")
        print(f"wrote {n} blank retrieval judging rows per rater ({', '.join(names)}) for needs {', '.join(ids)}")
        return
    if (HUMAN / "blind_answers.json").exists() and not args.force:
        sys.exit("eval/human/blind_answers.json already exists. Re-running would reshuffle the labels and orphan any ratings; "
                 "pass --force only if nobody has rated yet.")
    runs = json.loads((OUT / "rag_runs_test.json").read_text(encoding="utf-8"))
    if not runs.get("reportable"):
        sys.exit("rag_runs_test.json is not marked reportable (dry run).")
    passages = {p.id: p for p in load_passages(DATA / "passages.jsonl")}
    qrels = read_csv(JUDG / "qrels.csv")
    blind, key = build_package(runs, passages, qrels, args.seed)
    HUMAN.mkdir(parents=True, exist_ok=True)
    (HUMAN / "blind_answers.json").write_text(json.dumps(blind, indent=1, ensure_ascii=False), encoding="utf-8")
    (HUMAN / "key_DO_NOT_OPEN.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    n_cites = sum(len(a["cited"]) for q in blind["questions"] for a in q["answers"].values())
    print(f"wrote eval/human/blind_answers.json: {len(blind['questions'])} questions, "
          f"{len(blind['questions']) * len(LABELS)} answers to rate, {n_cites} citations to check")
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if args.retrieval_needs != "none":
        needs = read_csv(NEEDS / "needs.csv")
        ids = [n["id"] for n in needs if n["split"] == "test"] if args.retrieval_needs == "all" else retrieval_sample(needs)
        n = write_retrieval_sheets(names, ids, HUMAN / "judgments")
        print(f"wrote {n} blank retrieval judging rows per rater for needs {', '.join(ids)} to eval/human/judgments/")


if __name__ == "__main__":
    main()
