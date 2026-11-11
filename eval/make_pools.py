"""Build the BLIND judging sheets for the pooled test collection, and merge them back. Judges nothing itself.

    python eval/make_pools.py --judges Tanish,Shashwat,Ruchir      # write sheets
    python eval/make_pools.py --merge                              # sheets -> data/judgments/qrels.csv

Pool = top `--depth` (default 8) passages of every system, for every need (dev and test). Passages are
shuffled with a fixed seed; the sheet shows no system names and no ranks. Which systems contributed
which passage is written to pool_key.json, which judges must not open. 10 needs are assigned to two
judges (for Cohen's kappa). Grades: 2 = answers the need, 1 = partly, 0 = not relevant.
Judging the test needs does not leak the test results: pooling uses system outputs, not scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.retrieval_common import JUDG, load_needs, run_system, setup, tickers_of  # noqa: E402

from trademesh.evaluation.pooling import assign_judges, build_pool  # noqa: E402

HEADER = ["query_id", "need", "passage_id", "ticker", "fiscal_year", "item", "kind", "passage_text", "rel", "judge"]


def make(args) -> None:
    passages, retrievers, reranker, dry = setup(args.allow_stub, with_reranker=True)
    by_id = {p.id: p for p in passages}
    needs = load_needs("all")
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    assignment = assign_judges([n["id"] for n in needs], judges, n_double=args.double)
    systems = {name: r.search for name, r in retrievers.items()}
    if reranker is not None and "hybrid" in retrievers:
        systems["hybrid+rerank"] = lambda q, k: reranker.rerank(q, retrievers["hybrid"].search(q, 30), k)
    runs = {name: run_system(s, needs, by_id, k=args.depth) for name, s in systems.items()}

    key, sheets = {}, {j: [] for j in judges}
    from trademesh.types import Hit
    for n in needs:
        pool, contrib = build_pool({s: [Hit(i, 0) for i in runs[s][n["id"]]] for s in runs}, depth=args.depth)
        key[n["id"]] = {"contributors": contrib, "judges": assignment[n["id"]]}
        for j in assignment[n["id"]]:
            for pid in pool:
                p = by_id[pid]
                sheets[j].append([n["id"], n["need"], pid, p.ticker, p.fiscal_year, p.item, p.kind, p.text, "", j])
    out = (JUDG / "dryrun") if dry else JUDG
    out.mkdir(parents=True, exist_ok=True)
    for j, rows in sheets.items():
        with open(out / f"pool_sheet_{j}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)
    (out / "pool_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"wrote {sum(map(len, sheets.values()))} judging rows for {len(judges)} judges to {out}"
          + ("  (DRY RUN: fixture/stub, not usable)" if dry else ""))


def merge(args) -> None:
    rows, seen = [], 0
    for f in sorted(JUDG.glob("pool_sheet_*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8", newline="")):
            seen += 1
            if r["rel"].strip() == "":
                continue
            if r["rel"].strip() not in ("0", "1", "2"):
                sys.exit(f"{f.name}: bad grade {r['rel']!r} for {r['query_id']}/{r['passage_id']} (use 0, 1 or 2)")
            rows.append([r["query_id"], r["passage_id"], r["rel"].strip(), r["judge"]])
    with open(JUDG / "qrels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["query_id", "passage_id", "rel", "judge"]); w.writerows(rows)
    print(f"merged {len(rows)} judged of {seen} sheet rows into data/judgments/qrels.csv "
          f"({seen - len(rows)} still blank)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judges", default="Tanish,Shashwat,Ruchir")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--double", type=int, default=10)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--allow-stub", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else make(a)
