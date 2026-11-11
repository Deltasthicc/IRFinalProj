"""Apply the pre-registered decision rules to the locked test run. No judgement calls at report time.

    python eval/hypotheses.py                      # reads eval/out/retrieval_test.json
Metric: nDCG@10 (binary gain), paired bootstrap over the 20 test needs, 95% CI of the mean difference.
Rule for "A beats B": CI lower bound > 0 -> SUPPORTED; CI upper bound < 0 -> REFUTED (B beats A);
otherwise INCONCLUSIVE ("tied" if |mean difference| < 0.02). With 20 queries many outcomes will be
inconclusive, and the report says so instead of reading a trend into noise. Covered here:
H1 bm25 > tfidf   H2 bm25 ~ dense on entity+terminology needs, dense > bm25 on topical needs
H3 hybrid >= bm25 and >= dense   H4 hybrid+rerank > hybrid   H6 bm25+prf > bm25 on terminology, < on entity
(H5, H7 are read off ablation_test.json and speed_quality_test.json; H8-H11 belong to other roles.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import OUT, ROOT  # noqa: E402

from trademesh.evaluation.stats import paired_bootstrap  # noqa: E402

M = "nDCG@10"


def compare(per_query, types, a, b, only_type=None):
    ids = [q for q in per_query[a] if only_type is None or types[q] == only_type]
    if not ids:
        return None
    r = paired_bootstrap([per_query[a][q][M] for q in ids], [per_query[b][q][M] for q in ids])
    r["verdict"] = ("A>B SUPPORTED" if r["ci_low"] > 0 else "B>A (A REFUTED)" if r["ci_high"] < 0
                    else "TIED (inconclusive)" if abs(r["mean_diff"]) < 0.02 else "INCONCLUSIVE")
    return r


def evaluate(per_query: dict, types: dict, names: dict | None = None) -> list[tuple[str, str, dict | None]]:
    n = {"tfidf": "tfidf", "bm25": "bm25", "dense": "dense", "hybrid": "hybrid", "rerank": "hybrid+rerank",
         "prf": "bm25+prf", **(names or {})}
    have = lambda *s: all(n[x] in per_query for x in s)
    out = []
    if have("bm25", "tfidf"):
        out.append(("H1", "bm25 vs tfidf (all needs)", compare(per_query, types, n["bm25"], n["tfidf"])))
    if have("bm25", "dense"):
        for t in ("entity", "terminology"):
            out.append(("H2a", f"bm25 vs dense, {t} needs (hoped: tied or better)", compare(per_query, types, n["bm25"], n["dense"], t)))
        out.append(("H2b", "dense vs bm25, topical needs (hoped: dense better)", compare(per_query, types, n["dense"], n["bm25"], "topical")))
    if have("hybrid", "bm25", "dense"):
        out.append(("H3a", "hybrid vs bm25", compare(per_query, types, n["hybrid"], n["bm25"])))
        out.append(("H3b", "hybrid vs dense", compare(per_query, types, n["hybrid"], n["dense"])))
    if have("rerank", "hybrid"):
        out.append(("H4", "hybrid+rerank vs hybrid", compare(per_query, types, n["rerank"], n["hybrid"])))
    if have("prf", "bm25"):
        out.append(("H6a", "bm25+prf vs bm25, terminology needs (hoped: better)", compare(per_query, types, n["prf"], n["bm25"], "terminology")))
        out.append(("H6b", "bm25+prf vs bm25, entity needs (hoped: not better)", compare(per_query, types, n["prf"], n["bm25"], "entity")))
    return out


def main() -> None:
    path = OUT / "retrieval_test.json"
    if not path.exists():
        sys.exit("TODO(run): eval/out/retrieval_test.json does not exist; run eval/run_retrieval.py --split test after the freeze.")
    data = json.loads(path.read_text())
    from eval.common import NEEDS, read_csv
    types = {r["id"]: r["type"] for r in read_csv(NEEDS / "needs.csv")}
    rows = evaluate(data["per_query"], types)
    print(f"{'id':5} {'comparison':58} {'diff':>7}  {'95% CI':>17}  W/L/T   verdict")
    for h, label, r in rows:
        if r is None:
            print(f"{h:5} {label:58} (no needs of that type)"); continue
        print(f"{h:5} {label:58} {r['mean_diff']:+7.3f}  [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]  "
              f"{r['wins']}/{r['losses']}/{r['ties']}   {r['verdict']}")
    (OUT / "hypotheses_test.json").write_text(json.dumps([{"id": h, "comparison": l, **(r or {})} for h, l, r in rows], indent=1))


if __name__ == "__main__":
    main()
