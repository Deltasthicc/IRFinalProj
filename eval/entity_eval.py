"""Entity accuracy on the 40 typo'd company names or tickers (proposal 10.3).

Metric: share of typo'd inputs mapped to the right ticker. Wrong ticker and "no match" are both
errors; we report them separately. We also compare the bigram-shortlist matcher with a brute
force Levenshtein scan over the whole dictionary, to show the shortlist loses nothing.

Run:  python eval/entity_eval.py
Output: eval/out/entity_eval.json  (every number comes from this run)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import NEEDS, ROOT, read_csv, write_json  # noqa: E402

from trademesh.data import DATA, load_companies  # noqa: E402
from trademesh.query.entities import EntityMatcher  # noqa: E402


def load_dictionary() -> tuple[list[dict], str]:
    real = DATA / "companies.json"
    if real.exists():
        return load_companies(real), str(real.relative_to(ROOT))
    path = NEEDS / "companies_proposal_list.json"
    return load_companies(path), str(path.relative_to(ROOT)) + " (proposal list; A's data/companies.json not found)"


def evaluate(companies: list[dict], rows: list[dict]) -> dict:
    shortlist = EntityMatcher(companies)
    brute = EntityMatcher(companies, shortlist=10**6, min_jaccard=0.0)
    results, by_kind = [], defaultdict(lambda: [0, 0])
    agree = 0
    for r in rows:
        m, b = shortlist.match_string(r["typo"]), brute.match_string(r["typo"])
        got = m.ticker if m else None
        ok = got == r["ticker"]
        agree += (got == (b.ticker if b else None))
        kind = r["kind"].split()[0] if r["kind"].startswith("ticker") else r["kind"]
        by_kind[kind][0] += ok
        by_kind[kind][1] += 1
        results.append({"typo": r["typo"], "expected": r["ticker"], "got": got, "ok": ok,
                        "distance": m.distance if m else None, "kind": r["kind"]})
    n = len(rows)
    correct = sum(x["ok"] for x in results)
    return {
        "n": n, "correct": correct, "accuracy": correct / n,
        "wrong_ticker": sum(1 for x in results if x["got"] and not x["ok"]),
        "no_match": sum(1 for x in results if x["got"] is None),
        "agreement_with_brute_force": agree / n,
        "by_kind": {k: {"correct": c, "n": t} for k, (c, t) in sorted(by_kind.items())},
        "failures": [x for x in results if not x["ok"]],
        "results": results,
    }


def main() -> None:
    companies, source = load_dictionary()
    rows = read_csv(NEEDS / "typos.csv")
    out = evaluate(companies, rows)
    out["dictionary"] = source
    path = write_json("entity_eval.json", out)
    print(f"dictionary: {source}")
    print(f"entity accuracy: {out['correct']}/{out['n']} = {out['accuracy']:.3f}  "
          f"(wrong ticker {out['wrong_ticker']}, no match {out['no_match']})")
    print(f"agreement with brute-force Levenshtein: {out['agreement_with_brute_force']:.3f}")
    for k, v in out["by_kind"].items():
        print(f"  {k:15s} {v['correct']}/{v['n']}")
    for f in out["failures"]:
        print(f"  FAIL {f['typo']!r}: expected {f['expected']}, got {f['got']}")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
