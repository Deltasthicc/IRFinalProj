"""Build data/passages.jsonl, data/companies.json and a quality report (Role A).

    export SEC_USER_AGENT="Your Name you@example.com"
    python -m trademesh.ingest.corpus                        # 20 companies from the proposal list, newest 10-K each
    python -m trademesh.ingest.corpus --tickers AAPL,MSFT    # a subset
    python -m trademesh.ingest.corpus --from-html raw_dir    # offline: files named TICKER_YEAR.html

Outputs (data/): passages.jsonl, companies.json, filings.json (accession and URL of every filing used),
corpus_report.json. READ THE REPORT: the parser was tested on synthetic filings, so a real filing with a
missing Item, a tiny Item or an odd passage count is flagged there and must be looked at by a person.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..types import Passage
from .chunk import chunk_item, words
from .parse import KEEP_ITEMS, html_to_blocks, split_items

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_TOTAL = (3000, 15000)       # sanity band around the proposal's 6,000 to 10,000
MIN_ITEM_PASSAGES = {"1": 15, "1A": 15, "7": 15, "8": 20}


def build_passages(html: str, ticker: str, company: str, fiscal_year: int) -> tuple[list[Passage], dict]:
    blocks = html_to_blocks(html)
    items = split_items(blocks)
    passages: list[Passage] = []
    per_item: dict[str, int] = {}
    for item in KEEP_ITEMS:
        if item in items:
            ps = chunk_item(items[item], ticker, company, fiscal_year, item)
            passages += ps
            per_item[item] = len(ps)
    warnings = [f"Item {i} not found" for i in KEEP_ITEMS if i in ("1", "1A", "7", "8") and i not in items]
    warnings += [f"Item {i} has only {per_item.get(i, 0)} passages" for i, n in MIN_ITEM_PASSAGES.items()
                 if i in per_item and per_item[i] < n]
    tables = sum(p.kind == "table" for p in passages)
    if passages and tables == 0:
        warnings.append("no table passages at all")
    lens = [words(p.text) for p in passages]
    report = {"ticker": ticker, "fiscal_year": fiscal_year, "blocks": len(blocks), "passages": len(passages),
              "table_passages": tables, "per_item": per_item, "median_words": sorted(lens)[len(lens) // 2] if lens else 0,
              "max_words": max(lens, default=0), "warnings": warnings}
    return passages, report


def load_company_list(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default=str(ROOT / "data/needs/companies_proposal_list.json"))
    ap.add_argument("--tickers", default="")
    ap.add_argument("--from-html", default="", help="directory of TICKER_YEAR.html files (no network)")
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--cache", default=str(ROOT / "data/raw"))
    args = ap.parse_args(argv)

    companies = load_company_list(Path(args.list))
    if args.tickers:
        want = {t.strip().upper() for t in args.tickers.split(",")}
        companies = [c for c in companies if c["ticker"] in want]
    names = {c["ticker"]: c["name"] for c in companies}
    all_passages: list[Passage] = []
    reports, filings = [], []

    if args.from_html:
        for f in sorted(Path(args.from_html).glob("*.htm*")):
            m = re.match(r"([A-Za-z.]+)_(\d{4})", f.stem)
            if not m or m.group(1).upper() not in names:
                print(f"skip {f.name}: name must be TICKER_YEAR.html with a ticker from the company list", file=sys.stderr)
                continue
            t, fy = m.group(1).upper(), int(m.group(2))
            ps, rep = build_passages(f.read_text(encoding="utf-8", errors="replace"), t, names[t], fy)
            all_passages += ps; reports.append(rep); filings.append({"ticker": t, "fiscal_year": fy, "source": f.name})
    else:
        from .edgar import EdgarClient, EdgarError
        client = EdgarClient(cache_dir=args.cache)
        ciks = client.ticker_to_cik()
        for c in companies:
            try:
                filing = client.latest_10k(c["ticker"], ciks)
                html = client.download_html(filing)
            except EdgarError as e:
                print(f"{c['ticker']}: {e}", file=sys.stderr)
                reports.append({"ticker": c["ticker"], "passages": 0, "warnings": [str(e)]})
                continue
            ps, rep = build_passages(html, c["ticker"], c["name"], filing.fiscal_year)
            all_passages += ps; reports.append(rep)
            filings.append({"ticker": c["ticker"], "fiscal_year": filing.fiscal_year, "accession": filing.accession,
                            "report_date": filing.report_date, "url": filing.url})
            print(f"{c['ticker']:5} FY{filing.fiscal_year}: {rep['passages']:5} passages ({rep['table_passages']} tables)"
                  + (f"  WARN {rep['warnings']}" if rep["warnings"] else ""))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    got = {p.ticker for p in all_passages}
    (out / "companies.json").write_text(json.dumps([c for c in companies if c["ticker"] in got], indent=1, ensure_ascii=False))
    (out / "filings.json").write_text(json.dumps(filings, indent=1))
    total = len(all_passages)
    flags = [w for r in reports for w in r.get("warnings", [])]
    if not EXPECTED_TOTAL[0] <= total <= EXPECTED_TOTAL[1] and not args.tickers:
        flags.append(f"total passages {total} outside the expected {EXPECTED_TOTAL}")
    summary = {"total_passages": total, "companies": len(got), "by_kind": dict(Counter(p.kind for p in all_passages)),
               "by_item": dict(Counter(p.item for p in all_passages)), "flags": flags}
    (out / "corpus_report.json").write_text(json.dumps({"summary": summary, "companies": reports}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
