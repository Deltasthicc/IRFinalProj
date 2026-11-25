"""Full-corpus page-image run, step 1: download the filings, render every page to an image, extract the page text, write the questions.

    python eval/vision_full_prepare.py --work /content/drive/MyDrive/TradeMesh_vision          # on Colab (see notebooks/TradeMesh_Vision_Colab.ipynb)
    python eval/vision_full_prepare.py --work data/vision_full_work --tickers AAPL,MSFT        # a local trial

Everything is resumable: a filing, PDF or page image that already exists in --work is not made again. Needs internet (SEC EDGAR),
SEC_USER_AGENT in the environment ("Your Name your.address@example.com", the SEC requires a contact), the poppler tools
(`pdftoppm`, `pdftotext`) and a headless browser: Playwright's Chromium (`pip install playwright; playwright install chromium`).

Writes into --work:
  html/<T>.htm        the filing (cached download)
  pdf/<T>.pdf         the filing rendered on A3 pages by Chromium
  pages/<T>_p<N>.png  every page with text, 70 dpi
  pages.json          {page id: {ticker, page, image, text, digit ratio}}
  questions.json      questions with the page ids that count as correct ("gold")
Questions: for every company three that name it (balance sheet, income statement, cash-flow statement), plus the eight hand-written
ones of the small demo. A page is gold when its text matches that question's keyword rules and it is table-like (enough digits);
this is a keyword rule, so it favours text search a little, which the report of the results says.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DPI = 70
STYLE = "@page{size:297mm 420mm;margin:8mm} body{margin:0 auto}"
MIN_TEXT = 80

# generic questions, one set per company: (key, question template, rules, minimum share of digits on the page)
GENERIC = [
    ("balance sheet", "{name} consolidated balance sheet with total assets and total liabilities",
     [r"Balance Sheets?", r"Total assets", r"Total liabilities"], 0.12),
    ("income statement", "{name} consolidated income statement with net income and earnings per share",
     [r"Statements? of (Income|Operations|Earnings)|(Income|Operations|Earnings) Statements?", r"Net (income|earnings)", r"per (common )?share"], 0.12),
    ("cash flow", "{name} consolidated statement of cash flows with operating, investing and financing activities",
     [r"Cash Flows?", r"operating|operations", r"investing", r"financing"], 0.12),
]
# the eight hand-written questions of the small demo (same wording), valid for one company each
HAND = [
    ("V1", "AAPL", "Which page shows Apple's net sales by product category: iPhone, Mac, iPad, Wearables and Services?",
     [r"iPhone", r"\bMac\b", r"iPad", r"Wearables", r"Services", r"Total net sales", r"disaggregated net sales|net sales by category"]),
    ("V2", "AAPL", "Where is the table of Apple's five-year cumulative total shareholder return compared with the S&P 500 index?",
     [r"Company Stock Performance", r"S&P 500"]),
    ("V3", "MSFT", "Microsoft revenue and operating income by segment: Productivity and Business Processes, Intelligent Cloud, More Personal Computing",
     [r"SEGMENT RESULTS OF OPERATIONS", r"Productivity and Business Processes", r"Intelligent Cloud", r"More Personal Computing"]),
    ("V4", "MSFT", "Microsoft income statement with net income and earnings per share",
     [r"INCOME STATEMENTS", r"Net income", r"Earnings per share"]),
    ("V5", "TSLA", "Tesla statement of operations: automotive sales, regulatory credits and energy generation and storage revenues",
     [r"Consolidated Statements of Operations", r"Automotive sales", r"Energy generation and storage"]),
    ("V6", "TSLA", "Tesla balance sheet with total assets and total liabilities",
     [r"Consolidated Balance Sheets", r"Total assets", r"Total liabilities"]),
    ("V7", "WMT", "Walmart net sales and operating income by segment: Walmart U.S., Walmart International and Sam's Club",
     [r"Walmart U\.S\.", r"Walmart International", r"Sam's Club", r"Net sales", r"Operating income"]),
    ("V8", "WMT", "Walmart consolidated balance sheet with total assets",
     [r"Consolidated Balance Sheets", r"Total assets", r"Total liabilities"]),
]


def run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def digit_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    return sum(c.isdigit() for c in chars) / len(chars) if chars else 0.0


def render_pdf(html: Path, pdf: Path) -> None:
    """HTML to PDF on A3 pages with headless Chromium (Playwright). The filing's own image files are not downloaded."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html.as_uri(), wait_until="load", timeout=600_000)
        page.add_style_tag(content=STYLE)
        page.pdf(path=str(pdf), width="297mm", height="420mm", margin={"top": "8mm", "bottom": "8mm", "left": "8mm", "right": "8mm"},
                 print_background=True)
        browser.close()


def build_questions(pages: dict, companies: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Questions with gold page ids; a question whose rules match no page is dropped and reported."""
    qs, dropped = [], []
    by_ticker: dict[str, list[tuple[str, dict]]] = {}
    for pid, p in pages.items():
        by_ticker.setdefault(p["ticker"], []).append((pid, p))

    def gold(ticker: str, rules: list[str], min_digits: float = 0.0) -> list[str]:
        return [pid for pid, p in by_ticker.get(ticker, []) if p["digits"] >= min_digits and all(re.search(r, p["text"], re.I) for r in rules)]

    for t in sorted(by_ticker):
        for key, tpl, rules, md in GENERIC:
            g = gold(t, rules, md)
            qid = f"{t}-{key.split()[0]}"
            if g:
                qs.append({"id": qid, "group": key, "company": t, "question": tpl.format(name=companies.get(t, t)), "gold": g})
            else:
                dropped.append(f"{qid}: no page matches")
    for qid, t, question, rules in HAND:
        if t in by_ticker:
            g = gold(t, rules)
            if g:
                qs.append({"id": qid, "group": "hand-written", "company": t, "question": question, "gold": g})
            else:
                dropped.append(f"{qid}: no page matches")
    return qs, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="working folder (on Colab: a folder in Google Drive)")
    ap.add_argument("--tickers", default="", help="comma-separated tickers (default: all 20)")
    args = ap.parse_args()
    work = Path(args.work).resolve()
    from trademesh.ingest.edgar import EdgarClient
    filings = {f["ticker"]: f for f in json.loads((ROOT / "data" / "filings.json").read_text(encoding="utf-8"))}
    companies = {c["ticker"]: c["name"].replace(" Inc.", "").replace(" Corporation", "").replace(", Inc.", "").replace(" & Co.", "") for c in
                 json.loads((ROOT / "data" / "companies.json").read_text(encoding="utf-8"))}
    tickers = [t for t in (args.tickers.split(",") if args.tickers else list(filings)) if t]
    for d in ("html", "pdf", "pages", "raw"):
        (work / d).mkdir(parents=True, exist_ok=True)
    client = EdgarClient(cache_dir=work / "raw")
    pages: dict[str, dict] = {}
    for t in tickers:
        html, pdf = work / "html" / f"{t}.htm", work / "pdf" / f"{t}.pdf"
        if not html.exists():
            html.write_bytes(client.get(filings[t]["url"]))
        if not pdf.exists():
            print(f"{t}: rendering to PDF ...", flush=True)
            render_pdf(html, pdf)
        texts = run("pdftotext", "-layout", str(pdf), "-").split("\f")
        if texts and not texts[-1].strip():
            texts = texts[:-1]
        if not list((work / "pages").glob(f"{t}_p*.png")):
            run("pdftoppm", "-r", str(DPI), "-png", str(pdf), str(work / "pages" / f"{t}_tmp"))
            for png in sorted((work / "pages").glob(f"{t}_tmp-*.png")):
                n = int(png.stem.rsplit("-", 1)[1])
                png.rename(work / "pages" / f"{t}_p{n}.png")
        kept = 0
        for i, tx in enumerate(texts, 1):
            png = work / "pages" / f"{t}_p{i}.png"
            if len(tx.strip()) < MIN_TEXT or not png.exists():
                continue
            pages[f"{t}_p{i}"] = {"ticker": t, "page": i, "image": f"pages/{t}_p{i}.png", "text": re.sub(r"[ \t]+", " ", tx).strip(),
                                  "digits": round(digit_ratio(tx), 3)}
            kept += 1
        print(f"{t}: {len(texts)} pages, {kept} with text", flush=True)
    (work / "pages.json").write_text(json.dumps(pages), encoding="utf-8")
    qs, dropped = build_questions(pages, companies)
    (work / "questions.json").write_text(json.dumps(qs, indent=1), encoding="utf-8")
    groups: dict[str, int] = {}
    for q in qs:
        groups[q["group"]] = groups.get(q["group"], 0) + 1
    print(f"{len(pages)} pages, {len(qs)} questions {groups}; gold pages per question: " + str({q["id"]: len(q["gold"]) for q in qs}))
    if dropped:
        print("dropped (no gold page):", "; ".join(dropped))


if __name__ == "__main__":
    if "@" not in os.environ.get("SEC_USER_AGENT", ""):
        sys.exit("Set SEC_USER_AGENT to 'Your Name your.address@example.com' (the SEC requires a contact address).")
    main()
