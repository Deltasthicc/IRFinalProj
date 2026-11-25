"""Step 1 of the small page-image demo: choose pages, render them to PNG, write the questions and the page texts.

    python eval/vision_prepare.py

Needs: data/vision/pdf/<TICKER>.pdf (a 10-K rendered to PDF by headless Chrome, see docs/vision_demo.md) and the poppler tools
`pdftotext` and `pdftoppm` on the PATH. Writes data/vision/pages/*.png, data/vision/pages.json and data/vision/questions.json.

Pool of pages: for each of four companies, every page that matches one of the question rules ("gold" pages), the four most
number-heavy other pages (table-like distractors) and three prose pages (text distractors). Choice is deterministic.
The eight questions were written by hand; a page counts as correct for a question if it matches that question's keyword rule
on the page text. This is a very small demonstration, not a benchmark.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "data" / "vision"
TICKERS = ["AAPL", "MSFT", "TSLA", "WMT"]
DPI = 70

# (id, question, company, [regexes that must all match the page text])
QUESTIONS = [
    ("V1", "Which page shows Apple's net sales by product category: iPhone, Mac, iPad, Wearables and Services?", "AAPL",
     [r"iPhone", r"\bMac\b", r"iPad", r"Wearables", r"Services", r"Total net sales", r"disaggregated net sales|net sales by category"]),
    ("V2", "Where is the table of Apple's five-year cumulative total shareholder return compared with the S&P 500 index?", "AAPL",
     [r"Company Stock Performance", r"S&P 500"]),
    ("V3", "Microsoft revenue and operating income by segment: Productivity and Business Processes, Intelligent Cloud, More Personal Computing", "MSFT",
     [r"SEGMENT RESULTS OF OPERATIONS", r"Productivity and Business Processes", r"Intelligent Cloud", r"More Personal Computing"]),
    ("V4", "Microsoft income statement with net income and earnings per share", "MSFT",
     [r"INCOME STATEMENTS", r"Net income", r"Earnings per share"]),
    ("V5", "Tesla statement of operations: automotive sales, regulatory credits and energy generation and storage revenues", "TSLA",
     [r"Consolidated Statements of Operations", r"Automotive sales", r"Energy generation and storage"]),
    ("V6", "Tesla balance sheet with total assets and total liabilities", "TSLA",
     [r"Consolidated Balance Sheets", r"Total assets", r"Total liabilities"]),
    ("V7", "Walmart net sales and operating income by segment: Walmart U.S., Walmart International and Sam's Club", "WMT",
     [r"Walmart U\.S\.", r"Walmart International", r"Sam's Club", r"Net sales", r"Operating income"]),
    ("V8", "Walmart consolidated balance sheet with total assets", "WMT",
     [r"Consolidated Balance Sheets", r"Total assets", r"Total liabilities"]),
]


def run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def page_texts(t: str) -> list[str]:
    n = int(re.search(r"Pages:\s+(\d+)", run("pdfinfo", str(V / "pdf" / f"{t}.pdf"))).group(1))
    return [run("pdftotext", "-f", str(p), "-l", str(p), "-layout", str(V / "pdf" / f"{t}.pdf"), "-") for p in range(1, n + 1)]


def digit_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    return sum(c.isdigit() for c in chars) / len(chars) if chars else 0.0


def main() -> None:
    for t in TICKERS:
        if not (V / "pdf" / f"{t}.pdf").exists():
            sys.exit(f"missing {V / 'pdf' / (t + '.pdf')}: render the filing to PDF first (docs/vision_demo.md)")
    (V / "pages").mkdir(parents=True, exist_ok=True)
    pool: dict[str, dict] = {}
    gold: dict[str, list[str]] = {q[0]: [] for q in QUESTIONS}
    for t in TICKERS:
        texts = page_texts(t)
        chosen: set[int] = set()
        for qid, _q, comp, rules in QUESTIONS:
            if comp != t:
                continue
            hits = [i for i, tx in enumerate(texts) if all(re.search(r, tx, re.I) for r in rules)]
            if not hits:
                sys.exit(f"{qid}: no page of {t} matches its rule; adjust the rule")
            for i in hits:
                gold[qid].append(f"{t}_p{i + 1}")
                chosen.add(i)
        numeric = sorted((i for i, tx in enumerate(texts) if len(tx) > 800 and i not in chosen), key=lambda i: -digit_ratio(texts[i]))[:4]
        prose = [i for i, tx in enumerate(texts) if len(tx) > 2500 and digit_ratio(tx) < 0.06 and i not in chosen and i not in numeric]
        prose = [prose[int(k * (len(prose) - 1) / 2)] for k in range(3)] if len(prose) >= 3 else prose
        for i in sorted(chosen | set(numeric) | set(prose)):
            pid = f"{t}_p{i + 1}"
            png = V / "pages" / f"{pid}"
            if not Path(str(png) + ".png").exists():
                run("pdftoppm", "-f", str(i + 1), "-l", str(i + 1), "-r", str(DPI), "-png", "-singlefile", str(V / "pdf" / f"{t}.pdf"), str(png))
            pool[pid] = {"ticker": t, "page": i + 1, "image": str((V / "pages" / f"{pid}.png").relative_to(ROOT)).replace("\\", "/"),
                         "text": re.sub(r"[ \t]+", " ", texts[i]).strip(),
                         "kind": "gold" if i in chosen else ("numbers" if i in numeric else "prose")}
    (V / "pages.json").write_text(json.dumps(pool, indent=1), encoding="utf-8")
    qs = [{"id": qid, "question": q, "company": comp, "gold": gold[qid]} for qid, q, comp, _ in QUESTIONS]
    (V / "questions.json").write_text(json.dumps(qs, indent=1), encoding="utf-8")
    kinds = {k: sum(1 for p in pool.values() if p["kind"] == k) for k in ("gold", "numbers", "prose")}
    print(f"{len(pool)} pages in the pool {kinds}; {len(qs)} questions; gold pages per question:",
          {q['id']: len(q['gold']) for q in qs})


if __name__ == "__main__":
    main()
