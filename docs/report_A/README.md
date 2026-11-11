# Role A: data, index, graph, vision (code written, not yet run on real filings)

## Run order on a machine with internet

    pip install -r requirements.txt
    export SEC_USER_AGENT="Your Name you@example.com"      # SEC requires a contact address
    make data        # newest 10-K of the 20 companies -> data/passages.jsonl, companies.json, filings.json, corpus_report.json
    make graph       # company mention graph, PageRank, HITS -> data/graph/company_graph.json
    python -m pytest -q

Raw filings are cached under `data/raw/` (git-ignored), so a second run downloads nothing.
No internet for one filing? Save it as `data/raw_html/TICKER_YEAR.html` and run
`python -m trademesh.ingest.corpus --from-html data/raw_html`.

## FIRST thing to do after `make data`: read `data/corpus_report.json`

The parser was tested on synthetic filings that imitate EDGAR's HTML, not on real ones (the sandbox this was written in cannot reach sec.gov).
Real filings are messier. The report lists, per company: passages, table passages, passages per Item, median and max passage length, and warnings
(an Item not found, an Item with too few passages, no tables at all). Expect to fix the parser for a few companies. Open 5 random passages per
company and check them by eye. The expected total is roughly 6,000 to 10,000 passages; the report flags a total outside 3,000 to 15,000.

Known limits of the parser (state them in the report, do not hide them):
* Items are found by their "Item N." heading; a table of contents is skipped by keeping, for each Item, the span with the most text. A heading that
  is split oddly, or an Item cross-reference alone on a line, can cut a section short. Kept Items: 1, 1A, 1C, 2, 3, 5, 7, 7A, 8, 9A.
* `fiscal_year` is the year of the period end date.
* `page` is empty: the HTML has no reliable page numbers (the vision track works on rendered pages instead).

## What the code is

| Module | What it does | Proposal |
|---|---|---|
| `ingest/edgar.py` | EDGAR client: declared User-Agent, 8 requests/s cap, disk cache, no retry on 403 | 10.1 |
| `ingest/parse.py`, `chunk.py` | HTML to blocks to Items to passages; each table is one passage with caption, header row and labelled rows | 9.1, 9.6 |
| `index/positional.py` | positional inverted index, skip pointers (comparison count exposed), AND/OR/NOT/parentheses, phrase, proximity | 9.1 |
| `index/wildcard.py` | permuterm and 2-gram wildcard expansion, tested against brute force | 9.1 |
| `index/tfidf.py` | TF-IDF (1 + log10 tf) x log10(N/df), cosine, heap top-K, champion lists with counted fallbacks | 9.2 |
| `index/boolean.py` | Boolean retriever for the RQ1 comparison (matches ordered by term count) | RQ1 |
| `graph/mentions.py` | strict company-mention edges, PageRank (d = 0.85), HITS, correlation with in-degree | 9.7 |
| `vision/maxsim.py` | MaxSim late-interaction scoring and the results file the app reads | 9.9 |

Retriever names exposed to the app and the evaluation scripts: `tfidf`, `tfidf-champ-50`, `tfidf-champ-200`, `boolean`
(Role B adds `bm25`, `bm25+prf`, `dense`, `hybrid`). `speed_quality.py` takes the champion systems by name:
`python eval/speed_quality.py --systems tfidf,tfidf-champ-50,tfidf-champ-200`.

## Vision track (gated, not built)

Only the scoring (MaxSim) is implemented and tested. The page images and the ColPali-family encoder need a GPU and a model download
(`pip install colpali-engine`; `ColPaliEncoder` in `vision/maxsim.py` is a stub that says so). Proposal go/no-go: embed 5 pages and check that
MaxSim returns the right page for 3 of 5 hand-written questions. If that fails, cancel the track and say so in the report; the app's Pages tab
already shows "not available yet" when `data/vision/results.json` is absent.
