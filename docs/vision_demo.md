# Page-image retrieval: the small demonstration (Pages tab)

This is a **small** demonstration of the optional vision track (RQ5, H11), added after the report was submitted. It searches page **images** of four 10-K filings with a ColPali-family model and compares the result with plain text search over the same pages. It is not a benchmark and it is not in the submitted report.

## What it is

* **Pages:** 39 pages from the latest 10-K of Apple, Microsoft, Tesla and Walmart (11 pages that contain the answer to a question, 16 number-heavy pages that look similar, 12 prose pages), rendered to images at 70 dpi.
* **Questions:** 8, written by hand, mostly about tables (net sales by product, revenue by segment, income statement, balance sheet, shareholder-return table). A page counts as correct if its text matches that question's keyword rule (see `eval/vision_prepare.py`).
* **Vision model (pre-trained, used as a component):** `vidore/colSmol-256M`, a 256-million-parameter ColPali-family model. It turns each page image into about 875 vectors and each question into one vector per token.
* **Scoring (ours):** MaxSim, in `src/trademesh/vision/maxsim.py`: for each query vector take its best-matching page vector and sum these maxima.
* **Text baseline (ours):** BM25 over the text extracted from the same pages with `pdftotext`.
* **Output:** `data/vision/results.json` (read by the app's Pages tab) and `eval/out/vision_eval.json`.

## Why it is small

The laptop has an RTX 3050 with 4 GB of memory, but the Python environment uses the CPU build of PyTorch, so the encoder ran on the CPU at roughly 20 to 45 seconds per page. The larger ColPali models (about 3 billion parameters) do not fit in 4 GB. Eight questions on 39 pages can show that the idea works; it cannot show which method is better in general.

Two honest limits: the filing HTML references its charts as separate image files, which were not downloaded, so a page that should show a chart shows an empty frame (the numbers around it are still real); and the questions and the keyword rules for correct pages were written by us.

## How to reproduce

```
# 1. download four 10-Ks (needs SEC_USER_AGENT with a contact address) and render them to PDF with Chrome: see the commands in this repository's history (data/vision/html, data/vision/pdf)
# 2. choose and render pages, write the questions
.\.venv\Scripts\python.exe eval/vision_prepare.py
# 3. encode with the vision model, in its own environment (colpali-engine, torch, transformers)
python -m venv .venv-vision; .\.venv-vision\Scripts\python.exe -m pip install colpali-engine
.\.venv-vision\Scripts\python.exe eval/vision_encode.py
# 4. score with our MaxSim and BM25, write the results
.\.venv\Scripts\python.exe eval/vision_score.py
```

Steps 1 and 3 need internet (SEC EDGAR; the model download from Hugging Face) and poppler's `pdftotext` and `pdftoppm`. The committed `data/vision/pages/*.png` and `data/vision/results.json` are enough to run the app and the Pages tab without any of this.

## The full-corpus version (Google Colab)

The small demonstration above uses 39 pages and a 256-million-parameter model on the laptop CPU. The full version runs on a Colab GPU over **every page of all 20 filings** (about 2,000 pages) with a larger ColPali-family model (`colqwen2` by default; `colpali`, `colsmol-500m` and `colsmol-256m` are options). It is in `notebooks/TradeMesh_Vision_Colab.ipynb` and uses three scripts that also run elsewhere:

| Script | What it does |
|---|---|
| `eval/vision_full_prepare.py` | downloads the 20 filings, renders every page (headless Chromium, A3 pages, 70 dpi), extracts each page's text, writes the questions: three per company (balance sheet, income statement, cash-flow statement) plus the eight hand-written ones |
| `eval/vision_full_encode.py` | encodes every page image and every question (GPU, resumable, one file per page) |
| `eval/vision_full_score.py` | ranks all pages for every question with our MaxSim and, as a baseline, BM25 over the page text; writes `data/vision_full/results.json` (what the Pages tab reads), about 200 page thumbnails and `eval/out/vision_full_eval.json` |

How to use it: open the notebook in Colab (T4 GPU), store the SEC contact string as the Colab Secret `SEC_USER_AGENT`, run the cells top to bottom (the long steps are resumable because the work is kept in a Drive folder), then either download the result bundle and unzip it into the project folder or switch on the optional push cell. The app prefers `data/vision_full/results.json` over the small demonstration when it exists, and the Pages tab then shows the full run: tiles, a table per question group, and a question picker.

What the full run can and cannot show: with about 68 questions where every page of every filing competes, the comparison between page images and text is much more informative than the 8-question demo, but the questions are still ours and the "correct page" rule is a keyword match on the page text (which favours the text baseline), so it is a demonstration at scale, not a published benchmark. The filings' charts are separate image files that are not downloaded.

## Result of the full run (Colab, Tesla T4, `colqwen2`, float16, about 33 minutes of encoding)

2,606 pages of the 20 filings, 66 questions (19 balance sheet, 20 income statement, 19 cash flow, 8 hand-written; two company questions had no matching page and were dropped), every page competing for every question:

| Question group | Questions | Page images: first result correct | Text BM25: first result correct | Page images: top 3 | Text BM25: top 3 |
|---|---|---|---|---|---|
| balance sheet | 19 | 9 | 4 | 11 | 12 |
| cash flow | 19 | 2 | 11 | 7 | 15 |
| hand-written | 8 | 4 | 4 | 7 | 6 |
| income statement | 20 | 1 | 11 | 5 | 15 |
| **all** | **66** | **16** | **30** | **30** | **48** |

Mean reciprocal rank: page images 0.392, text 0.616. So at full scale text search is better overall; page images are better on balance sheets and on layout-dependent segment tables and much worse on income and cash-flow statements. The 39-page demo above (8 of 8 against 7 of 8) therefore looked much better than the method is at scale. The likely reasons (not tested) are the limited resolution at which a whole A3 page is shown to the model, many near-identical statement pages across 20 companies, and a keyword-based definition of a correct page that favours the text baseline. Files: `data/vision_full/results.json`, `data/vision_full/eval.json`, `eval/out/vision_full_eval.json`.
