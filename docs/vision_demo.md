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
