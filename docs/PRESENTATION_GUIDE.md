# TradeMesh: presentation guide

How to present the project, run the demo, and connect every part of the report to the website and the code. All numbers come from the report (`docs/report/TradeMesh_Report.pdf`) and the result files in `eval/out/`.

## 1. The project in one minute

**Problem.** A Form 10-K (the yearly report of a US listed company) is hundreds of pages long, but the answer to a question is usually one sentence or one table row. Keyword search returns passages, not answers, and mixes up companies. A language model gives fluent answers but can quote an old figure or invent one, and does not show its source.

**Idea.** Search first, language model second, checker last. The model sees only the best five passages, must cite them as [P1] to [P5], and a program then verifies the citations and every number. If the evidence is thin, the system says "insufficient evidence".

**What we built.**
1. A search engine written from the course material (positional inverted index with skip pointers, Boolean / phrase / wildcard search, TF-IDF with cosine, champion lists, BM25, spelling-tolerant company matching).
2. Modern retrieval on top (dense sentence vectors, rank fusion, cross-encoder reranking, pseudo-relevance feedback).
3. A cited, checked answer layer with refusal ("insufficient evidence").
4. A company mention graph (PageRank, HITS).
5. A test collection we built (30 questions, pooled relevance judgments), an evaluation with pre-declared hypotheses, bootstrap intervals and blind human ratings.
6. A six-tab web app and a Docker image.

**The corpus.** The latest 10-K of 20 companies (AAPL, MSFT, NVDA, TSLA, AMZN, NFLX, JPM, GS, BAC, XOM, CVX, WMT, KO, PEP, PFE, JNJ, UNH, BA, DIS, INTC), cut into 11,964 passages (10,912 text, 1,052 table chunks, 1.46 million words).

### How it works (say this over the pipeline figure, Fig. 1 in the report)

Offline, once: download from SEC EDGAR, split by 10-K section (every table stays whole), build the indexes.
For every question:
1. **Understand**: find company names even when misspelt ("Telsa" becomes Tesla) and use the company to filter; a question about two companies becomes two searches.
2. **Retrieve**: BM25 and dense search each return candidates; reciprocal rank fusion merges the two lists.
3. **Rerank**: a cross-encoder rereads the top 30 and keeps five.
4. **Answer**: the language model writes an answer using only those five passages, citing each claim, temperature 0, every call cached.
5. **Check**: a plain program tests that each cited [Pn] exists, every sentence is cited, and every number appears in a cited passage.
6. **Decide**: show the answer with a trust panel, or decline when the model says the evidence is insufficient.

## 2. Novelty (say it modestly, as the report does)

We do **not** claim a new model. The novelty is the combination and the evaluation:

1. **From-scratch classic core next to modern parts**, compared like-for-like (ten retrieval setups, same questions, same judgments) on financial passages.
2. **A checkable answer layer**: a deterministic verifier for citations and numbers plus a refusal rule. We also report a negative result: a threshold on the retrieval score did not separate answerable from unanswerable questions (precision 0.38, recall 0.83), so we use the model's own verdict, which declined 6 of 6 unanswerable test questions and 0 of 20 answerable ones.
3. **Finance-aware handling**: every table is one chunk (so a figure stays with its row and column labels), and a typo-tolerant company matcher written from scratch (bigram shortlist plus Levenshtein; 39 of 40 misspelt names fixed).
4. **Evaluation discipline**: 11 hypotheses written before the test run, one locked test run, paired bootstrap intervals, pool-bias checks, blind human ratings by two raters with kappa 0.39, and honest reporting of unclear results (one reference answer, T10, was corrected after the first scoring and both results are reported).
5. **Company mention graph** with PageRank and HITS, and an honest negative result (PageRank matches simple in-degree, Spearman 0.93).
6. **Reproducible and runnable**: Docker, an offline answer cache, 137 automatic tests, every number in the report regenerated from a script.

What we did **not** build: the page-image (ColPali-style) track. The MaxSim scoring is implemented and tested; the encoder needs a GPU. The Pages tab says so.

## 3. Run the demo

### Before you present (day before, and 15 minutes before)

| Step | Command or action |
|---|---|
| 1 | Plug in the laptop, connect to the internet, close heavy programs (Docker Desktop makes this machine slow). |
| 2 | Free port 8501 (close the rating tool or any other Streamlit). |
| 3 | `.env` has `LLM_PROVIDER=gemini` and your `GEMINI_API_KEY` (never show it on screen). |
| 4 | Check the cache works offline (every demo question must be answered from the cache): see the block below. |
| 5 | Start the app **at least 5 minutes before** you present. The first start after a restart can take about 4 minutes (it loads the dense model); a warm start takes about a minute. Open the page and wait until the tab bar appears. |
| 6 | In the sidebar check: retriever **hybrid**, **cross-encoder reranker ticked**, tau 0.00, lambda 0.00. These are the reported settings. |
| 7 | Do one dry run of scenes 1 to 4. Screen-record it (Win+Alt+R) as a backup. |

```powershell
cd C:\Users\shash\Downloads\IRFinalProject

# 1. check that every demo question is cached (no network needed)
$env:LLM_OFFLINE="1"; .\.venv\Scripts\python.exe eval/demo_warm.py; Remove-Item Env:LLM_OFFLINE

# 2. start the app
.\.venv\Scripts\python.exe -m streamlit run src/trademesh/app/streamlit_app.py
# then open http://localhost:8501
```

If the venue has no internet: start it with `$env:LLM_OFFLINE="1"` first. The four example buttons are exactly the cached questions, so they still work; a new question would show "not in the answer cache".

Fallbacks: another port (`--server.port=8502`); Docker (`docker compose up --build`, or `$env:INSTALL_ML="1"; docker compose up --build` for the full setup); the backup video. If the page says "Building the search indexes", wait and do not reload.

### Live script (about 8 minutes; full version in `docs/demo_script.md`)

| Time | Do this | Say this |
|---|---|---|
| 0:00 | Show the home screen (11,964 passages, 20 companies, status chips). | "TradeMesh answers questions over 10-K filings. Every claim has a citation, a checker verifies the numbers, and it says so when the evidence is thin." |
| 0:30 | Click **Ask with a typo** ("Telsa"). | "Our own spelling module corrected Telsa to Tesla (39 of 40 names fixed). The tag P1 is the passage that backs the claim." Open the evidence card. |
| 1:30 | Ask tab, **Figure / segments** (Microsoft revenue by segment). | "The verifier checks every number against the cited passage. An invented figure would show red." |
| 2:15 | Switch **Closed-book** on, then off. | "Same model without the filings: fluent, outdated, no citations. In our blind rating, retrieval scored 1.20 out of 2 against 0.575." |
| 3:00 | **Compare two companies**. | "Two companies means one search per company, merged." |
| 3:45 | **Trap / future year** (Apple 2031). | "It declines instead of guessing. Be honest: our unanswerable questions are easy, and a score threshold did not work." |
| 4:30 | Search tab: type `supplier concentration`, pick bm25, tfidf, hybrid. | "Our retrievers side by side. 'Why this score' shows each word's contribution." Optionally pick **boolean** and type `"supply chain" AND (semiconductor OR chips)` or `cyber*`. |
| 5:15 | Evaluation tab. | "Hybrid with reranking 0.828 nDCG@10 against 0.699 for BM25. Two humans rated 60 answers blind." Name the limits. |
| 6:15 | Graph tab. | "Who mentions whom. PageRank correlates 0.93 with plain counting, so it adds little." |
| 6:45 | System tab; type `Micorsoft` in the typo lab. | "Readiness is honest: the page-image track is not built, and the Pages tab says so." Optionally type `MSTF` to show the known miss. |

**Avoid typing live:** possessive names ("Apple's", "Microsoft's"), which skip the company filter (known defect), and brand-new questions when offline.

## 4. Report section to demo to code

Report sections use the paper's numbering (Roman numerals). Code paths are relative to the repository root.

| Report | What it says | Where to show it in the demo | Where it lives in the code and results |
|---|---|---|---|
| **I. Introduction** (problem, importance, contributions) | Why filings are hard to search and why unchecked answers are risky. | Home screen and the tagline "Ask the filings. Trust every number." | Whole repository; `README.md` |
| **II. Related work** (Table I, 25 papers) | Papers used for each part. | Not in the app. Show Table I on a slide. | `docs/report/parts/10_references.md`, `docs/report_B/references.md` |
| **III. Objective** (RQ1 to RQ5, 11 hypotheses) | The questions we test. | Evaluation tab answers RQ1 and RQ3; Graph tab answers RQ4; Pages tab states RQ5 was not built. | `eval/hypotheses.py` (decision rules), `eval/out/hypotheses_test.json` |
| **IV. Proposed model** (Fig. 1 pipeline, app screenshots) | Offline build plus the six online steps. | **Ask tab**: the strip of seven chips (Query, Entity, Retrieve, Context, LLM, Verify, Decision) is the pipeline. | `src/trademesh/rag/pipeline.py`, `src/trademesh/app/streamlit_app.py` |
| **V-A. Ten retrieval setups** (Table II) | Boolean, TF-IDF, champion lists, BM25, BM25 + feedback, dense, hybrid, reranked variants. | **Search tab**: the "Retrievers (up to 3)" box lists them; sidebar "Retriever for answers" and the reranker tick. | `src/trademesh/retrievers.py` (the registry), `src/trademesh/index/`, `src/trademesh/ranking/` |
| **V-B. Search engine from scratch** (index, skip pointers, Boolean, phrase, wildcard, TF-IDF, champion lists, BM25) | The classic core and its formulas. | **Search tab**: choose `bm25` and open "Why this score"; choose `boolean` and try a phrase, AND/OR/NOT or `cyber*`. | `index/positional.py`, `index/boolean.py`, `index/wildcard.py`, `index/tfidf.py`, `ranking/bm25.py`; BM25 tuning in `eval/tune_bm25.py` |
| **V-C. Modern retrieval** (dense, RRF, cross-encoder, feedback) | Sentence vectors, rank fusion, reranker. | Sidebar: retriever **hybrid** and the reranker tickbox; Search tab compares `dense`, `hybrid`, `bm25+prf`. | `ranking/dense.py`, `ranking/fusion.py`, `ranking/rerank.py`, `ranking/prf.py` |
| **V-D. Company names** (spelling-tolerant matching, compare mode) | Bigram shortlist plus Levenshtein, company filter, split by company. | Ask tab: **Ask with a typo** and **Compare**; System tab typo lab. | `query/entities.py`, `query/section_prior.py` |
| **V-D. Company graph** (PageRank, HITS) | Who mentions whom. | **Graph tab**. | `graph/mentions.py`, `data/graph/company_graph.json` |
| **V-D. Answers and verifier** (prompt, [P1] to [P5], checks, refusal) | Cited answers, deterministic checks, "insufficient evidence". | **Ask tab**: answer with P-tags, evidence cards, the **trust panel** (Citations exist, Answer cited, Every sentence cited, Numbers grounded). | `rag/prompt.py`, `rag/verifier.py`, `rag/abstain.py`, `rag/llm.py` (cache and provider) |
| **V-E. Setup and software** (Table III) | Models, versions, seeds. | **System tab** readiness checklist; sidebar shows the LLM and cache counts ("0 live / N cache" chip). | `requirements*.txt`, `.env.example`, `Dockerfile` |
| **V-F. Evaluation protocol** (pooling, metrics, bootstrap, human ratings) | How the numbers were produced. | **Evaluation tab** (three sub-tabs: Retrieval, RAG answers, Abstention). | `src/trademesh/evaluation/` (metrics, pooling, qrels, stats); `eval/make_pools.py`, `eval/run_retrieval.py`, `eval/human_*.py` |
| **VI-A. Data** (Table IV, corpus figure) | 20 companies, 11,964 passages, 30 needs, 10 unanswerable, 40 typos. | Home chips (passages, companies); System tab headline tiles. | `data/passages.jsonl`, `data/needs/` (needs, unanswerable, typos), `data/judgments/qrels.csv`, `data/corpus_report.json`; crawler in `src/trademesh/ingest/` |
| **VI-B. Retrieval results** (Table V, nDCG bars, PR curve, by type) | Hybrid + reranker 0.828 vs BM25 0.699. | **Evaluation tab, Retrieval**: nDCG bars, metric table, precision-recall chart, by-need-type table. | `eval/out/retrieval_test.json` (one locked run); `eval/run_retrieval.py` |
| **VI-C. Design choices and hypotheses** (ablations, speed, 11 hypotheses) | What helped (metadata prefix), what did not (stemming, feedback). | Not in the app: show the report figures. | `eval/ablation.py`, `eval/out/ablation_test.json`, `eval/speed_quality.py`, `eval/cost_table.py` |
| **VI-D. Spelling correction** | 39 of 40 fixed; MSTF is the miss. | System tab typo lab (`Micorsoft`, then `MSTF`). | `eval/entity_eval.py`, `eval/out/entity_eval.json` |
| **VI-E. Answer quality** (Tables VIII, IX; rating plots) | Best retriever 1.20, BM25 answers 0.90, no retrieval 0.575; kappa 0.39. | **Evaluation tab, RAG answers** (human-rated panel first); Ask tab **Closed-book** toggle for the live contrast. | `eval/rag_run.py`, `eval/human_app.py`, `eval/human_score.py`, `eval/out/rag_scores_human.json`, `eval/human/raters/` |
| **VI-F. Abstention** (Table X) | Declined 6 of 6 unanswerable; score threshold failed. | Ask tab **Trap / future year**; **Evaluation tab, Abstention**; sidebar tau. | `rag/abstain.py`, `eval/abstention_eval.py` |
| **VI-G. Company graph** | 20 nodes, 14 links, Spearman 0.93. | **Graph tab**. | `graph/mentions.py` |
| **VI-H. Comparison with published work** (Table XI) | Agrees on fusion and on retrieval helping; BM25 not ahead of dense here. | Not in the app: show the table. | `docs/report_B/10_paper_comparison.md` |
| **VI-I to K. Discussion, failure cases, cost** | Table questions are hardest; the number checker never caught an error on this data. | System tab; the known "'s" defect (do not type possessives). | `docs/report_C/10_rag_results.md`, `eval/out/cost_table.json` |
| **VII. Conclusions and limitations** | Small test set, model-made labels, two team raters, no page-image track. | **Pages tab** (page-image retrieval says "not available", by design). | `src/trademesh/vision/maxsim.py` (MaxSim written and tested) |

## 5. Suggested slides (10 to 12, about 10 minutes plus demo)

1. Title and team (names and roll numbers).
2. The problem: long filings, keyword search, unchecked language models.
3. The idea: search first, model second, checker last; contributions.
4. Pipeline diagram (Fig. 1) and the app.
5. Built from scratch: index, TF-IDF, BM25 (two formulas on the slide).
6. Modern retrieval: dense, reciprocal rank fusion, reranker.
7. Checkable answers: flow diagram and the Tesla example.
8. Data and test collection: 20 companies, 30 questions, pooling, model-made labels.
9. Retrieval results: nDCG bar chart and Table V.
10. Answer quality and abstention: 1.20 vs 0.575, kappa 0.39, 6 of 6 declined.
11. **Live demo** (the script above).
12. Limitations and future work; questions.

## 6. Questions you may be asked (short answers)

* **What is built from scratch?** The positional index with skip pointers, Boolean/phrase/wildcard search, TF-IDF, BM25, champion lists, spelling correction, rank fusion, the verifier, the metrics, pooling and the bootstrap. The sentence encoder and the cross-encoder are pre-trained models used as parts.
* **How does BM25 differ from TF-IDF?** It saturates term frequency (k1) and normalises by passage length (b). We tuned k1 = 0.6, b = 0.9 on the 10 tuning questions only.
* **What is reciprocal rank fusion?** Each passage scores 1 / (60 + rank) from each list; the scores are added. It needs only ranks, so BM25 and cosine scores need no calibration.
* **Why a reranker on the top 30?** A cross-encoder reads question and passage together, which is more accurate but too slow for the whole corpus.
* **How is a number checked?** After normalising dollars, commas, percentages and units, it must appear in a cited passage. It checks presence, not meaning, so it cannot catch an omission (the Walmart segments case).
* **Why is tau 0?** The retrieval score from rank fusion did not separate answerable from unanswerable questions, so the tuned threshold was 0 and we rely on the model's own refusal.
* **What is nDCG@10?** A score that rewards putting the most relevant passages first in the top 10; 1.0 is perfect.
* **What is pooling?** We merge the top 8 of every system for each question and judge only those passages, so judging effort stays small. It can favour the systems that built the pool (Zobel, 1998), so we ran bias checks.
* **What is kappa?** Agreement between raters beyond chance. Our two human raters have 0.39 (moderate); a language-model pass pair had weighted kappa 0.785.
* **Why a paired bootstrap?** With 20 test questions a classic test is unreliable; we resample questions 10,000 times to get confidence intervals and call a difference "clear" only if the interval excludes zero.
* **What are champion lists?** For each word we keep only its top r passages; scoring only those is faster (r = 50 was about three times faster at almost no loss).
* **How does the spelling module work?** A bigram shortlist (Jaccard overlap) finds candidate company names, then Levenshtein distance (at most two edits, one for short strings) confirms the match.
* **Why did you not build the vision track?** It was optional and gated; the encoder needs a GPU and the smoke test was not run. MaxSim scoring is implemented and tested. We say so in the report and in the Pages tab.
* **What are the main weaknesses?** 20 test questions, relevance labels made by a language model, two raters from the team (one checked 48 of 95 citations), easy unanswerable questions, and company names with "'s" skip the filter.
* **What did you learn that surprised you?** BM25 was not ahead of dense retrieval here; PageRank added nothing over counting mentions; the retrieval score is a poor confidence signal.

## 7. Where things are

* Report: `docs/report/TradeMesh_Report.pdf` (short, 10 pages); `docs/report/TradeMesh_Report_Detailed.pdf` (every table and ablation, good for the viva).
* Demo script with fixes if something breaks: `docs/demo_script.md`.
* Every requested item and where it is: `docs/report/REQUIREMENTS_TRACEABILITY.md`.
* Repository: https://github.com/Deltasthicc/IRFinalProj
