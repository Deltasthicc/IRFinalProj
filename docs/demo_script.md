# TradeMesh live demo: script, set-up and fallbacks

Length: about 8 minutes, plus questions. Everything below uses buttons that exist in the app, so nothing needs typing except two short phrases.

## 1. Before the demo (do this the day before, and again 15 minutes before)

| # | Do | Why |
|---|---|---|
| 1 | Plug the laptop in, connect to the internet, close other heavy programs (Docker Desktop makes this machine slow) | the first page load builds the search indexes (about 15 s on BM25 only, about 50 s with the ML install, longer if the machine is busy); fonts come from Google Fonts |
| 2 | `.env` contains `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...` (see `.env.example`) | live answers; the key stays out of Git |
| 3 | **Install the best retrieval setup once:** `.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt` (about 1 GB; two small models download on first use) | the reported best system is hybrid retrieval plus the cross-encoder reranker, and the committed demo answers were produced with it. Without it the app falls back to BM25 and some demo answers change (see section 5) |
| 3b | **First start after the ML install computes the dense embeddings for all 11,964 passages and saves them to `cache/emb_*.npy`** (not in Git; later starts reuse them). Do this once on the demo laptop, well before presenting; on a new machine the very first start is slow | |
| 4 | Check the cache: `$env:LLM_OFFLINE="1"; .\.venv\Scripts\python.exe eval/demo_warm.py` then `Remove-Item Env:LLM_OFFLINE` | every demo question should be answered from the cache. If one stops with "not in the answer cache", run the same command without `LLM_OFFLINE` once so it is cached |
| 5 | Close anything on port 8501 (for example the rating tool), then start the app: `.\.venv\Scripts\python.exe -m streamlit run src/trademesh/app/streamlit_app.py` and open http://localhost:8501 | start it at least 5 minutes before you present; it needs about 50 s to load once the laptop is warm, but a cold first start (after a restart or the first launch of the day) took about 4 minutes on this machine. Open the page once and wait for the tabs to appear before the audience arrives |
| 6 | In the sidebar check: retriever **hybrid**, reranker **ticked**, tau 0.00, lambda 0.00 | these are the reported settings |
| 7 | Do a dry run of scenes 1 to 4 | the first live call after start can take a few seconds |
| 8 | **Record a backup:** screen-record one full run (Windows: Win+Alt+R). Keep the video on the desktop | if anything fails you can narrate over it |
| 9 | Docker fallback: `docker compose up --build` (full setup: `$env:INSTALL_ML="1"; docker compose up --build` in PowerShell) then http://localhost:8501 | tested from a clean checkout; without a key it shows a "no LLM provider" notice |

Offline demo: `LLM_OFFLINE=1` makes the app use only cached answers and never call the network. The four example buttons are exactly the cached questions.

## 2. Run of show

The page layout, top to bottom: hero (two buttons), company strip, status chips, tab bar (Ask, Search, Graph, Pages, Evaluation, System).

| Time | Scene | What you do | What the audience sees | What you say |
|---|---|---|---|---|
| 0:00 | **Hook** | Nothing yet; show the home screen | 11,964 passages, 20 companies, LLM and retriever status chips | "TradeMesh answers questions over the latest 10-K filings of 20 companies. Every claim carries a citation, a checker verifies the numbers, and it says so when the evidence is thin. The search core is written from scratch." |
| 0:30 | **1. A misspelt company** | Click **Ask with a typo** (question: "How does Telsa describe the risk of depending on its Chief Executive Officer?") | A red "Corrected Telsa >>> Tesla, Inc." bar; the pipeline strip (Entity: TSLA / 1 fixed); an answer about Elon Musk with a small `P1` tag; trust panel all Pass; the cited evidence card highlighted | "Spelling correction is our own bigram shortlist plus Levenshtein distance, no library. It resolved 39 of 40 misspelt names. The tag P1 is the passage that backs the claim; open it." Hover the pipeline slices to show what each step does. |
| 1:30 | **2. Numbers that are checked** | Ask tab, click **Figure / segments** ("What was Microsoft's revenue by segment in the latest fiscal year?") | An answer with dollar figures per segment; trust panel "Numbers grounded: Pass" with each figure marked "in P1" | "The verifier checks every number against the cited passage. A figure the model invented would show red here." |
| 2:15 | **2b. Same question, closed-book** | Switch on the **Closed-book** toggle | An answer with older fiscal-year figures, no citations, a red "unverified" panel | "Same model, no filings: fluent but out of date and uncheckable. In our blind human rating by two of us, retrieval plus reranking scored 1.20 out of 2 against 0.575 closed-book." Switch the toggle off. |
| 3:00 | **3. Compare two companies** | Click **Compare two companies** | Evidence split into an MSFT column and an AMZN column; one answer citing both | "Naming two companies makes one search per company and merges the results." |
| 3:45 | **4. Saying "I don't know"** | Click **Trap / future year** ("What was Apple's net sales in fiscal year 2031?") | A red "Insufficient evidence" panel; the pipeline strip shows Decision: Abstain | "It declines instead of guessing. The model's own verdict separated all 6 unanswerable test questions from 20 answerable ones. Be honest: our unanswerable questions are easy, and a score threshold did not work, so we use the model's verdict." |
| 4:30 | **5. The search engine** | Search tab: type `supplier concentration`; choose two or three retrievers | Side-by-side result columns, a "hits by 10-K item" bar chart, highlighted snippets, a "Why this score" bar chart per term | "These are our BM25, TF-IDF and hybrid retrievers side by side. 'Why this score' shows each query term's contribution." |
| 5:15 | **6. The evidence** | Evaluation tab | Retrieval: bars, hybrid + rerank 0.828 nDCG@10 against 0.699 for BM25. RAG answers: the human-rated panel first | "Retrieval with reranking beats BM25 by 0.13 nDCG@10. For answers, two of us each rated 60 answers blind: best-RAG is reliably better than closed-book and than BM25-RAG. Limits up front: both raters are on the project and stricter than the model-assisted pass, one of them checked only half the citations, and one reference (T10) was corrected after the first scoring, which we report openly." |
| 6:15 | **7. The company graph** | Graph tab | PageRank bars (AAPL first), a node graph | "Who mentions whom. PageRank correlates 0.93 with plain in-degree, so it adds little here." |
| 6:45 | **8. System and the typo lab** | System tab; type `Micorsoft` in the typo lab | Readiness checklist; four headline tiles; "Micorsoft >>> Microsoft Corporation (MSFT), edits 2" | "Readiness is honest: the page-image (vision) track was scoped out and the Pages tab says so. Try the spelling module yourself." (Optionally type `MSTF` to show the known miss: a swapped-letter ticker costs two edits.) |
| 7:30 | **Close** | Home | | "Everything runs from Docker or pip, the results regenerate from scripts, and every number in the report comes from a file in the repository." |

## 3. What to say if asked

| Question | Short answer |
|---|---|
| What is built from scratch? | Positional index, Boolean/phrase/wildcard, TF-IDF, BM25 (k1 = 0.6, b = 0.9 tuned on dev), spelling correction (bigram Jaccard shortlist, Levenshtein), reciprocal rank fusion, the verifier, metrics, pooling, bootstrap. Dense encoder and cross-encoder are pre-trained. |
| How is a number checked? | Normalised ($, commas, %, million/billion; "(in millions)" tables honoured) and matched exactly against the cited passages. A rounded figure is flagged, which is strict by design. |
| Why is tau 0? | The retrieval score comes from rank fusion and does not separate answerable from unanswerable questions, so tau tuned on dev came out as 0. Abstention is the model's own verdict. |
| How were answers judged? | Blind, 0/1/2 against a reference written from the filings. First by a language model (two passes), then by two humans, whose mean is the headline. Each human agrees with the model on only about half the answers and is stricter; the two humans agree exactly on 41 of 60 answers (kappa 0.39). |
| What changed after the first human scoring? | Question T10: the first reference said "no answer in the corpus", but Amazon's and Microsoft's 10-Ks do answer it. Corrected after the first scoring; both results are reported (in Shashwat's ratings alone, best-RAG minus BM25-RAG went from +0.25, not established, to +0.40; with both raters it is +0.30, interval +0.125 to +0.50). Ruchir's independent T10 ratings match the corrected ones. |
| Biggest limitations? | Two human raters, both on the project; model-made relevance judgments; 20 test questions; one small generator; easy unanswerable questions; possessive company names ("Microsoft's") are not recognised, so three company-naming needs (R06, R07, R09) ran without the company filter; vision track not built. |
| Why does the page say "companies: none" for "Microsoft's ..."? | Known defect (see limitations): the possessive form occurs in the filings' vocabulary, so it is not treated as a misspelling. It does not break retrieval, it just skips the company filter. |
| Is the vision track done? | No. Only the MaxSim scoring is implemented and tested; the page-image encoder needs a GPU. It is stated as gated in the report and in the app. |
| Is the human rating independent? | No: two raters, both on the project. They agree on 41 of 60 answers (kappa 0.39). BM25-RAG over closed-book depends on which rater: +0.55 for Ruchir, +0.10 for Shashwat. |

## 4. Things to avoid typing live

* **Possessive company names** ("Apple's", "Microsoft's"): the company filter is skipped for these (known defect). Use the example buttons, or write "Apple supply chain risk".
* **New questions with no cached answer when offline**: the app shows "not in the answer cache" (safe, but breaks the flow). With internet and a key a new question makes one live call, usually a few seconds.
* Questions that need several years of data (out of scope) or tables (weaker: in R09 both RAG answers omitted a segment).

## 5. If something goes wrong

| Problem | Fix |
|---|---|
| Page stays blank or says "Building the search indexes" | Wait up to a minute (machine busy); never reload during it. |
| An example answers "Insufficient evidence" when it should not | You are on BM25 without the ML install (the numeric question declines there). Install `requirements-ml.txt`, restart, or switch scenes and use the recording for that one. |
| A live answer takes a long time (seen: over a minute) | The client waits out Gemini's per-minute limits on purpose. Keep talking over the pipeline strip, or use cached examples (`LLM_OFFLINE=1`) so nothing waits. |
| "LLM call failed" or a quota message | The free tier has per-minute and per-day limits. Switch to `LLM_OFFLINE=1` (cached answers only) or the recording. |
| Port 8501 in use | Run with `--server.port=8502` and open that address. |
| Anything else | Play the backup video and narrate. |

## 6. Setup facts to quote

* Corpus: latest 10-K of 20 companies, 11,964 passages (text and table chunks). Retrievers: TF-IDF (plain and two champion-list variants), Boolean, BM25, BM25 with pseudo-relevance feedback, plus dense, hybrid and reranked ones with the ML install.
* Reported results: nDCG@10 hybrid + rerank 0.828 against 0.699 BM25 (20 test needs); human-rated answer quality (mean of two raters) 1.20 / 0.90 / 0.575 for best-RAG / BM25-RAG / closed-book; entity accuracy 39 of 40; 6 of 6 unanswerable questions declined.
* Tests: `python -m pytest -q` (137 tests, offline). Packaging: `docker compose up`.

## 7. Verified on 19 Nov 2026

With `requirements-ml.txt` installed and the app started with `LLM_OFFLINE=1` (no network calls allowed):

* `eval/demo_warm.py` answered all 9 demo questions from the cache (0 live calls), including the closed-book twins.
* In the app, with the sidebar on **hybrid** and the reranker ticked, scenes 1 to 4 and 2b ran with **0 live / 6 from cache**: the typo is corrected to Tesla (cited P1), the Microsoft segment figures are cited to P5 with "Numbers grounded: Pass", closed-book shows the stale FY2024 figures with the red "unverified" panel, the comparison shows MSFT and AMZN columns with three citations, and the future-year question abstains.
* App start-up with the ML install took about 50 s; the full test suite (137 tests) passes with it installed.

