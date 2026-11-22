# Requirements traceability: every item asked for, and where it is (or why it is not)

Sources: the instructor's email (report sections, reference format, ZIP contents), the approved plan `TradeMesh_v2_Proposal.pdf` (Core / Plan / Gate / Stretch / Skip tiers, hypotheses H1 to H11, rubric mapping) and our own chat decisions. "Report" means `docs/report/TradeMesh_Report.pdf`, a short IEEE-style two-column paper (10 pages, 16 figures, 12 tables, 25 references) written in plain language. It follows the instructor's section order with IEEE numbering (the Abstract and References are unnumbered; Introduction = I, Related work = II, Objective = III, Proposed model = IV, Methodology = V, Experiments and results = VI, Conclusions and limitations = VII). The long version with every table, ablation and pool-bias check is `docs/report/TradeMesh_Report_Detailed.pdf` (23 pages, also in the ZIP under `extra/`); the decimal section numbers below (for example 7.6) refer to that detailed version, where section n is Roman numeral n minus 2 and subsection m is letter m (7.6 = V-F). Citations in the text are author-year and the reference list uses the instructor's required format, because her email prescribes it.

Status: **Done** = built and in the report. **Done, changed** = done but not exactly as planned (the change is stated in the report). **Not done, justified** = deliberately not done, with the reason.

## A. The instructor's report checklist

| # | Required | Status | Where |
|---|---|---|---|
| 1 | Title | Done | Title block |
| 2 | Abstract | Done (152 words in the short report; the plan said about 150) | Section 2 |
| 3 | Introduction: background, importance, motivation, contributions | Done | 3.1 to 3.6 |
| 4 | Literature review: at least 5 to 7 papers, plus what is new | Done: 25 references (24 distinct papers and one book chapter) in six themes, novelty stated modestly | Section 4, 4.7 |
| 5 | Objective | Done: aim, 7 objectives, RQ1 to RQ5, H1 to H11 with outcomes | Section 5 |
| 6 | Proposed model with diagram | Done: Figure 1, six-tab application | Section 6 |
| 7 | Methodology: algorithms and formulas | Done: 7.1 to 7.10 (index, TF-IDF, BM25, dense/hybrid/rerank, PRF, query processing, graph, RAG, vision, protocol) | Section 7 |
| 8 | Experiments and results: datasets, graphs/tables, precision, recall, accuracy, F1, discussion, comparison with at least one existing paper | Done: 8.1 datasets; P/R/F1@5,10, MAP, MRR, nDCG, bpref, PR curves, ablations, speed, pool-bias checks; entity accuracy; abstention P/R/F1/accuracy; comparison with BEIR, DPR, RRF, BERT reranking, FinanceBench, ColPali (8.11) | Section 8, 9 figures |
| 9 | Conclusions and limitations | Done | Section 9 |
| 10 | References in the instructor's format ("Author, Initial., Year, Month. Title. In Proceedings ... (pp. x-y).") | Done: 25 references, alphabetical, instructor format; all checked against publisher, ACL Anthology, arXiv or two databases | Section 10 |
| 11 | Submission: ZIP named CSD358_GroupMember1_Member2_Member3, with report PDF, GitHub link, source code and dataset | Done: `CSD358_ShashwatRajan_TanishShitanshu_RuchirSharma.zip` (name chosen by the team; names and roll numbers are in the report title block) | ZIP |

## B. The approved plan: Core items (never cut)

| Item | Status | Evidence |
|---|---|---|
| Polite EDGAR crawler (User-Agent, rate limit, cache, robots rules) | Done, with one gap: declared User-Agent, cap of 8 requests/s (limit 10), disk cache. It does not parse robots.txt; the SEC's published access rules (User-Agent and 10 requests/s) are what governs EDGAR and we follow those | `ingest/edgar.py` |
| Parser and passage chunker (Items, overlap, metadata) | Done, without overlap: passages follow 10-K Items and tables, each with ticker, year, Item, page; no overlapping windows (not ablated) | `ingest/parse.py`, `chunk.py` |
| Positional inverted index with skip pointers | Done (skips are implicit, spacing sqrt(L)) | 7.1, tests |
| Boolean, phrase and wildcard (permuterm or k-gram) | Done: permuterm for one star, 2-gram for several | 7.1, tests |
| TF-IDF cosine with heap top-K; BM25 | Done | 7.2, 7.3 |
| Test collection, pooling, metrics (P/R/F1, MAP, nDCG, MRR, PR curve, kappa) | Done, changed: judgments by LLM, not three humans (see D) | 7.10, 8.1, 8.2 |
| Snippets with highlighted terms; Search tab | Done | app, 6 |
| Spelling-tolerant company names (k-gram shortlist, edit distance) and entity filter | Done: 39 of 40 | 7.6, 8.12 |

## C. Plan items (intended to ship)

| Item | Status | Evidence |
|---|---|---|
| Champion lists, speed vs quality | Done (r = 50 gives about 3x speed-up at 0.005 MAP; r = 200 none; fallback count not stored) | 7.2, 8.7 |
| Section-type static quality score | Done; lambda tuned on dev only (best +0.019, no demonstrated benefit); lambda = 0 in all test runs | 7.6, 8.12 |
| Preprocessing ablation grid (stop words, stemming, numbers, metadata) | Done: 16-way grid plus table, entity, fusion and PRF ablations | 8.6 |
| Pseudo-relevance feedback | Done (no measurable effect) | 7.5, 8.5 |
| Basic cited RAG on BM25 context | Done | 7.8, 8.12 |
| Dense retrieval and RRF hybrid | Done | 7.4 |
| Cross-encoder reranker on top-30 | Done | 7.4 |
| Verifier, abstention, RAG study (closed-book vs RAG, citation precision, number grounding) | Done, changed: abstention uses the model's verdict because tau tuned to 0; ratings by two humans (Shashwat and Ruchir) plus two model passes; Ruchir checked only 48 of the 95 citations and nobody did the optional retrieval-judging step | 7.8, 8.12 |
| Table-aware chunks | Done (1,052 table chunks) | 7.6, 8.1 |
| Compare-companies mode | Done | 7.6, app |
| Docker Compose packaging | Done, tested from a clean checkout | README, demo script |
| Company mention graph, PageRank, HITS, network view | Done: 20 nodes, 14 edges; PageRank vs in-degree Spearman 0.93 | 7.7 |

## D. Where we changed the plan, and said so in the report

| Plan | What we did | Why |
|---|---|---|
| All three members judge pooled relevance pairs (about 300 each) | All 589 pooled pairs graded by a language model blind to the system, 91 pairs graded twice; kappa 0.785 between the two passes; no human graded a pair | Time; stated in the abstract, 8.1, 8.2 and 9.2 as the main limitation of the retrieval results |
| Reference answers written during judging, ratings by two of us | Two project members (Shashwat Rajan and Ruchir Sharma) each rated the 60 answers blind against references they wrote for the 20 test questions; model-assisted ratings kept for comparison; one reference (T10, Shashwat's) corrected after the first scoring and disclosed | The plan said two of us; both did the ratings, but only one checked all citations. Results are reported for the mean of the two and for each rater, with the kappa between them (0.39) |
| Abstain when best rerank score is below tau | tau tuned on dev came out 0, so abstention is the model's INSUFFICIENT EVIDENCE verdict | The score rule did not separate the classes (precision 0.385 on test); disclosed that we saw the test numbers before choosing the verdict rule |
| BM25 defaults k1 = 1.2, b = 0.75, small grid on dev | Grid on dev gave k1 = 0.6, b = 0.9 (flat surface) | Reported with the whole grid |
| Stretch: single RAGAS cross-check, learned fusion, boilerplate flags, multimodal answers | Not done | See E |

## E. Gate and Stretch items: not done, with the justification

| Item | Tier | Why it is not needed |
|---|---|---|
| Vision track (ColPali-style page retrieval, 15 table/chart questions, H11, RQ5) | Gate (go/no-go 9 Nov) | The plan says that if the gate is not passed the track becomes a paragraph of future work and the other tiers are unaffected. The MaxSim scoring is implemented and tested; the encoder needs a GPU and a multi-gigabyte model, and the smoke test was never run. Section 7.9 and the app's Pages tab state this. It is the one planned research question the report cannot answer. |
| Learned fusion (logistic regression, leave-one-query-out) | Stretch ("only after the freeze, only if all else is green") | The plan itself rates it risky because 30 queries are too few to fit and validate weights; RRF and CombSUM/CombMNZ are already compared (8.6) |
| Near-duplicate boilerplate flags (shingling and Jaccard) | Stretch | Optional extra; nothing in the research questions depends on it |
| RAGAS cross-check | Stretch | The plan replaced RAGAS by deterministic checks and blind ratings (LLM-judge cost, opacity); it is cited in the literature review (Es et al. 2023) |
| Multimodal answers reading page images | Stretch | Depends on the vision track |
| Multi-region deployment, Prometheus/Grafana, MLflow, Locust, CI, Qdrant, fine-tuning, multi-turn chat | Skip | Deliberately removed in v2 (not IR; the latency result would be decided by injected delay) |

## F. Research questions and hypotheses

| | Answered? | Result |
|---|---|---|
| RQ1 retriever comparison | Yes | 8.3 to 8.5: hybrid + rerank 0.828 nDCG@10 best; hybrid tied with dense |
| RQ2 preprocessing, structure, champion lists | Yes | 8.6, 8.7 |
| RQ3 RAG vs closed-book, retrieval vs answer quality, abstention | Yes | 8.9, 8.12 |
| RQ4 graph vs in-degree | Yes | 7.7 |
| RQ5 vision vs text (gated) | No | 7.9 |
| H1 to H4, H6 | Tested with pre-declared rules | Section 5 table, 8.5 |
| H5 | Partly: all-needs effect only, table-need subset not computed (per-need values not stored, test run locked) | 8.6 |
| H7 | Consistent for r = 50, not significance-tested | 7.2 |
| H8, H9 | Tested; second half of H9 not tested (tau = 0) | 8.12 |
| H10 | Tested: not supported | 7.7 |
| H11 | Not tested | 7.9 |

## G. Rubric (Demo 10, Viva 5, Novelty 4, Report 6)

| Rubric | What exists |
|---|---|
| Demo | Six-tab app: cited Q&A with trust panel, closed-book toggle, compare mode, Boolean/phrase/wildcard and side-by-side rankers, spelling-corrected names, an unanswerable question that abstains, filing graph, Evaluation and System tabs. `docker compose up` or plain Python. Cached offline mode. `docs/demo_script.md` has the 8-minute run of show and fixes. Pages view says the vision track was not built |
| Viva | Every formula is in section 7; `docs/demo_script.md` section 3 has short answers; each member must still be able to explain the others' modules (the plan's mock-Viva rule) |
| Novelty | Section 4.7: domain test collection, controlled comparison, deterministic verifier with abstention and a negative result on score thresholds, pre-declared hypotheses with bootstrap, graph analysis. Text-versus-vision study not delivered |
| Report | 10 sections in the instructor's order, 16 figures (pipeline diagram, answer-checking flow, four app screenshots, plots), 12 tables, 25 references, comparison with published work |

## H. Why more filings or datasets were not added

The plan targeted 6,000 to 10,000 passages from the latest 10-K of 20 companies. The corpus has 11,964 passages and 1.46 million words, above the target. Nothing in RQ1 to RQ5 needs a larger corpus, and changing it now would invalidate the locked test run, the 680 relevance judgments, the 60 human ratings and the cached demo answers. A second fiscal year per company is listed as future work (the plan parks it as well).

## I. Checks run for this audit

* Full test suite: 137 passed (offline).
* Headline numbers in the report were re-read from `eval/out/*.json` and `data/graph/company_graph.json` (retrieval nDCG@10 and MAP for all ten systems, entity 39/40, graph correlations).
* Report assembled by `docs/report/build_report.py`, which stops if an expected string is missing; the PDF was inspected page by page.

## J. Reproducibility and wording of claims (from the final review)

* The paper names the answer model (Gemini, `gemini-3.5-flash-lite`, temperature 0), the encoder, the reranker, the package versions and the bootstrap seed (Table III). The exact version of the model that made the relevance labels was not recorded, and the paper says so. `requirements-ml.txt` pins sentence-transformers, torch and transformers to the versions used.
* The dataset used is the processed corpus (`dataset/data/passages.jsonl`, 11,964 passages, with `filings.json` listing each filing's accession number and URL), the questions, judgments, ratings and results. The original SEC filing files are public and are not bundled; `python -m trademesh.ingest.corpus` fetches them again.
* Claims are kept narrow in the abstract, section VI-F and the conclusions: 20 test questions, model-made relevance labels, two raters from the team (kappa 0.39), and an abstention result on easy questions that relies on the model's own verdict.
* `make_submission_zip.py` checks its own output (report PDF, link, README, code, dataset, both raters) and refuses to finish if a draft or old-name file is inside.
