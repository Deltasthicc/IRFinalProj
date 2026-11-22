# Role B: ranking, evaluation, report sections

This folder holds the source text of the ranking and retrieval-evaluation sections of the report. The detailed
report, `docs/report/TradeMesh_Report_Detailed.pdf`, is assembled from these files and the parts in `docs/report/parts/` by
`python docs/report/build_report.py`; the short report `docs/report/TradeMesh_Report.pdf` is built by `build_concise.py` from
`docs/report/concise/paper.md` and the result files. Edit the sources, not the PDFs. Every number in these files comes from `eval/out/*.json`
(the locked test-split run) and can be regenerated with `python eval/fill_report_results.py`.

## What is here

| File | Content | Section of the report |
|---|---|---|
| `09_3_to_9_5_ranking_methods.md` | BM25, dense, hybrid, cross-encoder, pseudo-relevance feedback | V-C to V-E |
| `09_10_evaluation_protocol.md` | collection, pooling, metrics, significance, pool-bias checks | V-J |
| `10_retrieval_results.md` | judgments and agreement, main comparison, by-type results, hypotheses, ablations, speed, pool bias, retrieval vs answer quality | VI-B to VI-J |
| `10_paper_comparison.md` | observed ordering against BEIR, DPR, RRF, BERT reranking, FinanceBench and ColPali | VI-K |
| `00_abstract_introduction.md` | an earlier abstract and introduction, superseded by `docs/report/parts/02_abstract.md` and `03_introduction.md` | not used |
| `references.md` | the verified bibliographic entries, with how each was checked | feeds the reference list |
| `audit/` | the two-pass model judgments and their agreement | cited in VI-B |

## How the results were produced

Code: `src/trademesh/ranking/` (BM25, dense, fusion, rerank, PRF), `src/trademesh/evaluation/` (metrics, qrels, pooling, bootstrap,
kappa, bpref) and the scripts in `eval/`. The run order is the Makefile order: `make pools`, `make merge`, `make tune` (dev split only),
`make retrieval-dev` and `make ablation-dev`, then once after the feature freeze `make retrieval-test` followed by
`python eval/hypotheses.py`. The test run is locked after its first execution. Dry runs on the test fixture go to `eval/out/dryrun/`
and are refused unless `--allow-stub` is passed.

## Facts a reader should know

* The relevance judgments are model-assisted: a language model graded all 589 pooled pairs and a second pass graded 91 of them
  (weighted kappa 0.785 between the passes, `audit/`). No human graded a relevance pair.
* Role B was owned by Tanish Shitanshu (see "Team and roles" in the root README).
* Role A's systems (TF-IDF, champion lists, Boolean and phrase search) are part of the comparison under the names `tfidf`,
  `tfidf-champ-50`, `tfidf-champ-200` and `boolean`.
