# Draft figure captions

**fig_ndcg_systems.png.** Mean nDCG@10 of the ten retrieval systems over the 20 test questions, as horizontal bars sorted in descending order; the column on the right gives each system's paired bootstrap difference from BM25 with its 95% interval. No per-system error bars are drawn because `retrieval_test.json` stores only paired differences against BM25 (source: `eval/out/retrieval_test.json`, identical to `data/results/retrieval_test.json`).

**fig_pr_curves.png.** 11-point interpolated precision-recall curves (recall 0.0 to 1.0 in steps of 0.1) for TF-IDF, BM25, dense, hybrid and hybrid + rerank, each averaged over the 20 test questions. Source: the `pr_11pt` entry of `eval/out/retrieval_test.json`.

**fig_ndcg_by_type.png.** Mean nDCG@10 of BM25, hybrid and hybrid + rerank for each of the three need types (entity, topical, terminology), computed from the 20 test questions. Source: the `by_type` entry of `eval/out/retrieval_test.json`.

**fig_rag_ratings.png.** Mean answer rating (scale 0 to 2) for closed-book, BM25-RAG and best-RAG over the same 20 test questions, comparing the human ratings with the model-assisted ratings; the text box lists the human paired differences with 95% bootstrap intervals. Sources: `eval/out/rag_scores_human.json` (human ratings and paired differences) and `eval/out/rag_scores_test.json` (model-assisted ratings).

**fig_entity_accuracy.png.** Share of misspelled company names resolved to the correct ticker, by typo kind, over 40 misspelled-name queries (39 correct overall); the number above each bar is correct/total for that kind. Source: the `by_kind` entry of `eval/out/entity_eval.json`.

**fig_pagerank.png.** PageRank (bars, left axis) and in-degree (diamonds, right axis) of the 12 highest-ranked companies in the company mention graph, which has 20 companies and 14 edges; the title gives the Spearman correlations stored with the graph. Source: `data/graph/company_graph.json` (nodes and `correlations` entries).

**fig_abstention.png.** Precision, recall and F1 of the score-only abstention rule and the model-verdict abstention rule on the 26 test items, with the positive class being "should abstain"; true/false positive and negative counts are printed below the plot. Sources: `eval/out/abstention_scoreonly.json` and `eval/out/abstention_eval.json` (the `test` entries).

**fig_speed_quality.png.** Median (p50) query latency in milliseconds, with a bar extending to the p95 latency, against test-set nDCG@10 for plain TF-IDF and TF-IDF with champion lists of 200 and 50 passages per term, over an index of 11,964 passages; nDCG@10 is the mean over the 20 test questions. Sources: `eval/out/cost_table.json` (latency; the query set it was timed on is not recorded in the file) and `eval/out/retrieval_test.json` (nDCG@10).
