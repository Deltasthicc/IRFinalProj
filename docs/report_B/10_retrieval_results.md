# 10. Retrieval results (Role B)

All numbers below are read from the locked test-split files (`eval/out/*_test.json`, `pool_bias.json`, `cost_table.json`). The test split was run once, after the freeze; the lock files are `eval/out/.testrun_*`. Regenerate this page with `python eval/fill_report_results.py`.

## 10.1 Judgments and agreement

**The relevance judgments are LLM-assisted. No human graded any pair.** Every pooled pair was graded 0/1/2 by an LLM (Claude), blind to the retrieving system; a second, independent LLM pass graded a stratified subset. A planned human spot-check was not done, so the agreement figure below measures consistency between two LLM passes, not agreement with a human.

| Item | Value | Source |
|---|---|---|
| Pairs judged by judge AI | 589 (grade 2: 217, grade 1: 146, grade 0: 226) | `pool_sheet_AI.csv` |
| Pairs judged by second pass AI2 | 91, spread over all 30 needs | `pool_sheet_AI2.csv` |
| Exact agreement, before reconciliation | 77 of 91 | `audit/ai_pass_agreement.csv` |
| Weighted kappa (linear weights), before reconciliation | 0.785 | `trademesh.evaluation.qrels.weighted_kappa` |
| Same, after correcting a truncation error | 80 of 91, kappa 0.854 (not independent; do not headline) | same |
| Pool size per need (mean, range) | 19.6, 11 to 30 | `pool_sheet_AI.csv` |

Where two judges graded the same pair, the consolidated grade is the lower of the two (`qrels.consolidate`). Both passes first viewed only the start of long passages; a few relevant passages were graded 0 and were corrected after reading them in full (see `audit/README.md`).

## 10.2 Main comparison (RQ1), 20 test needs

| System | P@5 | P@10 | R@10 | F1@10 | MAP | MRR | nDCG@10 | bpref | judged@10 |
|---|---|---|---|---|---|---|---|---|---|
| tfidf | 0.570 | 0.530 | 0.395 | 0.428 | 0.455 | 0.600 | 0.560 | 0.431 | 0.905 |
| tfidf-champ-50 | 0.570 | 0.530 | 0.395 | 0.428 | 0.450 | 0.600 | 0.560 | 0.423 | 0.905 |
| tfidf-champ-200 | 0.570 | 0.530 | 0.395 | 0.428 | 0.455 | 0.600 | 0.560 | 0.431 | 0.905 |
| boolean | 0.360 | 0.285 | 0.211 | 0.231 | 0.220 | 0.600 | 0.364 | 0.224 | 0.325 |
| bm25 | 0.700 | 0.615 | 0.512 | 0.519 | 0.574 | 0.821 | 0.699 | 0.577 | 0.920 |
| bm25+prf | 0.690 | 0.590 | 0.502 | 0.505 | 0.556 | 0.800 | 0.679 | 0.571 | 0.890 |
| dense | 0.780 | 0.640 | 0.606 | 0.566 | 0.606 | 0.908 | 0.782 | 0.618 | 0.875 |
| hybrid | 0.760 | 0.670 | 0.604 | 0.584 | 0.678 | 0.858 | 0.784 | 0.650 | 0.960 |
| bm25+rerank | 0.750 | 0.655 | 0.575 | 0.560 | 0.632 | 0.887 | 0.776 | 0.661 | 0.865 |
| hybrid+rerank | 0.770 | 0.705 | 0.658 | 0.621 | 0.693 | 0.887 | 0.828 | 0.690 | 0.930 |

Paired bootstrap against bm25 (95 percent interval of the mean per-need difference; W/L/T = needs won, lost, tied):

| System | Metric | Difference | 95% CI | W/L/T | p |
|---|---|---|---|---|---|
| tfidf | AP | -0.119 | [-0.198, -0.054] | 3/16/1 | 0.003 |
| tfidf | nDCG@10 | -0.139 | [-0.223, -0.064] | 5/13/2 | 0.001 |
| tfidf-champ-50 | AP | -0.124 | [-0.202, -0.059] | 2/17/1 | 0.002 |
| tfidf-champ-50 | nDCG@10 | -0.139 | [-0.223, -0.064] | 5/13/2 | 0.001 |
| tfidf-champ-200 | AP | -0.119 | [-0.198, -0.054] | 3/16/1 | 0.003 |
| tfidf-champ-200 | nDCG@10 | -0.139 | [-0.223, -0.064] | 5/13/2 | 0.001 |
| boolean | AP | -0.354 | [-0.470, -0.239] | 2/18/0 | 0.000 |
| boolean | nDCG@10 | -0.336 | [-0.476, -0.197] | 3/15/2 | 0.000 |
| bm25+prf | AP | -0.018 | [-0.053, +0.009] | 8/11/1 | 0.266 |
| bm25+prf | nDCG@10 | -0.020 | [-0.051, +0.010] | 5/10/5 | 0.198 |
| dense | AP | +0.032 | [-0.067, +0.142] | 9/11/0 | 0.553 |
| dense | nDCG@10 | +0.083 | [+0.002, +0.177] | 13/7/0 | 0.064 |
| hybrid | AP | +0.104 | [+0.029, +0.174] | 14/6/0 | 0.004 |
| hybrid | nDCG@10 | +0.085 | [+0.019, +0.155] | 13/4/3 | 0.015 |
| bm25+rerank | AP | +0.059 | [+0.006, +0.133] | 12/7/1 | 0.064 |
| bm25+rerank | nDCG@10 | +0.076 | [+0.012, +0.149] | 12/3/5 | 0.032 |
| hybrid+rerank | AP | +0.119 | [+0.032, +0.211] | 16/4/0 | 0.009 |
| hybrid+rerank | nDCG@10 | +0.129 | [+0.052, +0.213] | 13/3/4 | 0.002 |

Boolean's judged@10 is 0.325: much of its top 10 was never judged, so its scores are pessimistic; read bpref alongside. The 11-point interpolated precision-recall data for the figure are in `pr_11pt` of `retrieval_test.json`.

## 10.3 By need type (nDCG@10)

Test needs per type: entity 7, topical 7, terminology 6. Subsets this small make per-type differences fragile.

| System | entity | topical | terminology |
|---|---|---|---|
| tfidf | 0.380 | 0.727 | 0.576 |
| tfidf-champ-50 | 0.380 | 0.727 | 0.576 |
| tfidf-champ-200 | 0.380 | 0.727 | 0.576 |
| boolean | 0.122 | 0.341 | 0.672 |
| bm25 | 0.580 | 0.778 | 0.747 |
| bm25+prf | 0.574 | 0.723 | 0.750 |
| dense | 0.671 | 0.846 | 0.839 |
| hybrid | 0.697 | 0.819 | 0.844 |
| bm25+rerank | 0.664 | 0.878 | 0.787 |
| hybrid+rerank | 0.713 | 0.923 | 0.852 |

## 10.4 Hypotheses H1 to H4, H6

| ID | Comparison | Difference | 95% CI | W/L/T | Verdict |
|---|---|---|---|---|---|
| H1 | bm25 vs tfidf (all needs) | +0.139 | [+0.064, +0.223] | 13/5/2 | A>B SUPPORTED |
| H2a | bm25 vs dense, entity needs (hoped: tied or better) | -0.092 | [-0.208, +0.019] | 2/5/0 | INCONCLUSIVE |
| H2a | bm25 vs dense, terminology needs (hoped: tied or better) | -0.091 | [-0.324, +0.079] | 2/4/0 | INCONCLUSIVE |
| H2b | dense vs bm25, topical needs (hoped: dense better) | +0.068 | [-0.055, +0.201] | 4/3/0 | INCONCLUSIVE |
| H3a | hybrid vs bm25 | +0.085 | [+0.019, +0.155] | 13/4/3 | A>B SUPPORTED |
| H3b | hybrid vs dense | +0.002 | [-0.059, +0.051] | 12/6/2 | TIED (inconclusive) |
| H4 | hybrid+rerank vs hybrid | +0.044 | [-0.012, +0.113] | 9/4/7 | INCONCLUSIVE |
| H6a | bm25+prf vs bm25, terminology needs (hoped: better) | +0.002 | [-0.069, +0.070] | 2/2/2 | TIED (inconclusive) |
| H6b | bm25+prf vs bm25, entity needs (hoped: not better) | -0.006 | [-0.035, +0.028] | 2/3/2 | TIED (inconclusive) |

Rows whose interval contains zero are inconclusive, not trends.

## 10.5 Ablations (RQ2, H5)

The default is bm25 with stop-word removal, no stemming, numbers kept and no metadata prefix. Fusion rows are hybrid variants; PRF rows are bm25 + PRF.

| Group | Configuration | MAP | nDCG@10 | Diff (nDCG@10 vs default) | 95% CI | p |
|---|---|---|---|---|---|---|
| default | stop+nostem+num+nometa | 0.574 | 0.699 | +0.000 | [+0.000, +0.000] | 1.000 |
| A prep | stop+nostem+num+meta | 0.623 | 0.759 | +0.060 | [-0.020, +0.152] | 0.169 |
| A prep | stop+nostem+nonum+nometa | 0.579 | 0.708 | +0.009 | [-0.009, +0.030] | 0.364 |
| A prep | stop+nostem+nonum+meta | 0.633 | 0.783 | +0.083 | [+0.011, +0.170] | 0.042 |
| A prep | stop+stem+num+nometa | 0.525 | 0.633 | -0.066 | [-0.129, -0.011] | 0.025 |
| A prep | stop+stem+num+meta | 0.577 | 0.707 | +0.008 | [-0.090, +0.110] | 0.875 |
| A prep | stop+stem+nonum+nometa | 0.528 | 0.629 | -0.070 | [-0.140, -0.010] | 0.030 |
| A prep | stop+stem+nonum+meta | 0.588 | 0.717 | +0.017 | [-0.082, +0.120] | 0.736 |
| A prep | nostop+nostem+num+nometa | 0.575 | 0.713 | +0.013 | [-0.007, +0.036] | 0.240 |
| A prep | nostop+nostem+num+meta | 0.622 | 0.770 | +0.071 | [-0.005, +0.160] | 0.094 |
| A prep | nostop+nostem+nonum+nometa | 0.571 | 0.702 | +0.003 | [-0.027, +0.032] | 0.857 |
| A prep | nostop+nostem+nonum+meta | 0.617 | 0.757 | +0.058 | [-0.010, +0.135] | 0.118 |
| A prep | nostop+stem+num+nometa | 0.505 | 0.613 | -0.086 | [-0.180, -0.007] | 0.047 |
| A prep | nostop+stem+num+meta | 0.560 | 0.678 | -0.021 | [-0.138, +0.093] | 0.720 |
| A prep | nostop+stem+nonum+nometa | 0.509 | 0.623 | -0.076 | [-0.174, +0.006] | 0.092 |
| A prep | nostop+stem+nonum+meta | 0.563 | 0.685 | -0.014 | [-0.134, +0.104] | 0.814 |
| B struct | drop table chunks | 0.565 | 0.699 | -0.001 | [-0.002, +0.000] | 0.632 |
| B struct | entity filter off | 0.556 | 0.689 | -0.010 | [-0.023, +0.000] | 0.103 |
| C fusion | rrf k=10 | 0.746 | 0.820 | +0.121 | [+0.062, +0.189] | 0.001 |
| C fusion | rrf k=60 | 0.678 | 0.784 | +0.085 | [+0.019, +0.155] | 0.015 |
| C fusion | rrf k=100 | 0.671 | 0.777 | +0.078 | [+0.007, +0.151] | 0.036 |
| C fusion | combsum | 0.723 | 0.808 | +0.109 | [+0.033, +0.193] | 0.008 |
| C fusion | combmnz | 0.694 | 0.786 | +0.087 | [+0.015, +0.170] | 0.026 |
| D prf | k=3, m=5 | 0.569 | 0.699 | -0.001 | [-0.035, +0.035] | 0.970 |
| D prf | k=3, m=10 | 0.570 | 0.703 | +0.004 | [-0.033, +0.043] | 0.834 |
| D prf | k=3, m=20 | 0.561 | 0.685 | -0.015 | [-0.062, +0.034] | 0.562 |
| D prf | k=5, m=5 | 0.551 | 0.672 | -0.027 | [-0.054, -0.001] | 0.045 |
| D prf | k=5, m=10 | 0.556 | 0.679 | -0.020 | [-0.051, +0.010] | 0.198 |
| D prf | k=5, m=20 | 0.549 | 0.683 | -0.017 | [-0.051, +0.020] | 0.356 |
| D prf | k=10, m=5 | 0.561 | 0.689 | -0.011 | [-0.034, +0.010] | 0.349 |
| D prf | k=10, m=10 | 0.551 | 0.676 | -0.023 | [-0.048, -0.001] | 0.059 |
| D prf | k=10, m=20 | 0.551 | 0.677 | -0.023 | [-0.046, +0.002] | 0.067 |

H5 (table chunks and numeric tokens matter for table needs) was run over all 20 needs only. The five-table-need subset analysis was not computed. TODO(run): per-need ablation values are not stored, and the test run is locked.

## 10.6 Speed and cost (H7)

Index: 11964 passages, BM25 build 0.64 s. Latency per query (ms) and quality:

| System | p50 ms | p95 ms | MAP | nDCG@10 |
|---|---|---|---|---|
| tfidf | 0.24 | 0.55 | 0.455 | 0.560 |
| tfidf-champ-50 | 0.08 | 0.12 | 0.450 | 0.560 |
| tfidf-champ-200 | 0.23 | 0.31 | 0.455 | 0.560 |
| boolean | 0.06 | 0.19 | 0.220 | 0.364 |
| bm25 | 0.06 | 0.09 | 0.574 | 0.699 |
| bm25+prf | 0.45 | 0.52 | 0.556 | 0.679 |
| dense | 5.29 | 6.80 | 0.606 | 0.782 |
| hybrid | 5.47 | 7.53 | 0.678 | 0.784 |

Reranking the top 30 with the cross-encoder adds p50 106.3 ms and p95 211.0 ms per query. This figure was printed by `eval/cost_table.py` but is not stored in `cost_table.json`; hybrid+rerank is therefore about the hybrid latency plus this. TODO(run): store the rerank latency in the JSON on the next cost run (a cost-only rerun does not touch the locked retrieval results).

## 10.7 Pool-bias checks

| System | MAP | MAP without its own unique finds | Change | Kendall tau of system order |
|---|---|---|---|---|
| tfidf | 0.455 | 0.455 | +0.000 | 0.96 |
| tfidf-champ-50 | 0.431 | 0.429 | -0.003 | 0.82 |
| tfidf-champ-200 | 0.455 | 0.455 | +0.000 | 0.96 |
| boolean | 0.216 | 0.202 | -0.014 | 0.82 |
| bm25 | 0.604 | 0.601 | -0.003 | 0.82 |
| bm25+prf | 0.605 | 0.600 | -0.005 | 0.75 |
| dense | 0.604 | 0.507 | -0.097 | 0.89 |
| hybrid | 0.683 | 0.671 | -0.012 | 0.96 |

The two reranked systems entered the pool as well but are not part of this leave-one-system-out check.

## 10.8 Does better retrieval give better answers? (RQ3, second half)

TODO(run): `python eval/retrieval_vs_answer.py` needs Role C's blind answer ratings. With about 40 pairs the interval is wide; the report states n.

## 10.9 Reading the results

Findings, each with its interval; where the interval contains zero it is called inconclusive.

- **Hybrid with reranking is the strongest system.** nDCG@10 0.828 against 0.699 for bm25, a difference of +0.129 (95% CI +0.052 to +0.213), winning 13 needs and losing 3.
- **H1 supported:** bm25 beats tf-idf by +0.139 nDCG@10 (CI +0.064 to +0.223). Champion lists did not change this.
- **H3a supported, H3b tied:** hybrid beats bm25 (+0.085, CI +0.019 to +0.155) but is indistinguishable from dense alone (+0.002, CI -0.059 to +0.051).
- **H4 inconclusive:** the reranker's gain over hybrid is +0.044 (CI -0.012 to +0.113). The gain of hybrid+rerank over bm25 is significant; over hybrid it is not.
- **H2 and H6 inconclusive.** The per-type subsets hold 6 or 7 needs each; no conclusion about dense versus lexical retrieval by need type is supported.
- **Pseudo-relevance feedback did not help** on test in any of the 9 settings tried. On the 10 dev needs it looked slightly positive; the test result is the one that counts.
- **Stemming was not adopted.** With no metadata prefix, stemming lowers nDCG@10 by 0.066 to 0.086 on test (the interval excludes zero in three of four cases) and every stemmed dev configuration also scored below the default. With the metadata prefix stemming makes no measurable difference. The choice was made on dev before the test run.
- **Table chunks and the entity filter showed no measurable effect over all 20 test needs** (nDCG@10 differences -0.001 and -0.010). On dev they looked mildly helpful; the test data do not confirm it. The five-table-need subset was not analysed (see 10.5).
- **Pooling limits.** The pool was built from these same systems and judged by an LLM. Removing a system's unique finds changes MAP by at most 0.097 (dense, the most pool-dependent) and the system ordering is stable (tau 0.75 to 0.96), but the absolute scores should not be compared with other collections.
- **Judgments are LLM-assisted with no human validation.** All conclusions are conditional on that.
