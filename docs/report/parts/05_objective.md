## 5. Objective

**Aim.** To build and evaluate a search and question-answering system over SEC 10-K filings in which the retrieval core is understood and implemented from the course material, the answers are checkable against the filing, and every claim about which method is better is backed by a pre-declared test with an interval, not by a single score.

**Objectives.**

1. Implement a complete retrieval stack for 10-K passages: positional index with Boolean, phrase and wildcard search, TF-IDF with champion lists, BM25, pseudo-relevance feedback, dense retrieval, rank fusion and cross-encoder reranking (sections 7.1 to 7.5).
2. Handle the way people actually ask: tolerate misspelt company names, filter and split by company, and keep tables searchable (section 7.6).
3. Produce answers that cite their passages, pass a deterministic check of citations and numbers, and abstain when the evidence is insufficient (section 7.8).
4. Build a test collection by pooling and measure precision, recall, F1, MAP, MRR, nDCG and bpref with paired bootstrap intervals (sections 7.10 and 8).
5. Compare retrieval-grounded answers with closed-book answers by blind human rating (section 8.12).
6. Explore a company mention graph (section 7.7) and, subject to a go/no-go check, a page-image retrieval track (section 7.9).
7. Deliver a runnable application and a reproducible repository (section 6).

**Research questions.**

* **RQ1.** Which retriever works best on 10-K passages: Boolean/phrase, TF-IDF, BM25, dense, hybrid, or hybrid with reranking?
* **RQ2.** Which preprocessing and structural choices help on financial text (stop words, stemming, numeric tokens, metadata prefix, table chunks, company filter, section prior, pseudo-relevance feedback), and what do champion lists trade for speed?
* **RQ3.** Do retrieval-grounded answers beat closed-book answers; does better retrieval give better answers; does abstention make the system more trustworthy?
* **RQ4.** Does PageRank on a company mention graph say anything that in-degree does not?
* **RQ5.** Does page-image retrieval help on visually rich pages? (gated: only attempted if a small go/no-go check passes)

**Hypotheses, fixed before the test run, and what happened.** The decision rules are in `eval/hypotheses.py`: a hypothesis that A beats B is "supported" only if the 95% bootstrap interval of the mean nDCG@10 difference excludes 0, "refuted" if it lies wholly on the other side, and otherwise inconclusive. The results are in sections 7.2, 7.7 and 8.3 to 8.12.

| H | Hypothesis | Outcome on the test split |
|---|---|---|
| H1 | BM25 beats TF-IDF | **Supported** (+0.139, CI +0.064 to +0.223) |
| H2 | BM25 matches dense on entity and terminology needs; dense wins on topical needs | Inconclusive on all three subsets (n = 6 or 7 each) |
| H3 | Hybrid is at least as good as BM25 and as dense | Beats BM25 (**supported**, +0.085, CI +0.019 to +0.155); **tied** with dense (+0.002) |
| H4 | Hybrid + reranker beats hybrid | Inconclusive (+0.044, CI -0.012 to +0.113) |
| H5 | Table chunks and numeric tokens matter for table needs | Neither supported nor refuted: only the all-needs effect was computed (about zero) |
| H6 | Pseudo-relevance feedback helps terminology needs and not entity needs | Tied on both (+0.002 and -0.006) |
| H7 | Champion lists cut latency at small quality cost | Consistent for r = 50 (median 0.237 to 0.080 ms, MAP 0.455 to 0.450), not significance-tested |
| H8 | RAG beats closed-book on correctness and number grounding | Best-RAG **supported** on correctness (+0.625, CI +0.350 to +0.900); BM25-RAG only weakly (+0.325, CI +0.050 to +0.575, carried by one rater); number grounding not comparable |
| H9 | Best-RAG is at least as good as BM25-RAG; abstention raises citation precision | First half **supported** (+0.300, CI +0.125 to +0.500); second half not tested (tau tuned to 0) |
| H10 | PageRank ranks companies differently from in-degree | **Not supported** (Spearman 0.933) |
| H11 | Page-image retrieval beats text retrieval on visually rich pages | **Not tested**: track gated and not built (section 7.9) |

RQ1 is answered in sections 8.3 to 8.5, RQ2 in 8.6 and 8.7, RQ3 in 8.9 and 8.12, RQ4 in 7.7, and RQ5 is not answered.
