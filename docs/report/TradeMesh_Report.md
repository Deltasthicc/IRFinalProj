# TradeMesh: Search and Verified Question Answering over SEC 10-K Filings

Shashwat Rajan (2510111019), Tanish Shitanshu (2510110008), Ruchir Sharma (2310110728)

**Abstract.** People who read company filings need exact answers, not hundreds of pages. We built TradeMesh, a search and question-answering system over the latest 10-K filings of 20 US companies (11,964 passages). It has a search engine that we wrote ourselves from the course material (inverted index, TF-IDF, BM25, spelling correction) and modern add-ons (dense retrieval, rank fusion, reranking). Every answer cites its source passages, a checker verifies the citations and numbers, and the system says "insufficient evidence" when it is unsure. On 20 test questions the best retriever (hybrid with reranking) reached nDCG@10 0.828, against 0.699 for BM25. Two people rated the answers blind: answers built on the best retriever scored 1.20 out of 2, against 0.575 for answers without retrieval. The system correctly declined all 6 unanswerable test questions. Limits: a small test set, relevance labels made by a language model, two raters from our own team, and no page-image search.

**Index Terms.** information retrieval, BM25, dense retrieval, retrieval-augmented generation, financial filings, verification.


## I. Introduction

### A. Background, importance and motivation

Every listed US company files a yearly report called the Form 10-K. It is long (hundreds of pages) and the answer to a question is often one sentence or one table row. Keyword search finds the right words but returns passages, not answers, and it often mixes up companies. A language model gives fluent answers, but it can quote an old figure or invent one, and it does not show where the answer came from. Filings are the main public source on a company's money and risks, so a missing or invented number can mislead a decision. This is why both problems matter.

Our idea is to use search first and the language model second. The model only sees the best five passages, must cite them, and a program then checks the citations and numbers. So the quality of the answer depends on the quality of the search, which is why this is an information retrieval project.

### B. Contributions

1. A search engine built from the course material: positional index, Boolean and phrase search, TF-IDF, BM25 and a spelling-tolerant company matcher.
2. A fair comparison of ten retrieval setups on a test collection of 30 questions that we built.
3. Answers with citations, a number checker, and an "I don't know" rule, tested against answers without retrieval.
4. A careful evaluation: hypotheses written before the test run, confidence intervals, and blind human ratings.

## II. Related Work

**Classic ranking.** Salton et al. (1975) introduced the vector space model behind TF-IDF. BM25 (Robertson and Zaragoza, 2009) improves it with term saturation and length normalisation and is the standard strong baseline. Rocchio (1971) feedback expands a query using the top results. Manning et al. (2008) is our course text.

**Neural and hybrid search.** Sentence encoders (Reimers and Gurevych, 2019; Wang et al., 2020) turn text into vectors; Karpukhin et al. (2020) used them for passage retrieval. Fox and Shaw (1994) and Cormack et al. (2009) show how to fuse several rankings, and reciprocal rank fusion needs only ranks. Nogueira and Cho (2019) rerank the top results with a cross-encoder. Thakur et al. (2021) found in BEIR that BM25 is a robust baseline and that reranking is best but slow. Khattab and Zaharia (2020) and Faysse et al. (2024) use late interaction, the latter on page images.

**Answering and evaluation.** Lewis et al. (2020) combine retrieval with a text generator (RAG); Es et al. (2023) propose automatic RAG scores. For filings, Islam et al. (2023) report in FinanceBench that GPT-4-Turbo with retrieval answered wrongly or refused 81% of 150 questions, and Jimeno Yepes et al. (2024) study how to split reports into chunks. Brin and Page (1998) and Kleinberg (1999) rank nodes in a link graph. For evaluation we follow Järvelin and Kekäläinen (2002) for nDCG, Zobel (1998) and Buckley and Voorhees (2004) on pooled judgments, Cohen (1960, 1968) for agreement and Efron and Tibshirani (1993) for the bootstrap.

**What is new.** None of the parts is new. What we add is the combination, on one domain: a from-scratch search core next to the modern parts, an answer checker with an honest "I don't know", and an evaluation that reports weak or unclear results as they are.

## III. Objective

We want to build and test a search and question-answering system for 10-K filings. We ask four questions (plus one optional):

* **RQ1.** Which retriever works best: Boolean, TF-IDF, BM25, dense, hybrid, or hybrid with reranking?
* **RQ2.** Which choices help: stop words, stemming, numbers, table chunks, company filter, feedback, champion lists?
* **RQ3.** Are answers with retrieval better than answers without, and can the system refuse when it should?
* **RQ4.** Does PageRank on a "who mentions whom" graph tell us more than simple counting?
* **RQ5 (optional).** Does page-image search help? We did not build this part.

We wrote 11 hypotheses before the test run (Table III).

## IV. Proposed Model



[figure]



Figure 1 shows the pipeline. **Offline** (once): we download each filing from SEC EDGAR, cut it into passages by 10-K section (every table is kept whole), and build the indexes. **Online** (for each question): (1) detect company names and fix spelling mistakes; (2) retrieve candidate passages with BM25 and dense search and fuse them; (3) rerank and keep the best five; (4) a language model writes an answer that cites them as [P1] to [P5]; (5) a checker verifies the citations and numbers; (6) the system shows the answer or says "insufficient evidence". We also built a six-tab web app (Ask, Search, Graph, Pages, Evaluation, System) and a Docker image. Figure 2 shows two of its tabs: in the Ask tab a misspelt company name ("Telsa") is corrected before searching, and the answer carries a citation (P1) that opens the passage behind it.



[figure]



## V. Methodology

### A. Search engine built from scratch

The **positional inverted index** stores, for each word, the passages and positions where it occurs; skip pointers speed up AND queries, and positions allow phrase search. Wildcards use a permuterm or 2-gram index. **TF-IDF** weights a word by how often it occurs in a passage and how rare it is overall, and passages are ranked by cosine similarity:

<p class="formula">w(t,d) = (1 + log10 tf) &middot; log10(N / df)</p>

**Champion lists** keep only the top r passages per word to make search faster. **BM25** adds saturation and length normalisation:

<p class="formula">score(q,d) = &sum; idf(t) &middot; tf (k1 + 1) / ( tf + k1 (1 &minus; b + b |d| / avgdl) )</p>

We tuned k1 = 0.6 and b = 0.9 on the 10 tuning questions only.

### B. Modern retrieval

**Dense retrieval** compares the cosine similarity of sentence vectors (all-MiniLM-L6-v2). **Hybrid** search fuses the BM25 and dense rankings with reciprocal rank fusion, which gives each passage the score 1 / (60 + rank) from each list and adds them. A **cross-encoder** then rereads the top 30 and keeps five. **Pseudo-relevance feedback** adds terms from the top results to the query.

### C. Company names, graph, answers

The query processor finds company names, even when misspelt, with a bigram shortlist and edit distance (at most two edits), and uses the company to filter the search; a question about two companies is split into two searches. The **company graph** links company A to B if A's filing mentions B in at least two passages; we rank companies with **PageRank** (damping 0.85) and HITS. For **answers**, the model must use only the five passages, cite every claim, and otherwise reply "INSUFFICIENT EVIDENCE" (temperature 0). The **verifier** checks that every citation exists and that every number in the answer appears in a cited passage. The system refuses when the model says the evidence is insufficient. Figure 3 shows this flow.



[figure]



### D. How we evaluate

We pooled the top 8 results of every system for each question and had each passage graded 0, 1 or 2 by a language model that did not see which system found it (a second model pass on 91 passages agreed with kappa 0.785). We report precision, recall, F1, MAP and nDCG, and use a paired bootstrap over questions for confidence intervals. For answers, two of us rated each answer 0, 1 or 2 against a reference answer we wrote from the filings, without knowing which system wrote it.

## VI. Experiments and Results

### A. Data



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>THE DATA WE BUILT AND USED</div><table><tr><th>Item</th><th>Size</th></tr><tr><td>Companies / filings</td><td>20 companies, latest 10-K each (fiscal year 2025 or 2026)</td></tr><tr><td>Passages searched</td><td>11,964 (10,912 text, 1,052 tables); 1.46 million words</td></tr><tr><td>Test collection</td><td>30 information needs: 10 for tuning, 20 for the final test</td></tr><tr><td>Relevance labels</td><td>680 judgments (0, 1, 2) made by a language model</td></tr><tr><td>Unanswerable questions</td><td>10 (4 tuning, 6 test)</td></tr><tr><td>Misspelt company names</td><td>40 (for the spelling module)</td></tr></table></div>



### B. Which retriever is best?



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>RETRIEVAL RESULTS ON THE 20 TEST QUESTIONS</div><table><tr><th>System</th><th>P@10</th><th>R@10</th><th>F1@10</th><th>MAP</th><th>nDCG@10</th></tr><tr><td>TF-IDF</td><td>0.530</td><td>0.395</td><td>0.428</td><td>0.455</td><td>0.560</td></tr><tr><td>BM25</td><td>0.615</td><td>0.512</td><td>0.519</td><td>0.574</td><td>0.699</td></tr><tr><td>Dense</td><td>0.640</td><td>0.606</td><td>0.566</td><td>0.606</td><td>0.782</td></tr><tr><td>Hybrid (BM25 + dense)</td><td>0.670</td><td>0.604</td><td>0.584</td><td>0.678</td><td>0.784</td></tr><tr><td><b>Hybrid + reranker</b></td><td><b>0.705</b></td><td><b>0.658</b></td><td><b>0.621</b></td><td><b>0.693</b></td><td><b>0.828</b></td></tr></table></div>





[figure]



Hybrid search with reranking is best (nDCG@10 0.828), ahead of dense (0.782), hybrid (0.784), BM25 (0.699) and TF-IDF (0.560). Clear differences (intervals not touching zero) include BM25 over TF-IDF, hybrid over BM25, and hybrid with reranking over BM25. Hybrid and dense are tied, and the reranker's gain over hybrid is not certain with only 20 questions.



[figure]



### C. What helped and what did not

Adding the company, year and section to each passage helped most (+0.083 nDCG@10 in the best variant). Stemming hurt, and table chunks, the company filter and pseudo-relevance feedback made no measurable difference. Champion lists with r = 50 made TF-IDF about three times faster (0.24 to 0.08 ms) at almost no loss. Hybrid search with reranking was best for every type of question (Fig. 6); BM25 was weakest on questions about one named company (nDCG@10 0.58 against 0.71). Table III lists all hypotheses.



[figure]





[figure]





<div class="tbl wide"><div class="tcap">TABLE @@NO@@<br>OUR HYPOTHESES (WRITTEN BEFORE THE TEST RUN) AND WHAT HAPPENED</div><table><tr><th></th><th>Hypothesis</th><th>Result</th></tr><tr><td>H1</td><td>BM25 beats TF-IDF</td><td><b>Yes</b> (+0.139 nDCG@10)</td></tr><tr><td>H2</td><td>BM25 ties dense on exact-term questions; dense wins on topical ones</td><td>Unclear (too few questions per type)</td></tr><tr><td>H3</td><td>Hybrid beats BM25 and dense</td><td><b>Beats BM25</b> (+0.085); ties dense (+0.002)</td></tr><tr><td>H4</td><td>Reranker improves hybrid</td><td>Unclear (+0.044, interval includes 0)</td></tr><tr><td>H5</td><td>Table chunks and numbers help table questions</td><td>Not shown (no effect over all questions)</td></tr><tr><td>H6</td><td>Pseudo-relevance feedback helps</td><td>No (tied with BM25)</td></tr><tr><td>H7</td><td>Champion lists are faster at little cost</td><td>Yes for r = 50 (0.24 to 0.08 ms)</td></tr><tr><td>H8</td><td>Answers with retrieval beat answers without</td><td><b>Yes</b> for best-RAG (+0.625)</td></tr><tr><td>H9</td><td>Hybrid + rerank answers beat BM25 answers</td><td><b>Yes</b> (+0.300)</td></tr><tr><td>H10</td><td>PageRank differs from counting mentions</td><td>No (Spearman 0.93)</td></tr><tr><td>H11</td><td>Page-image retrieval beats text</td><td>Not tested (track not built)</td></tr></table></div>



### D. Spelling correction

The spelling module fixed 39 of 40 misspelt company names (97.5%). The one miss was the ticker MSTF for MSFT, because swapping two letters costs two edits.



[figure]



### E. Answer quality



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>AVERAGE ANSWER RATING (0 TO 2) FROM TWO BLIND HUMAN RATERS</div><table><tr><th>Answers made with</th><th>Mean of both</th><th>Shashwat</th><th>Ruchir</th></tr><tr><td>Without retrieval (closed-book)</td><td>0.575</td><td>0.70</td><td>0.45</td></tr><tr><td>BM25 + answer</td><td>0.90</td><td>0.80</td><td>1.00</td></tr><tr><td><b>Hybrid + reranker + answer</b></td><td><b>1.20</b></td><td>1.20</td><td>1.20</td></tr></table></div>





[figure]



Answers built on the best retriever were rated clearly higher than answers without retrieval (+0.625, 95% interval +0.35 to +0.90) and than answers built on BM25 (+0.300, interval +0.12 to +0.50). BM25 answers were only weakly better than no retrieval (+0.325, interval +0.05 to +0.57), and this comes mostly from one rater. The two raters agree on 41 of 60 answers (kappa 0.39) and order the three systems the same way. Whenever a human and the language-model rater differed, the model gave the higher score, so we use the human ratings as the main result. Better retrieval scores did not predict better answer ratings (Spearman -0.14).

### F. Knowing when not to answer



[figure]



The system declined 6 of 6 unanswerable test questions and answered all 20 answerable ones. A rule based on the retrieval score did not work (precision 0.38, recall 0.83), so we use the model's own verdict. Our unanswerable questions are easy (clearly outside the filings), so this result should not be over-read.

### G. Company graph

The graph has 20 companies and 14 links ("A's filing mentions B"). PageRank gives almost the same order as simply counting incoming mentions (Spearman 0.93), so on this small graph it adds little; HITS has too few links to be useful.



[figure]



### H. Comparison with published work



<div class="tbl wide"><div class="tcap">TABLE @@NO@@<br>OUR RESULTS COMPARED WITH PUBLISHED FINDINGS</div><table><tr><th>Published claim</th><th>Paper</th><th>Our result</th><th>Same?</th></tr><tr><td>BM25 is a strong baseline</td><td>BEIR (Thakur et al., 2021)</td><td>Dense was ahead of BM25 here; BM25 not ahead</td><td>No</td></tr><tr><td>Dense wins on paraphrased questions</td><td>DPR (Karpukhin et al., 2020)</td><td>Same direction, but unclear</td><td>Partly</td></tr><tr><td>Fusing rankings helps</td><td>RRF (Cormack et al., 2009)</td><td>Hybrid beats BM25, ties dense</td><td>Yes</td></tr><tr><td>Cross-encoder reranking helps</td><td>Nogueira and Cho (2019)</td><td>Same sign, interval includes 0</td><td>Partly</td></tr><tr><td>Retrieval matters for filings</td><td>FinanceBench (Islam et al., 2023)</td><td>Best-RAG +0.625 over closed-book</td><td>Yes</td></tr><tr><td>Page images help on visual pages</td><td>ColPali (Faysse et al., 2024)</td><td>Not tested</td><td>n/a</td></tr></table></div>



Our results agree with the literature on fusion and on the value of retrieval for filings. They differ on BM25: it was not ahead of dense search here, probably because of the domain and our small test set.

## VII. Conclusions and Limitations

**Conclusions.** (1) Hybrid retrieval with a reranker gave the best search quality (0.828 nDCG@10). (2) Answers built on it were rated 1.20 out of 2, against 0.575 without retrieval. (3) A citation and number checker plus a refusal rule made the answers easier to trust. (4) Many differences between systems are unclear with 20 questions, and we report them that way.

**Limitations.**

* Small test set: 20 test questions, so many intervals are wide.
* Relevance labels were made by a language model; no human checked them.
* Both answer raters are on our team, and one checked only 48 of the 95 citations.
* One small answer-generating model; the unanswerable questions are easy.
* We did not build the page-image track (RQ5), so the system cannot search charts or scans.
* Company names with "'s" (for example "Microsoft's") are not recognised by the spelling module and then skip the company filter.
* Only the latest filing of each company is indexed.

**Future work.** More raters and test questions, human checks of the relevance labels, fixing the "'s" problem, and the page-image track.


## REFERENCES

Brin, S. and Page, L., 1998. The anatomy of a large-scale hypertextual Web search engine. Computer Networks and ISDN Systems, 30(1–7), pp. 107–117.

Buckley, C. and Voorhees, E.M., 2004, July. Retrieval evaluation with incomplete information. In Proceedings of the 27th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 25–32). Sheffield.

Cohen, J., 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), pp. 37–46.

Cohen, J., 1968. Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit. Psychological Bulletin, 70, pp. 213–220.

Cormack, G.V., Clarke, C.L.A. and Buettcher, S., 2009, July. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 758–759). Boston.

Efron, B. and Tibshirani, R.J., 1993. An Introduction to the Bootstrap. New York: Chapman & Hall.

Es, S., James, J., Espinosa-Anke, L. and Schockaert, S., 2023. Ragas: Automated evaluation of retrieval augmented generation. arXiv preprint arXiv:2309.15217.

Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C. and Colombo, P., 2024. ColPali: Efficient document retrieval with vision language models. arXiv preprint arXiv:2407.01449.

Fox, E.A. and Shaw, J.A., 1994. Combination of multiple searches. In Harman, D.K. (ed.), The Second Text REtrieval Conference (TREC-2), NIST Special Publication 500-215 (pp. 243–252).

Islam, P., Kannappan, A., Kiela, D., Qian, R., Scherrer, N. and Vidgen, B., 2023. FinanceBench: A new benchmark for financial question answering. arXiv preprint arXiv:2311.11944.

Järvelin, K. and Kekäläinen, J., 2002. Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems, 20(4), pp. 422–446.

Jimeno Yepes, A. et al., 2024. Financial report chunking for effective retrieval augmented generation. arXiv preprint arXiv:2402.05131.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D. and Yih, W., 2020, November. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 6769–6781).

Khattab, O. and Zaharia, M., 2020, July. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 39–48).

Kleinberg, J.M., 1999. Authoritative sources in a hyperlinked environment. Journal of the ACM, 46(5), pp. 604–632.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S. and Kiela, D., 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

Manning, C.D., Raghavan, P. and Schütze, H., 2008. Introduction to Information Retrieval. Cambridge: Cambridge University Press.

Nogueira, R. and Cho, K., 2019. Passage re-ranking with BERT. arXiv preprint arXiv:1901.04085.

Reimers, N. and Gurevych, I., 2019, November. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 3982–3992). Hong Kong.

Robertson, S. and Zaragoza, H., 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4), pp. 333–389.

Rocchio, J.J., 1971. Relevance feedback in information retrieval. In Salton, G. (ed.), The SMART Retrieval System: Experiments in Automatic Document Processing (pp. 313–323). Englewood Cliffs, NJ: Prentice-Hall.

Salton, G., Wong, A. and Yang, C.S., 1975. A vector space model for automatic indexing. Communications of the ACM, 18(11), pp. 613–620.

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. and Gurevych, I., 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Advances in Neural Information Processing Systems 34, Datasets and Benchmarks Track (NeurIPS 2021).

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N. and Zhou, M., 2020. MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

Zobel, J., 1998, August. How reliable are the results of large-scale information retrieval experiments? In Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 307–314). Melbourne.
