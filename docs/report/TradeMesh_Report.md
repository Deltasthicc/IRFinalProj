# TradeMesh: Search and Verified Question Answering over SEC 10-K Filings

Shashwat Rajan (2510111019), Tanish Shitanshu (2510110008), Ruchir Sharma (2310110728)

**Abstract.** Readers of company filings need precise answers rather than hundreds of pages of text. This paper presents TradeMesh, a search and question-answering system over the latest Form 10-K filings of 20 US companies (11,964 passages). The system combines a search engine implemented from the course material (positional inverted index, TF-IDF, BM25 and spelling-tolerant company matching) with dense retrieval, rank fusion and cross-encoder reranking. Each answer cites the passages that support it, a deterministic checker verifies the citations and every number, and the system abstains when the evidence is insufficient. On 20 held-out test questions, hybrid retrieval with reranking achieved an nDCG@10 of 0.828, compared with 0.699 for BM25. In a blind rating by two human raters, answers produced with this retriever scored 1.20 on a scale from 0 to 2, compared with 0.575 for answers produced without retrieval, and the system declined all 6 unanswerable test questions. These results rest on a small test set, relevance labels produced by a language model and two raters from the authors' own team, and a page-image retrieval component was not built.

**Index Terms.** information retrieval, BM25, dense retrieval, retrieval-augmented generation, financial filings, verification.


## I. Introduction

### A. Background and motivation

The Form 10-K is the annual report that every publicly listed US company files with the Securities and Exchange Commission. A single filing runs to hundreds of pages, yet the answer to a reader's question is often one sentence or one table row. Keyword search locates the relevant words but returns passages rather than answers, and it frequently confuses companies. Large language models produce fluent answers, but they may quote outdated figures or invent numbers, and they do not indicate where an answer comes from. Filings are the principal public source of information on a company's finances and risks, so a missing or fabricated figure can mislead a decision. Both shortcomings therefore matter.

We address them by placing retrieval before generation. The language model receives only the five best passages, is required to cite them, and its output is then checked by a program. The quality of an answer consequently depends on the quality of retrieval, which makes the problem one of information retrieval.

### B. Contributions

1. A search engine implemented from the course material: a positional inverted index, Boolean and phrase search, TF-IDF, BM25 and a spelling-tolerant company matcher.
2. A controlled comparison of ten retrieval configurations on a test collection of 30 information needs constructed for this study.
3. An answer pipeline with citations, a deterministic checker for citations and numbers, and an abstention rule, evaluated against answers generated without retrieval.
4. An evaluation protocol with hypotheses fixed before the test run, bootstrap confidence intervals and blind human ratings.
5. A working web application and a container image for demonstration.

### C. Summary of results

On 20 held-out test questions, hybrid retrieval with cross-encoder reranking achieved an nDCG@10 of 0.828, compared with 0.699 for BM25. In a blind rating by two human raters, answers produced with this retriever received a mean score of 1.20 on a scale from 0 to 2, compared with 0.575 for answers produced without retrieval. The spelling module corrected 39 of 40 misspelt company names (97.5%), and the system declined all 6 unanswerable test questions while declining 0 of the 20 answerable ones.

### D. Organisation of the paper

The remainder of the paper is organised as follows. Section II reviews related work, Section III states the objectives, Section IV describes the proposed model, Section V presents the methodology, and Section VI reports the experiments and results. Section VII gives the conclusions and limitations.

## II. Related Work

Table I lists the work on which the system builds and the part of the system for which each paper was used.



<div class="tbl narrow long"><div class="tcap">TABLE @@NO@@<br>PAPERS WE USED AND FOR WHAT</div><table><tr><th>Paper</th><th>Idea</th><th>How we use it</th></tr><tr><td>Salton et al. (1975)</td><td>Vector space model, TF-IDF, cosine</td><td>TF-IDF system</td></tr><tr><td>Robertson and Zaragoza (2009)</td><td>BM25</td><td>Main baseline, tuned on 10 questions</td></tr><tr><td>Rocchio (1971)</td><td>Relevance feedback</td><td>Pseudo-relevance feedback</td></tr><tr><td>Reimers and Gurevych (2019); Wang et al. (2020)</td><td>Sentence embeddings (MiniLM)</td><td>Dense retrieval</td></tr><tr><td>Karpukhin et al. (2020)</td><td>Dense passage retrieval</td><td>Source of hypothesis H2</td></tr><tr><td>Fox and Shaw (1994); Cormack et al. (2009)</td><td>Rank fusion, reciprocal rank fusion</td><td>Hybrid search</td></tr><tr><td>Nogueira and Cho (2019)</td><td>BERT reranking</td><td>Cross-encoder reranker</td></tr><tr><td>Thakur et al. (2021)</td><td>BEIR: BM25 is a robust baseline</td><td>Comparison in Section VI-H</td></tr><tr><td>Lewis et al. (2020); Es et al. (2023)</td><td>RAG; automatic RAG scores</td><td>Answer stage; we use our own checks and human ratings</td></tr><tr><td>Islam et al. (2023); Jimeno Yepes et al. (2024)</td><td>FinanceBench; chunking financial reports</td><td>Motivation; every table kept as one chunk</td></tr><tr><td>Khattab and Zaharia (2020); Faysse et al. (2024)</td><td>Late interaction; page-image retrieval</td><td>MaxSim written; track not built</td></tr><tr><td>Brin and Page (1998); Kleinberg (1999)</td><td>PageRank; HITS</td><td>Company mention graph</td></tr><tr><td>Järvelin and Kekäläinen (2002); Zobel (1998); Buckley and Voorhees (2004); Cohen (1960, 1968); Efron and Tibshirani (1993); Manning et al. (2008)</td><td>nDCG; pooling and its bias; kappa; bootstrap; course text</td><td>Metrics, bias checks, agreement, intervals</td></tr></table></div>



**Classic ranking.** The vector space model of Salton et al. (1975) underlies TF-IDF weighting. BM25 (Robertson and Zaragoza, 2009) adds term-frequency saturation and document-length normalisation and is the standard strong lexical baseline; Manning et al. (2008) is the course text. Rocchio (1971) relevance feedback is the basis of our query-expansion experiment.

**Neural and hybrid retrieval.** Sentence encoders (Reimers and Gurevych, 2019; Wang et al., 2020) map text to vectors that can be compared by meaning, and Karpukhin et al. (2020) applied them to passage retrieval. Rankings from several systems can be combined cheaply (Fox and Shaw, 1994), and reciprocal rank fusion (Cormack et al., 2009) uses ranks only. Nogueira and Cho (2019) showed that a cross-encoder can rerank the top results effectively. Thakur et al. (2021) reported in BEIR that BM25 is a robust zero-shot baseline, whereas reranking gives the best results at the highest cost. Late-interaction scoring (Khattab and Zaharia, 2020) and page-image retrieval (Faysse et al., 2024) are related, but we implemented only the scoring function and did not build the image component.

**Answering and evaluation.** Retrieval-augmented generation (Lewis et al., 2020) conditions a generator on retrieved text, and automatic scores for such systems have been proposed (Es et al., 2023). For financial filings, Islam et al. (2023) report in FinanceBench that GPT-4-Turbo with a retrieval system answered incorrectly or refused 81% of 150 questions; this motivates our checkable answers. Jimeno Yepes et al. (2024) study how financial reports should be divided into chunks, and Brin and Page (1998) and Kleinberg (1999) rank the nodes of a link graph. For evaluation we follow Järvelin and Kekäläinen (2002) for nDCG, Cohen (1960, 1968) for agreement and Efron and Tibshirani (1993) for the bootstrap. Pooled judgments can favour the systems that built the pool (Zobel, 1998; Buckley and Voorhees, 2004), and we ran checks for this bias, which are reported in the extended version of this report.

**Novelty.** Each component is established. The contribution lies in their combination within one domain and in the evaluation: a search core written from scratch is placed beside neural components, an answer checker abstains instead of guessing, and unclear or negative results are reported as such.

## III. Objective

The aim of this work is to build and evaluate a search and question-answering system for 10-K filings. Four research questions guide the study, and a fifth, optional question was not pursued.

* **RQ1.** Which retriever performs best: Boolean search, TF-IDF, BM25, dense retrieval, hybrid retrieval, or hybrid retrieval with reranking?
* **RQ2.** Which design choices help: stop words, stemming, numeric tokens, table chunks, a company filter, feedback and champion lists?
* **RQ3.** Do answers based on retrieval outperform answers without it, and can the system decline to answer when it should?
* **RQ4.** Does PageRank on a graph of companies that mention one another provide information beyond simple counting?
* **RQ5 (optional).** Does page-image retrieval help? This question was not addressed.

Eleven hypotheses were fixed before the test run; they are listed with their outcomes in Section VI-C.

## IV. Proposed Model

Fig. 1 shows the pipeline. In the **offline** stage, performed once, each filing is downloaded from SEC EDGAR, divided into passages by 10-K section (every table is kept as a single chunk), and indexed. In the **online** stage, every question is processed in six steps: (1) company names are detected and spelling mistakes are corrected; (2) candidate passages are retrieved with BM25 and dense search, and the two rankings are fused; (3) the candidates are reranked and the best five are kept; (4) a language model writes an answer that cites them as [P1] to [P5]; (5) a checker verifies the citations and numbers; (6) the system presents the answer or declines with "insufficient evidence".



[figure]



The system is delivered as a six-tab web application (Ask, Search, Graph, Pages, Evaluation, System) and a Docker image. Fig. 2 shows two of its tabs. In the Ask tab, a misspelt company name ("Telsa") is corrected before searching, and the answer carries a citation (P1) that opens the supporting passage. The Search tab runs up to three retrievers side by side and highlights the query words. The Evaluation and Graph tabs (Fig. 3) present the stored results and the company graph, read directly from the result files.



[figure]





[figure]



## V. Methodology

### A. The ten retrieval configurations

Ten configurations are compared (Table II). The first five are implemented by us from the course formulas; the sentence encoder and the reranker are pre-trained models that are used as components.



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>THE TEN RETRIEVAL SETUPS (NINE ROWS; THE CHAMPION-LIST ROW IS TWO SETUPS)</div><table><tr><th>Setup</th><th>What it does</th><th>Made by</th></tr><tr><td>Boolean</td><td>AND / OR / NOT, phrase and wildcard search; unranked</td><td>Built by us</td></tr><tr><td>TF-IDF</td><td>Cosine similarity of TF-IDF vectors</td><td>Built by us</td></tr><tr><td>TF-IDF + champion lists</td><td>Same, scoring only the top r passages per word (r = 50, 200); 2 setups</td><td>Built by us</td></tr><tr><td>BM25</td><td>Probabilistic ranking with length normalisation</td><td>Built by us</td></tr><tr><td>BM25 + feedback</td><td>BM25 with Rocchio query expansion</td><td>Built by us</td></tr><tr><td>Dense</td><td>Cosine of sentence vectors</td><td>Pre-trained encoder, our search code</td></tr><tr><td>Hybrid</td><td>BM25 and dense fused with reciprocal rank fusion</td><td>Our fusion code</td></tr><tr><td>BM25 + reranker</td><td>Cross-encoder rereads BM25's top 30</td><td>Pre-trained reranker</td></tr><tr><td>Hybrid + reranker</td><td>Cross-encoder rereads the hybrid's top 30</td><td>Pre-trained reranker</td></tr></table></div>



### B. Search engine built from scratch

The **positional inverted index** stores, for each word, the passages and the positions in which it occurs. Skip pointers accelerate AND queries, and the positions enable phrase search. Wildcard queries use a permuterm or 2-gram index. **TF-IDF** weights a word by its frequency in a passage and by its rarity in the collection, and passages are ranked by cosine similarity:

<p class="formula">w(t,d) = (1 + log10 tf) &middot; log10(N / df)</p>

**Champion lists** restrict scoring to the top r passages of each word and thereby reduce latency. **BM25** adds term-frequency saturation and length normalisation:

<p class="formula">score(q,d) = &sum; idf(t) &middot; tf (k1 + 1) / ( tf + k1 (1 &minus; b + b |d| / avgdl) )</p>

The parameters k1 = 0.6 and b = 0.9 were tuned on the 10 tuning questions only.

### C. Modern retrieval

**Dense retrieval** ranks passages by the cosine similarity between sentence vectors (all-MiniLM-L6-v2). **Hybrid** retrieval fuses the BM25 and dense rankings by reciprocal rank fusion, in which each passage receives 1 / (60 + rank) from each list and the contributions are summed. A **cross-encoder** then rereads the top 30 candidates and keeps five. **Pseudo-relevance feedback** (Rocchio, 1971) expands the query with terms from the top results.

<p class="formula">RRF(d) = &sum; 1 / (60 + rank<sub>r</sub>(d)), summed over the BM25 and dense lists r</p>

### D. Company names, graph and answers

The query processor detects company names, including misspelt ones, with a bigram shortlist and an edit-distance check of at most two edits. The detected company is used to filter the search, and a question that names two companies is split into two searches. The **company graph** links company A to company B if A's filing mentions B in at least two passages. Companies are ranked with **PageRank** (damping factor d = 0.85) and HITS (Brin and Page, 1998; Kleinberg, 1999):

<p class="formula">PR(v) = (1 &minus; d) / |V| + d &sum; PR(u) / outdeg(u), over all companies u that mention v</p>

For **answers**, the language model is instructed to use only the five passages, to cite every claim and otherwise to reply "INSUFFICIENT EVIDENCE" (temperature 0). The **verifier** checks that every citation exists and that every number in the answer appears in a cited passage. The system declines when the model reports that the evidence is insufficient. Fig. 4 shows this flow, and the box below gives a real example from the application.



[figure]



<div class="example"><b>Example (real output).</b> <i>Question:</i> "How does Telsa describe the risk of depending on its Chief Executive Officer?" The spelling module corrects "Telsa" to Tesla, Inc. <i>Answer:</i> "Tesla states that it is highly dependent on the services of Elon Musk, who is the Technoking of Tesla and Chief Executive Officer [P1]. Although Mr. Musk spends significant time with Tesla and is highly active in management, he does not devote his full time and attention to Tesla because he also holds management positions at other companies and is involved in other ventures [P1]." <i>Checks:</i> citation P1 exists (pass); every sentence is cited (pass); the answer contains no numbers.</div>

### E. Setup and software

Table III lists the models and software behind every reported number so that the work can be repeated. The language-model answers are cached on disk, so the demonstration and the tests also run offline.



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>MODELS AND SOFTWARE BEHIND THE RESULTS</div><table><tr><th>Part</th><th>Choice</th></tr><tr><td>Answer model</td><td>Google Gemini API, model gemini-3.5-flash-lite, temperature 0; every answer cached</td></tr><tr><td>Dense encoder</td><td>sentence-transformers/all-MiniLM-L6-v2 (384 dimensions), pre-trained</td></tr><tr><td>Reranker</td><td>cross-encoder/ms-marco-MiniLM-L-6-v2 on the top 30, pre-trained</td></tr><tr><td>Relevance labels</td><td>a language model (Claude), two passes; its exact version was not recorded</td></tr><tr><td>Software</td><td>Python 3.14.6, sentence-transformers 6.3.0, torch 2.15.0, transformers 5.23.0, numpy 2.5.3, pandas 3.1.1, Streamlit 1.66.1</td></tr><tr><td>Statistics</td><td>paired bootstrap over questions, 10,000 resamples, seed 358</td></tr><tr><td>Run on</td><td>a laptop, no GPU needed; the test run was done once and locked</td></tr></table></div>



### F. Evaluation protocol

For every question, the top 8 results of each system were pooled, and each pooled passage was graded 0, 1 or 2 by a language model that did not know which system had retrieved it (a second model pass on 91 passages agreed with a weighted kappa of 0.785). We report precision, recall, F1, MAP and nDCG, and we use a paired bootstrap over questions to obtain confidence intervals. For the answers, two of the authors rated each answer 0, 1 or 2 against a reference answer written from the filings, without knowing which system had produced it.

## VI. Experiments and Results

### A. Data

All datasets were constructed by us from public filings, and no external training data were used. We wrote 30 information needs of three kinds: needs about one named company (entity), needs spanning several companies (topical), and needs about a defined term or a figure in a table (terminology). Ten needs were used for tuning and 20 were reserved for the final test, which was run once.



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>THE DATA WE BUILT AND USED</div><table><tr><th>Item</th><th>Size</th></tr><tr><td>Companies / filings</td><td>20 companies, latest 10-K each (fiscal year 2025 or 2026)</td></tr><tr><td>Passages searched</td><td>11,964 (10,912 text, 1,052 tables); 1.46 million words</td></tr><tr><td>Test collection</td><td>30 information needs: 10 for tuning, 20 for the final test</td></tr><tr><td>Relevance labels</td><td>680 judgments (0, 1, 2) made by a language model</td></tr><tr><td>Unanswerable questions</td><td>10 (4 tuning, 6 test)</td></tr><tr><td>Misspelt company names</td><td>40 (for the spelling module)</td></tr></table></div>





[figure]



### B. Retrieval effectiveness

P@10 is the share of the top 10 results that are relevant, R@10 the share of all relevant passages found in the top 10, and F1 their harmonic mean. MAP averages precision over the whole ranking, and nDCG@10 additionally rewards placing the most relevant passages first.



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>RETRIEVAL RESULTS ON THE 20 TEST QUESTIONS</div><table><tr><th>System</th><th>P@10</th><th>R@10</th><th>F1@10</th><th>MAP</th><th>nDCG@10</th></tr><tr><td>TF-IDF</td><td>0.530</td><td>0.395</td><td>0.428</td><td>0.455</td><td>0.560</td></tr><tr><td>BM25</td><td>0.615</td><td>0.512</td><td>0.519</td><td>0.574</td><td>0.699</td></tr><tr><td>Dense</td><td>0.640</td><td>0.606</td><td>0.566</td><td>0.606</td><td>0.782</td></tr><tr><td>Hybrid (BM25 + dense)</td><td>0.670</td><td>0.604</td><td>0.584</td><td>0.678</td><td>0.784</td></tr><tr><td><b>Hybrid + reranker</b></td><td><b>0.705</b></td><td><b>0.658</b></td><td><b>0.621</b></td><td><b>0.693</b></td><td><b>0.828</b></td></tr></table></div>





[figure]



Hybrid retrieval with reranking performs best (nDCG@10 0.828), followed by dense retrieval (0.782), hybrid retrieval (0.784), BM25 (0.699) and TF-IDF (0.560). Several differences are clear in the sense that their intervals exclude zero, including BM25 over TF-IDF, hybrid retrieval over BM25, and hybrid retrieval with reranking over BM25. Hybrid and dense retrieval are statistically tied, and with only 20 questions the gain of the reranker over hybrid retrieval is uncertain.



[figure]





[figure]



Hybrid retrieval with reranking was the best system for every type of question (Fig. 8). BM25 was weakest on questions about a single named company (nDCG@10 0.58 against 0.71).

### C. Design choices and hypotheses

The table below varies one design choice at a time, with BM25 as the base and the 20 test questions. Adding the company, year and section to each passage helped most, and stemming reduced effectiveness. Table chunks, the company filter and pseudo-relevance feedback made no measurable difference. Champion lists with r = 50 made TF-IDF about three times faster (0.24 to 0.08 ms) with almost no loss of quality.



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>ONE CHANGE AT A TIME (NDCG@10, 20 TEST QUESTIONS)</div><table><tr><th>Change</th><th>nDCG@10</th><th>Difference</th><th>95% interval</th><th>Clear?</th></tr><tr><td>BM25 as is (base)</td><td>0.699</td><td>-</td><td>-</td><td>-</td></tr><tr><td>+ company, year, section added to each passage</td><td>0.759</td><td>+0.060</td><td>-0.02 to +0.15</td><td>no</td></tr><tr><td>+ the same, numbers not indexed</td><td>0.783</td><td>+0.083</td><td>+0.01 to +0.17</td><td><b>yes</b></td></tr><tr><td>Stemming words</td><td>0.633</td><td>-0.066</td><td>-0.13 to -0.01</td><td><b>yes</b></td></tr><tr><td>Without table chunks</td><td>0.699</td><td>-0.001</td><td>-0.00 to +0.00</td><td>no</td></tr><tr><td>Without the company filter</td><td>0.689</td><td>-0.010</td><td>-0.02 to +0.00</td><td>no</td></tr><tr><td>+ pseudo-relevance feedback</td><td>0.679</td><td>-0.020</td><td>-0.05 to +0.01</td><td>no</td></tr><tr><td>Hybrid fusion, k = 60 (our choice)</td><td>0.784</td><td>+0.085</td><td>+0.02 to +0.15</td><td><b>yes</b></td></tr><tr><td>Hybrid fusion, k = 10</td><td>0.820</td><td>+0.121</td><td>+0.06 to +0.19</td><td><b>yes</b></td></tr></table></div>





[figure]





[figure]



The following table lists the 11 hypotheses, which were written before the test run, together with their outcomes.



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>OUR HYPOTHESES (WRITTEN BEFORE THE TEST RUN) AND WHAT HAPPENED</div><table><tr><th></th><th>Hypothesis</th><th>Result</th></tr><tr><td>H1</td><td>BM25 beats TF-IDF</td><td><b>Yes</b> (+0.139 nDCG@10)</td></tr><tr><td>H2</td><td>BM25 ties dense on exact-term questions; dense wins on topical ones</td><td>Unclear (too few questions per type)</td></tr><tr><td>H3</td><td>Hybrid beats BM25 and dense</td><td><b>Beats BM25</b> (+0.085); ties dense (+0.002)</td></tr><tr><td>H4</td><td>Reranker improves hybrid</td><td>Unclear (+0.044, interval includes 0)</td></tr><tr><td>H5</td><td>Table chunks and numbers help table questions</td><td>Not shown (no effect over all questions)</td></tr><tr><td>H6</td><td>Pseudo-relevance feedback helps</td><td>No (tied with BM25)</td></tr><tr><td>H7</td><td>Champion lists are faster at little cost</td><td>Yes for r = 50 (0.24 to 0.08 ms)</td></tr><tr><td>H8</td><td>Answers with retrieval beat answers without</td><td><b>Yes</b> for best-RAG (+0.625)</td></tr><tr><td>H9</td><td>Hybrid + rerank answers beat BM25 answers</td><td><b>Yes</b> (+0.300)</td></tr><tr><td>H10</td><td>PageRank differs from counting mentions</td><td>No (Spearman 0.93)</td></tr><tr><td>H11</td><td>Page-image retrieval beats text</td><td>Not tested (track not built)</td></tr></table></div>





[figure]



### D. Spelling correction

The spelling module corrected 39 of 40 misspelt company names (97.5%). The single failure was the ticker MSTF for MSFT, because swapping two letters costs two edits.



[figure]



### E. Answer quality



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>AVERAGE ANSWER RATING (0 TO 2) FROM TWO BLIND HUMAN RATERS</div><table><tr><th>Answers made with</th><th>Mean of both</th><th>Shashwat</th><th>Ruchir</th></tr><tr><td>Without retrieval (closed-book)</td><td>0.575</td><td>0.70</td><td>0.45</td></tr><tr><td>BM25 + answer</td><td>0.90</td><td>0.80</td><td>1.00</td></tr><tr><td><b>Hybrid + reranker + answer</b></td><td><b>1.20</b></td><td>1.20</td><td>1.20</td></tr></table></div>





[figure]





[figure]



Answers based on the best retriever were rated clearly higher than answers without retrieval (+0.625, 95% interval +0.35 to +0.90) and than answers based on BM25 (+0.300, interval +0.12 to +0.50). BM25-based answers were only weakly better than answers without retrieval (+0.325, interval +0.05 to +0.57), and this difference is driven mainly by one rater. The two raters agree exactly on 41 of 60 answers (kappa 0.39) and order the three systems identically. Whenever a human and the language-model rater disagreed, the model gave the higher score, so the human ratings are taken as the main result. Retrieval quality did not predict answer ratings at the level of individual answers (Spearman -0.14). The automatic checks on the answers were clean:



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>CHECKS ON THE ANSWERS</div><table><tr><th>Answers made with</th><th>Citations that support their sentence</th><th>Numbers found in cited text</th><th>Verifier passed</th></tr><tr><td>Without retrieval</td><td>none (no citations)</td><td>n/a</td><td>n/a</td></tr><tr><td>BM25 + answer</td><td>16 of 16</td><td>1.00</td><td>1.00</td></tr><tr><td>Hybrid + reranker + answer</td><td>31 of 32</td><td>1.00</td><td>1.00</td></tr></table></div>



### F. Abstention



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>DECIDING WHEN TO REFUSE (26 TEST ITEMS). P = PRECISION, R = RECALL, ACC. = ACCURACY</div><table><tr><th>Rule</th><th>P</th><th>R</th><th>F1</th><th>Acc.</th><th>TP/FP/FN/TN</th></tr><tr><td>Retrieval-score threshold</td><td>0.38</td><td>0.83</td><td>0.53</td><td>0.65</td><td>5/8/1/12</td></tr><tr><td><b>Model's own verdict</b></td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>6/0/0/20</td></tr></table></div>





[figure]



The system declined 6 of 6 unanswerable test questions and answered all 20 answerable ones. A rule based on the retrieval score did not work (precision 0.38, recall 0.83), so the model's own verdict is used. The unanswerable questions are easy, since they lie clearly outside the filings, and the refusal relies on the model's own verdict. This result therefore shows that the approach works on clear cases, not that it is robust in general.

### G. Company graph

The graph contains 20 companies and 14 links ("A's filing mentions B"). PageRank orders the companies almost exactly as the number of incoming mentions does (Spearman 0.93), so on this small graph it adds little, and HITS has too few links to be informative.



[figure]



### H. Comparison with published work



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>OUR RESULTS COMPARED WITH PUBLISHED FINDINGS</div><table><tr><th>Published claim</th><th>Paper</th><th>Our result</th><th>Agrees?</th></tr><tr><td>BM25 is a strong baseline</td><td>BEIR (Thakur et al., 2021)</td><td>Dense was ahead of BM25 here; BM25 not ahead</td><td>No</td></tr><tr><td>Dense wins on paraphrased questions</td><td>DPR (Karpukhin et al., 2020)</td><td>Same direction, but unclear</td><td>Partly</td></tr><tr><td>Fusing rankings helps</td><td>RRF (Cormack et al., 2009)</td><td>Hybrid beats BM25, ties dense</td><td>Yes</td></tr><tr><td>Cross-encoder reranking helps</td><td>Nogueira and Cho (2019)</td><td>Same sign, interval includes 0</td><td>Partly</td></tr><tr><td>Retrieval matters for filings</td><td>FinanceBench (Islam et al., 2023)</td><td>Best-RAG +0.625 over closed-book</td><td>Yes</td></tr><tr><td>Page images help on visual pages</td><td>ColPali (Faysse et al., 2024)</td><td>Not tested</td><td>n/a</td></tr></table></div>



Our results agree with the literature on rank fusion and on the value of retrieval for filings. They differ on BM25, which was not ahead of dense retrieval here; the domain and the small test set are plausible explanations.

### I. Discussion

* **BM25 and TF-IDF.** Passages differ widely in length, and BM25 corrects for length and for very frequent words. This is the likely reason for its advantage, although we did not test it separately.
* **Hybrid retrieval.** Exact words such as company names and defined terms favour BM25, whereas paraphrased questions favour the dense model; fusing the two lists exploits both signals.
* **From retrieval to answers.** The better retriever also produced better answers (+0.30 over BM25-based answers), but the retrieval score of an individual question did not predict its answer rating. Five good passages appear to be sufficient for the model, so small ranking differences below the top few results do not change the answer.
* **Remaining weaknesses.** Questions that require a table row are the hardest: for one test question (Walmart segments), both retrieval-based answers omitted a segment. Two further table questions also ran without the company filter because of the possessive-name problem described in Section VII.

### J. Failure cases

The table lists the failures we found. They are the reason why we do not claim general robustness.



<div class="tbl narrow"><div class="tcap">TABLE @@NO@@<br>WHERE THE SYSTEM FAILS OR IS NOT PROVEN</div><table><tr><th>Case</th><th>What happened</th><th>Why</th></tr><tr><td>Ticker typo MSTF (for MSFT)</td><td>Not corrected</td><td>Swapping two letters costs two edits; short strings allow one</td></tr><tr><td>Walmart sales by segment</td><td>Both answers left out a segment</td><td>The checker tests presence of citations and numbers, not completeness; the company filter was skipped ("Walmart's"), which may have contributed</td></tr><tr><td>Microsoft revenue by segment</td><td>The BM25 answer declined</td><td>The company filter was skipped ("Microsoft's"), which may have contributed</td></tr><tr><td>Topical questions, BM25 answers</td><td>Declined 3 of 7</td><td>The model judged the BM25 passages insufficient</td></tr><tr><td>The number checker</td><td>Flagged none of 55 answers</td><td>On this data it never showed that it can catch an error (its unit tests do)</td></tr></table></div>



### K. Cost and reproducibility

The BM25 index builds in 0.64 s. The passages file occupies 11 MB, and the dense vectors take 18 MB and are computed once. The application starts in about a minute on a laptop. Every result in this paper is regenerated by a script in the repository, the language-model answers are cached so that the demonstration also runs offline, a Docker image packages the application, and 137 automatic tests pass.

## VII. Conclusions and Limitations

**Conclusions.** (1) Hybrid retrieval with a cross-encoder reranker gave the best search quality (0.828 nDCG@10). (2) Answers based on it were rated 1.20 out of 2, against 0.575 without retrieval. (3) A citation and number checker, together with a refusal rule, makes the answers easier to trust. (4) Many differences between systems are unclear with 20 questions, and they are reported as such. These results come from a small test set, model-produced relevance labels and two raters from the authors' own team, so they show a consistent pattern but are not proof of general performance.

**Limitations.**

* The test set is small (20 test questions), so many intervals are wide.
* The relevance labels were produced by a language model, and no human checked them.
* Both answer raters belong to the authors' team, and one of them checked only 48 of the 95 citations.
* A single small answer-generating model was used, and the unanswerable questions are easy.
* The page-image component (RQ5) was not built, so the system cannot search charts or scanned pages.
* Company names with a possessive ending (for example "Microsoft's") are not recognised by the spelling module and then skip the company filter.
* Only the latest filing of each company is indexed.
* The system is a research prototype and not financial advice; answers must be checked against the filing.

**Future work.** More raters and test questions, human checks of the relevance labels, a fix for the possessive-name problem, and the page-image component.


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
