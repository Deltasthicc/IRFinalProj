ABSTRACT: Readers of company filings need precise answers rather than hundreds of pages of text. This paper presents TradeMesh, a search and question-answering system over the latest Form 10-K filings of 20 US companies (11,964 passages). The system combines a search engine implemented from the course material (positional inverted index, TF-IDF, BM25 and spelling-tolerant company matching) with dense retrieval, rank fusion and cross-encoder reranking. Each answer cites the passages that support it, a deterministic checker verifies the citations and every number, and the system abstains when the evidence is insufficient. On 20 held-out test questions, hybrid retrieval with reranking achieved an nDCG@10 of {{ndcg_best}}, compared with {{ndcg_bm25}} for BM25. In a blind rating by two human raters, answers produced with this retriever scored {{h_best}} on a scale from 0 to 2, compared with {{h_cb}} for answers produced without retrieval, and the system declined all 6 unanswerable test questions. These results rest on a small test set, relevance labels produced by a language model and two raters from the authors' own team, and a page-image retrieval component was not built.
KEYWORDS: information retrieval, BM25, dense retrieval, retrieval-augmented generation, financial filings, verification

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

[[fig:fig_glance_s.png]] summarises the main results. The remainder of the paper is organised as follows: Section II reviews related work, Section III states the objectives, Section IV describes the proposed model, Section V presents the methodology, Section VI reports the experiments and results, and Section VII gives the conclusions and limitations.

@@FIG:fig_glance_s.png@@

## II. Related Work

Table I lists the work on which the system builds and the part of the system for which each paper was used.

@@T_RELATED@@

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

[[fig:fig_model.svg]] shows the pipeline. In the **offline** stage, performed once, each filing is downloaded from SEC EDGAR, divided into passages by 10-K section (every table is kept as a single chunk), and indexed. In the **online** stage, every question is processed in six steps: (1) company names are detected and spelling mistakes are corrected; (2) candidate passages are retrieved with BM25 and dense search, and the two rankings are fused; (3) the candidates are reranked and the best five are kept; (4) a language model writes an answer that cites them as [P1] to [P5]; (5) a checker verifies the citations and numbers; (6) the system presents the answer or declines with "insufficient evidence".

@@FIG:fig_model.svg@@

The system is delivered as a six-tab web application (Ask, Search, Graph, Pages, Evaluation, System) and a Docker image. [[fig:app]] shows two of its tabs. In the Ask tab, a misspelt company name ("Telsa") is corrected before searching, and the answer carries a citation (P1) that opens the supporting passage. The Search tab runs up to three retrievers side by side and highlights the query words. The Evaluation and Graph tabs ([[fig:app2]]) present the stored results and the company graph, read directly from the result files.

@@FIGROW:app@@

@@FIGROW:app2@@

## V. Methodology

### A. The ten retrieval configurations

Ten configurations are compared (Table II). The first five are implemented by us from the course formulas; the sentence encoder and the reranker are pre-trained models that are used as components.

@@T_SYSTEMS@@

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

For **answers**, the language model is instructed to use only the five passages, to cite every claim and otherwise to reply "INSUFFICIENT EVIDENCE" (temperature 0). The **verifier** checks that every citation exists and that every number in the answer appears in a cited passage. The system declines when the model reports that the evidence is insufficient. [[fig:fig_answer_flow.png]] shows this flow, and the box below gives a real example from the application.

@@FIG:fig_answer_flow.png@@

<div class="example"><b>Example (real output).</b> <i>Question:</i> "How does Telsa describe the risk of depending on its Chief Executive Officer?" The spelling module corrects "Telsa" to Tesla, Inc. <i>Answer:</i> "Tesla states that it is highly dependent on the services of Elon Musk, who is the Technoking of Tesla and Chief Executive Officer [P1]. Although Mr. Musk spends significant time with Tesla and is highly active in management, he does not devote his full time and attention to Tesla because he also holds management positions at other companies and is involved in other ventures [P1]." <i>Checks:</i> citation P1 exists (pass); every sentence is cited (pass); the answer contains no numbers.</div>

### E. Setup and software

Table III lists the models and software behind every reported number so that the work can be repeated. The language-model answers are cached on disk, so the demonstration and the tests also run offline.

@@T_SETUP@@

### F. Evaluation protocol

For every question, the top 8 results of each system were pooled, and each pooled passage was graded 0, 1 or 2 by a language model that did not know which system had retrieved it (a second model pass on 91 passages agreed with a weighted kappa of 0.785). We report precision, recall, F1, MAP and nDCG, and we use a paired bootstrap over questions to obtain confidence intervals. For the answers, two of the authors rated each answer 0, 1 or 2 against a reference answer written from the filings, without knowing which system had produced it.

## VI. Experiments and Results

### A. Data

All datasets were constructed by us from public filings, and no external training data were used. We wrote 30 information needs of three kinds: needs about one named company (entity), needs spanning several companies (topical), and needs about a defined term or a figure in a table (terminology). Ten needs were used for tuning and 20 were reserved for the final test, which was run once.

@@T_DATA@@

@@FIG:fig_corpus.png@@

### B. Retrieval effectiveness

P@10 is the share of the top 10 results that are relevant, R@10 the share of all relevant passages found in the top 10, and F1 their harmonic mean. MAP averages precision over the whole ranking, and nDCG@10 additionally rewards placing the most relevant passages first.

@@T_RET@@

@@FIG:fig_ndcg_systems.png@@

Hybrid retrieval with reranking performs best (nDCG@10 {{ndcg_best}}), followed by dense retrieval ({{ndcg_dense}}), hybrid retrieval ({{ndcg_hybrid}}), BM25 ({{ndcg_bm25}}) and TF-IDF ({{ndcg_tfidf}}). Several differences are clear in the sense that their intervals exclude zero, including BM25 over TF-IDF, hybrid retrieval over BM25, and hybrid retrieval with reranking over BM25. Hybrid and dense retrieval are statistically tied, and with only 20 questions the gain of the reranker over hybrid retrieval is uncertain.

@@FIG:fig_pr_curves_s.png@@

@@FIG:fig_ndcg_by_type_s.png@@

Hybrid retrieval with reranking was the best system for every type of question ([[fig:fig_ndcg_by_type_s.png]]). BM25 was weakest on questions about a single named company (nDCG@10 0.58 against 0.71).

### C. Design choices and hypotheses

The table below varies one design choice at a time, with BM25 as the base and the 20 test questions. Adding the company, year and section to each passage helped most, and stemming reduced effectiveness. Table chunks, the company filter and pseudo-relevance feedback made no measurable difference. Champion lists with r = 50 made TF-IDF about three times faster ({{champ_ms}} to {{champ50_ms}} ms) with almost no loss of quality.

@@T_ABL@@

@@FIG:fig_ablation_s.png@@

@@FIG:fig_speed_quality_s.png@@

The following table lists the 11 hypotheses, which were written before the test run, together with their outcomes.

@@T_HYP@@

@@FIG:fig_hypotheses_s.png@@

### D. Spelling correction

The spelling module corrected {{entity_ok}} of {{entity_n}} misspelt company names ({{entity_pct}}%). The single failure was the ticker MSTF for MSFT, because swapping two letters costs two edits.

@@FIG:fig_entity_accuracy_s.png@@

### E. Answer quality

@@T_HUMAN@@

@@FIG:fig_rag_ratings_s.png@@

@@FIG:fig_rating_dist_s.png@@

Answers based on the best retriever were rated clearly higher than answers without retrieval ({{d_best_cb}}, 95% interval {{ci_best_cb}}) and than answers based on BM25 ({{d_best_bm25}}, interval {{ci_best_bm25}}). BM25-based answers were only weakly better than answers without retrieval ({{d_bm25_cb}}, interval {{ci_bm25_cb}}), and this difference is driven mainly by one rater. The two raters agree exactly on 41 of 60 answers (kappa {{kappa}}) and order the three systems identically. Whenever a human and the language-model rater disagreed, the model gave the higher score, so the human ratings are taken as the main result. Retrieval quality did not predict answer ratings at the level of individual answers (Spearman {{rq3}}). The automatic checks on the answers were clean:

@@T_CITE@@

### F. Abstention

@@T_ABST@@

@@FIG:fig_abstention_s.png@@

The system declined {{abst_tp}} of {{abst_pos}} unanswerable test questions and answered all 20 answerable ones. A rule based on the retrieval score did not work (precision {{score_p}}, recall {{score_r}}), so the model's own verdict is used. The unanswerable questions are easy, since they lie clearly outside the filings, and the refusal relies on the model's own verdict. This result therefore shows that the approach works on clear cases, not that it is robust in general.

### G. Company graph

The graph contains 20 companies and 14 links ("A's filing mentions B"). PageRank orders the companies almost exactly as the number of incoming mentions does (Spearman {{pr_in}}), so on this small graph it adds little, and HITS has too few links to be informative.

@@FIG:fig_graph_net_s.png@@

### H. Comparison with published work

@@T_COMPARE@@

Our results agree with the literature on rank fusion and on the value of retrieval for filings. They differ on BM25, which was not ahead of dense retrieval here; the domain and the small test set are plausible explanations.

### I. Discussion

* **BM25 and TF-IDF.** Passages differ widely in length, and BM25 corrects for length and for very frequent words. This is the likely reason for its advantage, although we did not test it separately.
* **Hybrid retrieval.** Exact words such as company names and defined terms favour BM25, whereas paraphrased questions favour the dense model; fusing the two lists exploits both signals.
* **From retrieval to answers.** The better retriever also produced better answers (+0.30 over BM25-based answers), but the retrieval score of an individual question did not predict its answer rating. Five good passages appear to be sufficient for the model, so small ranking differences below the top few results do not change the answer.
* **Remaining weaknesses.** Questions that require a table row are the hardest: for one test question (Walmart segments), both retrieval-based answers omitted a segment. Two further table questions also ran without the company filter because of the possessive-name problem described in Section VII.

### J. Failure cases

The table lists the failures we found. They are the reason why we do not claim general robustness.

@@T_FAIL@@

### K. Cost and reproducibility

The BM25 index builds in {{bm25_build}} s. The passages file occupies 11 MB, and the dense vectors take 18 MB and are computed once. The application starts in about a minute on a laptop. Every result in this paper is regenerated by a script in the repository, the language-model answers are cached so that the demonstration also runs offline, a Docker image packages the application, and 137 automatic tests pass.

## VII. Conclusions and Limitations

**Conclusions.** (1) Hybrid retrieval with a cross-encoder reranker gave the best search quality ({{ndcg_best}} nDCG@10). (2) Answers based on it were rated {{h_best}} out of 2, against {{h_cb}} without retrieval. (3) A citation and number checker, together with a refusal rule, makes the answers easier to trust. (4) Many differences between systems are unclear with 20 questions, and they are reported as such. These results come from a small test set, model-produced relevance labels and two raters from the authors' own team, so they show a consistent pattern but are not proof of general performance.

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
