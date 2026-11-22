ABSTRACT: People who read company filings need exact answers, not hundreds of pages. We built TradeMesh, a search and question-answering system over the latest 10-K filings of 20 US companies (11,964 passages). It has a search engine that we wrote ourselves from the course material (inverted index, TF-IDF, BM25, spelling correction) and modern add-ons (dense retrieval, rank fusion, reranking). Every answer cites its source passages, a checker verifies the citations and numbers, and the system says "insufficient evidence" when it is unsure. On 20 test questions the best retriever (hybrid with reranking) reached nDCG@10 {{ndcg_best}}, against {{ndcg_bm25}} for BM25. Two people rated the answers blind: answers built on the best retriever scored {{h_best}} out of 2, against {{h_cb}} for answers without retrieval. The system correctly declined all 6 unanswerable test questions. Limits: a small test set, relevance labels made by a language model, two raters from our own team, and no page-image search.
KEYWORDS: information retrieval, BM25, dense retrieval, retrieval-augmented generation, financial filings, verification

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

@@FIG:fig_model.svg@@

Figure 1 shows the pipeline. **Offline** (once): we download each filing from SEC EDGAR, cut it into passages by 10-K section (every table is kept whole), and build the indexes. **Online** (for each question): (1) detect company names and fix spelling mistakes; (2) retrieve candidate passages with BM25 and dense search and fuse them; (3) rerank and keep the best five; (4) a language model writes an answer that cites them as [P1] to [P5]; (5) a checker verifies the citations and numbers; (6) the system shows the answer or says "insufficient evidence". We also built a six-tab web app (Ask, Search, Graph, Pages, Evaluation, System) and a Docker image. Figure 2 shows two of its tabs: in the Ask tab a misspelt company name ("Telsa") is corrected before searching, and the answer carries a citation (P1) that opens the passage behind it.

@@FIGROW:app@@

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

@@FIG:fig_answer_flow.png@@

### D. How we evaluate

We pooled the top 8 results of every system for each question and had each passage graded 0, 1 or 2 by a language model that did not see which system found it (a second model pass on 91 passages agreed with kappa 0.785). We report precision, recall, F1, MAP and nDCG, and use a paired bootstrap over questions for confidence intervals. For answers, two of us rated each answer 0, 1 or 2 against a reference answer we wrote from the filings, without knowing which system wrote it.

## VI. Experiments and Results

### A. Data

@@T_DATA@@

### B. Which retriever is best?

@@T_RET@@

@@FIG:fig_ndcg_systems.png@@

Hybrid search with reranking is best (nDCG@10 {{ndcg_best}}), ahead of dense ({{ndcg_dense}}), hybrid ({{ndcg_hybrid}}), BM25 ({{ndcg_bm25}}) and TF-IDF ({{ndcg_tfidf}}). Clear differences (intervals not touching zero) include BM25 over TF-IDF, hybrid over BM25, and hybrid with reranking over BM25. Hybrid and dense are tied, and the reranker's gain over hybrid is not certain with only 20 questions.

@@FIG:fig_pr_curves.png@@

### C. What helped and what did not

Adding the company, year and section to each passage helped most (+0.083 nDCG@10 in the best variant). Stemming hurt, and table chunks, the company filter and pseudo-relevance feedback made no measurable difference. Champion lists with r = 50 made TF-IDF about three times faster ({{champ_ms}} to {{champ50_ms}} ms) at almost no loss. Hybrid search with reranking was best for every type of question (Fig. 6); BM25 was weakest on questions about one named company (nDCG@10 0.58 against 0.71). Table III lists all hypotheses.

@@FIG:fig_ndcg_by_type.png@@

@@FIG:fig_speed_quality.png@@

@@T_HYP@@

### D. Spelling correction

The spelling module fixed {{entity_ok}} of {{entity_n}} misspelt company names ({{entity_pct}}%). The one miss was the ticker MSTF for MSFT, because swapping two letters costs two edits.

@@FIG:fig_entity_accuracy.png@@

### E. Answer quality

@@T_HUMAN@@

@@FIG:fig_rag_ratings.png@@

Answers built on the best retriever were rated clearly higher than answers without retrieval ({{d_best_cb}}, 95% interval {{ci_best_cb}}) and than answers built on BM25 ({{d_best_bm25}}, interval {{ci_best_bm25}}). BM25 answers were only weakly better than no retrieval ({{d_bm25_cb}}, interval {{ci_bm25_cb}}), and this comes mostly from one rater. The two raters agree on 41 of 60 answers (kappa {{kappa}}) and order the three systems the same way. Whenever a human and the language-model rater differed, the model gave the higher score, so we use the human ratings as the main result. Better retrieval scores did not predict better answer ratings (Spearman {{rq3}}).

### F. Knowing when not to answer

@@FIG:fig_abstention.png@@

The system declined {{abst_tp}} of {{abst_pos}} unanswerable test questions and answered all 20 answerable ones. A rule based on the retrieval score did not work (precision {{score_p}}, recall {{score_r}}), so we use the model's own verdict. Our unanswerable questions are easy (clearly outside the filings), so this result should not be over-read.

### G. Company graph

The graph has 20 companies and 14 links ("A's filing mentions B"). PageRank gives almost the same order as simply counting incoming mentions (Spearman {{pr_in}}), so on this small graph it adds little; HITS has too few links to be useful.

@@FIG:fig_pagerank.png@@

### H. Comparison with published work

@@T_COMPARE@@

Our results agree with the literature on fusion and on the value of retrieval for filings. They differ on BM25: it was not ahead of dense search here, probably because of the domain and our small test set.

## VII. Conclusions and Limitations

**Conclusions.** (1) Hybrid retrieval with a reranker gave the best search quality ({{ndcg_best}} nDCG@10). (2) Answers built on it were rated {{h_best}} out of 2, against {{h_cb}} without retrieval. (3) A citation and number checker plus a refusal rule made the answers easier to trust. (4) Many differences between systems are unclear with 20 questions, and we report them that way.

**Limitations.**

* Small test set: 20 test questions, so many intervals are wide.
* Relevance labels were made by a language model; no human checked them.
* Both answer raters are on our team, and one checked only 48 of the 95 citations.
* One small answer-generating model; the unanswerable questions are easy.
* We did not build the page-image track (RQ5), so the system cannot search charts or scans.
* Company names with "'s" (for example "Microsoft's") are not recognised by the spelling module and then skip the company filter.
* Only the latest filing of each company is indexed.

**Future work.** More raters and test questions, human checks of the relevance labels, fixing the "'s" problem, and the page-image track.
