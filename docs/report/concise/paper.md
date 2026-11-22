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
5. A working web app and a Docker image, so the system can be demonstrated.

@@FIG:fig_glance.png@@

## II. Related Work

We read the papers below and used each one for a specific part of the system. Table I summarises them; the text underneath adds what matters for our choices.

@@T_RELATED@@

**Classic ranking.** Salton et al. (1975) introduced the vector space model behind TF-IDF. BM25 (Robertson and Zaragoza, 2009) improves it with term saturation and length normalisation and is the standard strong baseline. Manning et al. (2008) is our course text.

**Neural and hybrid search.** Sentence encoders (Reimers and Gurevych, 2019; Wang et al., 2020) turn text into vectors that can be compared by meaning. Fusing several rankings (Fox and Shaw, 1994; Cormack et al., 2009) is cheap, because reciprocal rank fusion needs only ranks. Thakur et al. (2021) found in BEIR that BM25 is a robust baseline and that reranking is best but slow.

**Answering and evaluation.** For filings, Islam et al. (2023) report in FinanceBench that GPT-4-Turbo with retrieval answered wrongly or refused 81% of 150 questions. This is our main motivation for checkable answers. Pooled judgments can favour the systems that built the pool (Zobel, 1998), so we run bias checks.

**What is new.** None of the parts is new. What we add is the combination, on one domain: a from-scratch search core next to the modern parts, an answer checker with an honest "I don't know", and an evaluation that reports weak or unclear results as they are.

## III. Objective

We want to build and test a search and question-answering system for 10-K filings. We ask four questions (plus one optional):

* **RQ1.** Which retriever works best: Boolean, TF-IDF, BM25, dense, hybrid, or hybrid with reranking?
* **RQ2.** Which choices help: stop words, stemming, numbers, table chunks, company filter, feedback, champion lists?
* **RQ3.** Are answers with retrieval better than answers without, and can the system refuse when it should?
* **RQ4.** Does PageRank on a "who mentions whom" graph tell us more than simple counting?
* **RQ5 (optional).** Does page-image search help? We did not build this part.

We wrote 11 hypotheses before the test run; they are listed with their results in Section VI-C.

## IV. Proposed Model

[[fig:fig_model.svg]] shows the pipeline. **Offline** (once): we download each filing from SEC EDGAR, cut it into passages by 10-K section (every table is kept whole), and build the indexes. **Online** (for each question): (1) detect company names and fix spelling mistakes; (2) retrieve candidate passages with BM25 and dense search and fuse them; (3) rerank and keep the best five; (4) a language model writes an answer that cites them as [P1] to [P5]; (5) a checker verifies the citations and numbers; (6) the system shows the answer or says "insufficient evidence".

@@FIG:fig_model.svg@@

We also built a six-tab web app (Ask, Search, Graph, Pages, Evaluation, System) and a Docker image. [[fig:app]] shows two of its tabs: in the Ask tab a misspelt company name ("Telsa") is corrected before searching, and the answer carries a citation (P1) that opens the passage behind it. The Search tab runs up to three retrievers side by side and highlights the query words. The Evaluation and Graph tabs ([[fig:app2]]) show the stored results and the company graph, read from the result files.

@@FIGROW:app@@

@@FIGROW:app2@@

## V. Methodology

### A. The ten retrieval setups

We compare ten setups (Table II). The first five are written by us from the course formulas; the encoder and the reranker are pre-trained models that we use as parts.

@@T_SYSTEMS@@

### B. Search engine built from scratch

The **positional inverted index** stores, for each word, the passages and positions where it occurs; skip pointers speed up AND queries, and positions allow phrase search. Wildcards use a permuterm or 2-gram index. **TF-IDF** weights a word by how often it occurs in a passage and how rare it is overall, and passages are ranked by cosine similarity:

<p class="formula">w(t,d) = (1 + log10 tf) &middot; log10(N / df)</p>

**Champion lists** keep only the top r passages per word to make search faster. **BM25** adds saturation and length normalisation:

<p class="formula">score(q,d) = &sum; idf(t) &middot; tf (k1 + 1) / ( tf + k1 (1 &minus; b + b |d| / avgdl) )</p>

We tuned k1 = 0.6 and b = 0.9 on the 10 tuning questions only.

### C. Modern retrieval

**Dense retrieval** compares the cosine similarity of sentence vectors (all-MiniLM-L6-v2). **Hybrid** search fuses the BM25 and dense rankings with reciprocal rank fusion, which gives each passage the score 1 / (60 + rank) from each list and adds them. A **cross-encoder** then rereads the top 30 and keeps five. <p class="formula">RRF(d) = &sum; 1 / (60 + rank<sub>r</sub>(d)), summed over the BM25 and dense lists r</p>

**Pseudo-relevance feedback** (Rocchio, 1971) adds terms from the top results to the query.

### D. Company names, graph, answers

The query processor finds company names, even when misspelt, with a bigram shortlist and edit distance (at most two edits), and uses the company to filter the search; a question about two companies is split into two searches. The **company graph** links company A to B if A's filing mentions B in at least two passages; we rank companies with **PageRank** (damping d = 0.85) and HITS (Brin and Page, 1998; Kleinberg, 1999):

<p class="formula">PR(v) = (1 &minus; d) / |V| + d &sum; PR(u) / outdeg(u), over all companies u that mention v</p>

For **answers**, the model must use only the five passages, cite every claim, and otherwise reply "INSUFFICIENT EVIDENCE" (temperature 0). The **verifier** checks that every citation exists and that every number in the answer appears in a cited passage. The system refuses when the model says the evidence is insufficient. [[fig:fig_answer_flow.png]] shows this flow, and the box below is a real example from the app.

@@FIG:fig_answer_flow.png@@

<div class="example"><b>Example (real output).</b> <i>Question:</i> "How does Telsa describe the risk of depending on its Chief Executive Officer?" The spelling module changes "Telsa" to Tesla, Inc. <i>Answer:</i> "Tesla states that it is highly dependent on the services of Elon Musk, who is the Technoking of Tesla and Chief Executive Officer [P1]. Although Mr. Musk spends significant time with Tesla and is highly active in management, he does not devote his full time and attention to Tesla because he also holds management positions at other companies and is involved in other ventures [P1]." <i>Checks:</i> citation P1 exists (pass); every sentence cited (pass); numbers: none in the answer.</div>

### E. Setup and software

Table III lists the models and software behind every number, so the work can be repeated. The language-model answers are cached on disk, so the demo and the tests also run offline.

@@T_SETUP@@

### F. How we evaluate

We pooled the top 8 results of every system for each question and had each passage graded 0, 1 or 2 by a language model that did not see which system found it (a second model pass on 91 passages agreed with kappa 0.785). We report precision, recall, F1, MAP and nDCG, and use a paired bootstrap over questions for confidence intervals. For answers, two of us rated each answer 0, 1 or 2 against a reference answer we wrote from the filings, without knowing which system wrote it.

## VI. Experiments and Results

### A. Data

We built every dataset ourselves from public filings (no outside training data). We wrote 30 information needs of three kinds: about one named company (entity), across companies (topical), and about a defined term or a figure in a table (terminology). Ten are used for tuning and 20 are kept for the final test, which was run once.

@@T_DATA@@

@@FIG:fig_corpus.png@@

### B. Which retriever is best?

P@10 is the share of the top 10 results that are relevant, R@10 the share of all relevant passages found in the top 10, F1 combines the two, MAP averages precision over the ranking, and nDCG@10 also rewards putting the best passages first.

@@T_RET@@

@@FIG:fig_ndcg_systems.png@@

Hybrid search with reranking is best (nDCG@10 {{ndcg_best}}), ahead of dense ({{ndcg_dense}}), hybrid ({{ndcg_hybrid}}), BM25 ({{ndcg_bm25}}) and TF-IDF ({{ndcg_tfidf}}). Clear differences (intervals not touching zero) include BM25 over TF-IDF, hybrid over BM25, and hybrid with reranking over BM25. Hybrid and dense are tied, and the reranker's gain over hybrid is not certain with only 20 questions.

@@FIG:fig_pr_curves_s.png@@

@@FIG:fig_ndcg_by_type_s.png@@

Hybrid search with reranking was best for every type of question ([[fig:fig_ndcg_by_type_s.png]]); BM25 was weakest on questions about one named company (nDCG@10 0.58 against 0.71).

### C. What helped, and the hypotheses

The table below changes one choice at a time (BM25 as the base, 20 test questions). Adding the company, year and section to each passage helped most. Stemming hurt. Table chunks, the company filter and pseudo-relevance feedback made no measurable difference. Champion lists with r = 50 made TF-IDF about three times faster ({{champ_ms}} to {{champ50_ms}} ms) at almost no loss.

@@T_ABL@@

@@FIG:fig_ablation_s.png@@

@@FIG:fig_speed_quality_s.png@@

The next table lists our 11 hypotheses (written before the test run) and what happened.

@@T_HYP@@

@@FIG:fig_hypotheses_s.png@@

### D. Spelling correction

The spelling module fixed {{entity_ok}} of {{entity_n}} misspelt company names ({{entity_pct}}%). The one miss was the ticker MSTF for MSFT, because swapping two letters costs two edits.

@@FIG:fig_entity_accuracy_s.png@@

### E. Answer quality

@@T_HUMAN@@

@@FIG:fig_rag_ratings_s.png@@

@@FIG:fig_rating_dist_s.png@@

Answers built on the best retriever were rated clearly higher than answers without retrieval ({{d_best_cb}}, 95% interval {{ci_best_cb}}) and than answers built on BM25 ({{d_best_bm25}}, interval {{ci_best_bm25}}). BM25 answers were only weakly better than no retrieval ({{d_bm25_cb}}, interval {{ci_bm25_cb}}), and this comes mostly from one rater. The two raters agree on 41 of 60 answers (kappa {{kappa}}) and order the three systems the same way. Whenever a human and the language-model rater differed, the model gave the higher score, so we use the human ratings as the main result. Better retrieval scores did not predict better answer ratings (Spearman {{rq3}}). The checks on the answers were clean:

@@T_CITE@@

### F. Knowing when not to answer

@@T_ABST@@

@@FIG:fig_abstention_s.png@@

The system declined {{abst_tp}} of {{abst_pos}} unanswerable test questions and answered all 20 answerable ones. A rule based on the retrieval score did not work (precision {{score_p}}, recall {{score_r}}), so we use the model's own verdict. Our unanswerable questions are easy (clearly outside the filings) and the refusal relies on the model's own verdict, so this result shows that the idea works on clear cases, not that it is robust in general.

### G. Company graph

The graph has 20 companies and 14 links ("A's filing mentions B"). PageRank gives almost the same order as simply counting incoming mentions (Spearman {{pr_in}}), so on this small graph it adds little; HITS has too few links to be useful.

@@FIG:fig_graph_net_s.png@@

### H. Comparison with published work

@@T_COMPARE@@

Our results agree with the literature on fusion and on the value of retrieval for filings. They differ on BM25: it was not ahead of dense search here, probably because of the domain and our small test set.

### I. Discussion

* **Why BM25 beat TF-IDF.** Passages differ a lot in length, and BM25 corrects for length and for very frequent words; this is the likely reason (we did not test it separately).
* **Why hybrid helped.** Exact words (company names, defined terms) favour BM25, and paraphrased questions favour the dense model; fusing the two lists uses both signals.
* **From search to answers.** The better retriever also gave better answers (+0.30 over BM25 answers), but a per-question retrieval score did not predict the answer rating. Five good passages seem to be enough for the model, so small ranking differences below the top few do not change the answer.
* **Where it is still weak.** Questions that need a table row are the hardest: in one test question (Walmart segments) both RAG answers left out a segment. Two other table questions also ran without the company filter because of the "'s" problem described in Section VII.

### J. Where the system fails

The table lists the failures we found. They are the reason we do not claim general robustness.

@@T_FAIL@@

### K. Cost and reproducibility

The BM25 index builds in {{bm25_build}} s; the passages file is 11 MB and the dense vectors take 18 MB, computed once. The app starts in about a minute on a laptop. Every result in this paper is regenerated by a script in the repository, the language-model answers are cached so the demo also runs offline, a Docker image packages the app, and 137 automatic tests pass.

## VII. Conclusions and Limitations

**Conclusions.** (1) Hybrid retrieval with a reranker gave the best search quality ({{ndcg_best}} nDCG@10). (2) Answers built on it were rated {{h_best}} out of 2, against {{h_cb}} without retrieval. (3) A citation and number checker plus a refusal rule made the answers easier to trust. (4) Many differences between systems are unclear with 20 questions, and we report them that way. These results come from a small test set, model-made relevance labels and two raters from our own team, so they show a consistent pattern but are not proof of general performance.

**Limitations.**

* Small test set: 20 test questions, so many intervals are wide.
* Relevance labels were made by a language model; no human checked them.
* Both answer raters are on our team, and one checked only 48 of the 95 citations.
* One small answer-generating model; the unanswerable questions are easy.
* We did not build the page-image track (RQ5), so the system cannot search charts or scans.
* Company names with "'s" (for example "Microsoft's") are not recognised by the spelling module and then skip the company filter.
* Only the latest filing of each company is indexed.
* The system is a research prototype, not financial advice; answers must be checked against the filing.

**Future work.** More raters and test questions, human checks of the relevance labels, fixing the "'s" problem, and the page-image track.
