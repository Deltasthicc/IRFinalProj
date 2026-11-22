# TradeMesh: Search and Verified Question Answering over SEC 10-K Filings

Shashwat Rajan (2510111019), Tanish Shitanshu (2510110008), Ruchir Sharma (2310110728)

**Abstract.** Analysts who ask questions about company filings need exact figures and the passage that supports them. We present TradeMesh, search and question answering over the latest Form 10-K filings of 20 US companies (11,964 passages). A classic IR core written from the lecture formulas (positional index, TF-IDF, BM25, spelling-corrected company names) is combined with dense retrieval, rank fusion, cross-encoder reranking, a company mention graph, and answers that cite their passages, are checked by a verifier and abstain when evidence is thin. On 20 test needs, with hypotheses fixed beforehand and paired bootstrap intervals, hybrid retrieval with reranking is strongest (nDCG@10 0.828 against 0.699 for BM25, difference +0.129, 95% CI +0.052 to +0.213), but hybrid retrieval is tied with dense retrieval alone and the reranker's gain over hybrid is inconclusive. Spelling correction resolved 39 of 40 misspelt names. In an answer study rated blind by two humans (mean of the two raters), retrieval with reranking scores 1.20 of 2 against 0.575 for closed-book (+0.625, CI +0.350 to +0.900), while BM25-based RAG (0.90) is only weakly better than closed-book (+0.325, CI +0.050 to +0.575, carried by one of the two raters). The model's own insufficient-evidence verdict separated 6 of 6 unanswerable test questions from 20 answerable ones; a retrieval-score threshold did not. Limits: small collections, model-assisted relevance judgments, two human raters who are both on the team, one small generator, and a page-image track that was scoped out.

**Index Terms.** information retrieval, BM25, dense retrieval, reciprocal rank fusion, retrieval-augmented generation, financial filings, verification.

## I. Introduction

### A. Background

Every listed US company files an annual report on Form 10-K with the Securities and Exchange Commission. A 10-K is a long, semi-structured document, divided into numbered Items (business description, risk factors, management's discussion, financial statements and so on), and the answer to a typical question is one sentence or one table row inside a few hundred pages. The filings of the 20 companies in this project add up to about 1.46 million words. Two familiar tools serve the reader badly here. Keyword search finds the right words but not always the right company or section, and it returns passages, not answers. A language model answers fluently, but without access to the filing it may quote an outdated figure or invent one, and nothing in its answer tells the reader which sentence of which filing to check.

### B. Importance

Both failures are costly when the reader is making a decision about money. Financial question answering is also a demanding test for retrieval: the vocabulary is repetitive (every filing talks about "risk factors", "revenue" and "competition"), the same sentence pattern appears in all companies so the company name carries most of the signal, figures sit in tables that plain text indexing handles badly, and the correct answer must be exact, not approximately right. Published work on this setting reports that retrieval-augmented systems still fail often on filings (FinanceBench, Islam et al. 2023, reports that GPT-4-Turbo with a retrieval system answered incorrectly or refused 81% of 150 sampled questions).

### C. Motivation

The quality of a retrieval-augmented answer is bounded by the five or so passages handed to the language model. We therefore treat retrieval as the object of study. We implement the classic rankers ourselves from the lecture formulas, so that their behaviour can be explained term by term, add the modern ones (dense retrieval, rank fusion, cross-encoder reranking), and ask which combination works on financial text and why. On top of the retrieval layer we ask a second question: can an answer be made checkable, so that a reader can see which passage supports each claim, a program can test that every number in the answer occurs in the cited text, and the system can decline when the evidence is thin?

### D. What we built

TradeMesh answers questions over the most recent 10-K of 20 companies. A user can run the search engine and compare retrievers side by side, or ask a question and receive an answer in which every claim carries a citation to a labelled passage, a verifier checks that the citations exist and that every number occurs in the cited text, and the system says "insufficient evidence" instead of guessing. A misspelt company name ("Telsa") is corrected before retrieval, and a question that names two companies is split into one search per company. A six-tab web application (Ask, Search, Graph, Pages, Evaluation, System) and a Docker image make the system runnable and demonstrable.

### E. Contributions

1. A working, tested retrieval stack in which every ranker is written from the lecture formulas: positional index with skip pointers, Boolean, phrase and wildcard search, TF-IDF with champion lists, BM25, pseudo-relevance feedback, and our own spelling and company-name matcher. Dense encoder and cross-encoder are pre-trained models used as components.
2. A pooled test collection of 30 information needs with blind, LLM-assisted relevance judgments, consistency between two LLM passes measured by Cohen's kappa, and three explicit checks for pooling bias.
3. Hypotheses and decision rules fixed before the test run and applied mechanically, so that failed hypotheses are reported as findings and not explained away.
4. A comparison of our ordering of methods with published results (BEIR, DPR, RRF, BERT reranking, FinanceBench, ColPali), reported by direction of effect because the corpora differ.
5. A verified-RAG pipeline with a deterministic number and citation checker and a rule for abstaining, evaluated by blind ratings from two humans and, for comparison, two model-assisted passes; the report states where the two disagree and discloses a reference answer that was corrected after the first scoring.
6. A company mention graph with PageRank and HITS, and an honest finding that on this corpus it adds little beyond counting mentions.

### F. Organisation of the report

The rest of the paper is organised as follows. Section II reviews related work and states what is new. Section III gives the objectives, research questions and hypotheses. Section IV presents the proposed model. Section V describes the methods and formulas (V-A to V-J). Section VI describes the datasets and reports the experiments, results and a comparison with published findings (VI-A to VI-L). Section VII gives the conclusions and limitations, and the references follow.

## II. Literature review

We review the work our design rests on in six groups: classical ranking and query expansion, neural and hybrid retrieval, retrieval-augmented generation and its evaluation, financial question answering, link analysis, and evaluation methodology. Every paper cited here was checked against the publisher, the ACL Anthology, arXiv or at least two bibliographic databases; where only the abstract was read, the text below claims no more than the abstract does.

### A. Classical ranking and query expansion

Salton, Wong and Yang (1975) represent documents and queries as weighted term vectors and rank by their similarity; our TF-IDF system is this vector space model with cosine similarity in the logarithmic weighting taught in the course (Manning et al. 2008). Robertson and Zaragoza (2009) survey the probabilistic relevance framework behind BM25, which adds term-frequency saturation (k1) and document-length normalisation (b) to an inverse-document-frequency weight; BM25 remains the standard strong lexical baseline and is the reference system in our experiments. Rocchio (1971) introduced relevance feedback by moving the query vector towards judged relevant documents; with the top results assumed relevant it becomes pseudo-relevance feedback, which we evaluate (it does not help on our collection).

### B. Neural and hybrid retrieval

Karpukhin et al. (2020) show that a dense bi-encoder trained for question answering beats BM25 on most open-domain datasets, while noting that lexical overlap favours BM25 on one of them and that dense models struggle with rare, salient phrases. Reimers and Gurevych (2019) make sentence embeddings practical with siamese BERT networks, and Wang et al. (2020) compress such encoders with deep self-attention distillation; our dense retriever uses a MiniLM-based sentence encoder of this family. Khattab and Zaharia (2020) propose late interaction (ColBERT), encoding query and document separately and scoring them by a cheap token-level interaction. Nogueira and Cho (2019) show that a BERT cross-encoder re-ranking a BM25 candidate list improves passage ranking, which motivates our reranking stage. Fox and Shaw (1994) study how to combine several searches (CombSUM, CombMNZ), and Cormack, Clarke and Buettcher (2009) show that reciprocal rank fusion, which uses only ranks, outperforms Condorcet fusion and individual rank-learning methods; our hybrid system fuses BM25 and dense rankings this way, and we test the alternatives. Thakur et al. (2021) assemble BEIR, a zero-shot benchmark over heterogeneous domains, and find that BM25 is a robust baseline, that re-ranking and late-interaction models give the best average results at a high computational cost, and that dense and sparse retrievers are cheaper but often under-perform; this is the published ordering we compare against in section VI-K.

### C. Retrieval-augmented generation and its evaluation

Lewis et al. (2020) combine a pre-trained generator with a non-parametric memory that is retrieved at question time, the formulation that our answer stage follows: retrieve passages, then generate from them. Es et al. (2023) propose RAGAS, a reference-free set of metrics for scoring a RAG pipeline (retrieval quality, faithfulness of the answer to the context, relevance of the answer). Metrics of that kind use a language model as judge. We chose a different route for the two properties that matter most for financial answers: a deterministic checker that tests whether every cited passage exists and every number occurs in the cited text, and blind human ratings of answer quality against a reference written from the filings, kept separate from the model-assisted ratings so that the two can be compared.

### D. Question answering over financial filings

Islam et al. (2023) introduce FinanceBench, questions about public filings with reference answers, and report that 16 model configurations on a 150-case sample fail often: GPT-4-Turbo with a retrieval system answered incorrectly or refused 81% of the questions. This is the motivation for our checkable answers, although our corpus, questions and models differ, so we cannot compare absolute numbers. Jimeno Yepes et al. (2024) study how financial reports should be chunked for retrieval-augmented generation and argue for chunks that follow the structure of the report rather than a fixed size; we follow the same idea by splitting by 10-K Item and keeping every table as one chunk. Faysse et al. (2024) present ColPali, which retrieves document pages directly from page images with a vision-language model and late interaction, and a benchmark (ViDoRe) on which it outperforms text pipelines for visually rich documents. We implement the MaxSim scoring from this line of work but did not build the page-image track (section V-I).

### E. Link analysis

Brin and Page (1998) describe PageRank, a ranking based on the link structure of a graph, computed by power iteration with a damping factor, and Kleinberg (1999) gives HITS, which assigns each node a hub and an authority score. We apply both to a graph in which companies are linked by mentions in each other's filings. The graph is small and sparse, and our result is that PageRank adds little to simple in-degree there (section V-G).

### F. Evaluation methodology

Järvelin and Kekäläinen (2002) define cumulated-gain measures for graded relevance, of which nDCG is the one we report as the headline. Zobel (1998) examines pooled test collections and finds that relative system performance appears reliable but that recall is overestimated because relevant documents outside the pool are missed; Buckley and Voorhees (2004) propose bpref, which uses only judged documents and so is robust to incomplete judgments. Both motivate the pool-bias checks in section VI-H. Cohen (1960, 1968) gives kappa and weighted kappa for agreement between judges, and Efron and Tibshirani (1993) the bootstrap, which we use over questions because 20 test questions are too few for conventional tests.

### G. What is new here

None of the components is new, and we do not claim that they are. What this project adds is their combination and the way it is evaluated, in a setting where the literature mostly reports one component at a time:

* a single system in which a from-scratch classic core (index, TF-IDF, BM25, spelling-tolerant company matching) sits beside the neural stages, so that all ten retrieval configurations are compared on the same financial passages with the same judgments;
* a deterministic verifier that checks citations and numbers against the cited text, with abstention based on the model's own verdict, and an explicit negative result: a score threshold on the retrieval confidence does not separate answerable from unanswerable questions;
* an evaluation discipline new to most course projects: hypotheses and decision rules fixed before the test run, paired bootstrap intervals, pool-bias checks, and a blind human rating whose disagreement with the model-assisted rating is reported instead of hidden;
* a documented defect and a documented correction (a reference answer that was wrong and was fixed after the first scoring), with both results reported.

The limits on this claim are in section VII: the collections are small, the relevance judgments are model-assisted, and the human ratings come from two raters who are both on the team.

## III. Objective

**Aim.** To build and evaluate a search and question-answering system over SEC 10-K filings in which the retrieval core is understood and implemented from the course material, the answers are checkable against the filing, and every claim about which method is better is backed by a pre-declared test with an interval, not by a single score.

**Objectives.**

1. Implement a complete retrieval stack for 10-K passages: positional index with Boolean, phrase and wildcard search, TF-IDF with champion lists, BM25, pseudo-relevance feedback, dense retrieval, rank fusion and cross-encoder reranking (sections V-A to V-E).
2. Handle the way people actually ask: tolerate misspelt company names, filter and split by company, and keep tables searchable (section V-F).
3. Produce answers that cite their passages, pass a deterministic check of citations and numbers, and abstain when the evidence is insufficient (section V-H).
4. Build a test collection by pooling and measure precision, recall, F1, MAP, MRR, nDCG and bpref with paired bootstrap intervals (sections V-J and 8).
5. Compare retrieval-grounded answers with closed-book answers by blind human rating (section VI-L).
6. Explore a company mention graph (section V-G) and, subject to a go/no-go check, a page-image retrieval track (section V-I).
7. Deliver a runnable application and a reproducible repository (section IV).

**Research questions.**

* **RQ1.** Which retriever works best on 10-K passages: Boolean/phrase, TF-IDF, BM25, dense, hybrid, or hybrid with reranking?
* **RQ2.** Which preprocessing and structural choices help on financial text (stop words, stemming, numeric tokens, metadata prefix, table chunks, company filter, section prior, pseudo-relevance feedback), and what do champion lists trade for speed?
* **RQ3.** Do retrieval-grounded answers beat closed-book answers; does better retrieval give better answers; does abstention make the system more trustworthy?
* **RQ4.** Does PageRank on a company mention graph say anything that in-degree does not?
* **RQ5.** Does page-image retrieval help on visually rich pages? (gated: only attempted if a small go/no-go check passes)

**Hypotheses, fixed before the test run, and what happened.** The decision rules are in `eval/hypotheses.py`: a hypothesis that A beats B is "supported" only if the 95% bootstrap interval of the mean nDCG@10 difference excludes 0, "refuted" if it lies wholly on the other side, and otherwise inconclusive. The results are in sections V-B, V-G and VI-C to VI-L.

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
| H11 | Page-image retrieval beats text retrieval on visually rich pages | **Not tested**: track gated and not built (section V-I) |

RQ1 is answered in sections VI-C to VI-E, RQ2 in VI-F and VI-G, RQ3 in VI-I and VI-L, RQ4 in V-G, and RQ5 is not answered.

## IV. Proposed model

*[Fig.: fig_model.svg]* The TradeMesh model. Offline build (collection, indexes, graph), the two online stages (retrieve, then generate and check) and the evaluation scripts. The dashed box is the gated vision track, which was not built. Text retrieval and reranking run on a laptop CPU; generation is one cached language-model call per question.


TradeMesh answers questions over the latest 10-K filings of about 20 companies. The system has an offline stage that is run once and two online stages that run for every question.

**Offline.** A polite crawler downloads each filing; a parser splits it into passages by 10-K Item, with every table kept as one chunk (caption, header row and rows as text). The passages feed a positional inverted index, dense vectors, a company mention graph and, if the go/no-go gate passes, page-image vectors.

**Online, stage 1 (retrieve).** The *query processor* detects company names with tolerance for spelling errors, removes them from the topical part of the question, and, when two or more companies are named, issues one sub-query per company. The *retrievers* produce a candidate list; an optional section prior re-scores it; the cross-encoder keeps the top five.

**Online, stage 2 (generate and check).** The *context builder* labels the five passages `[P1]` to `[P5]` with ticker, fiscal year and Item. The language model (temperature 0, every call cached) must answer only from them, cite every claim, and otherwise output exactly `INSUFFICIENT EVIDENCE`. A deterministic *verifier* then checks the citations and every number, and an *abstention rule* decides whether to show the answer.

**Application.** The Streamlit app has six tabs:

* **Ask**: the answer with bold `[Pn]` markers, expandable evidence (cited passages open by default), a trust panel with the verifier results, a compare-companies mode and a closed-book toggle that shows the no-evidence baseline.
* **Search**: the classic engine. The user picks up to three retrievers, sees them side by side with key-word-in-context snippets and a per-term score explanation, and can filter by company.
* **Graph**: the company mention graph with PageRank, hub and authority scores.
* **Pages**: page-image retrieval from the vision track; it states that the track was not built.
* **Evaluation**: the stored retrieval, answer-rating, abstention and entity results, read from the result files.
* **System**: a readiness checklist, headline numbers and a spelling-correction lab.

A tab whose back end is not ready shows a "not available yet" message instead of failing.

## V. Methodology

This section describes the algorithms and formulas. The retrieval systems are numbered as in the experiments; each is a separate, switchable component with the same interface (query in, ranked list of passage numbers out), so that any of them can be run side by side in the application and in the evaluation scripts.

| System | Name in results | Section |
|---|---|---|
| Unranked Boolean/phrase search | `boolean` | V-A |
| TF-IDF with cosine, heap top-K | `tfidf` | V-B |
| TF-IDF with champion lists (r = 50, r = 200) | `tfidf-champ-50`, `tfidf-champ-200` | V-B |
| BM25 | `bm25` | V-C |
| BM25 with pseudo-relevance feedback | `bm25+prf` | V-E |
| Dense bi-encoder | `dense` | V-D |
| Hybrid: BM25 and dense fused by reciprocal rank fusion | `hybrid` | V-D |
| Cross-encoder reranking of BM25 or of hybrid | `bm25+rerank`, `hybrid+rerank` | V-D |

Around the retrievers sit the query processor with spelling-tolerant company matching, an optional section prior and table chunks (V-F), the company mention graph (V-G), the retrieval-augmented answer stage with its verifier and abstention rule (V-H), the gated vision track (V-I), and the evaluation protocol (V-J).

**Notation.** N is the number of passages (11,964), tf(t, d) the frequency of term t in passage d, df(t) its document frequency, |d| the length of d in tokens and avgdl the average length. Passages are the retrieval unit throughout: a text chunk of a 10-K Item or one whole table.

**Passage construction.** Each filing is split by 10-K Item into passages (mean 122 words, median 131, longest 713); every table is kept as one chunk (caption, header row and rows as text), and each passage carries its ticker, fiscal year, Item and page. Section VI-A describes the resulting corpus.

**Reproducibility.** The test split is run once, after the feature freeze, and a lock file in the scripts prevents a second run from overwriting the result; the BM25 parameters k1 and b were tuned on the 10 development needs only, the other settings are standard defaults, and variations of them are reported as ablations (section VI-F), not used to choose the system. Language-model calls use temperature 0 and a disk cache keyed by the model, prompt and parameters.

### A. Positional inverted index, Boolean, phrase and wildcard search

The Boolean and phrase machinery follows the standard treatment in Manning et al. 2008. Each passage is a document. The index maps a term to two parallel lists: the passage numbers in ascending order and, for each, the sorted token positions of the term. The analyzer is the one used elsewhere except that stop words are kept, because removing them would shift positions and break phrases such as "the company".

**Intersection with skip pointers.** AND is the merge of two sorted lists. For a list of length L the skip spacing is int(sqrt(L)), at least 1. Skips are implicit: position i carries one if i is a multiple of the spacing and i + spacing lies inside the list. While one list's current entry is smaller than the other's, the merge follows skips as long as the landing entry does not exceed the other entry, and otherwise advances by one. The function also returns the number of comparisons, so the saving can be measured.

**Phrase and proximity.** Candidates are the intersection of the terms' lists. For each candidate we keep a frontier of positions and, for each next term, retain the positions q for which some frontier position p satisfies 0 < q - p <= g. A phrase is g = 1; larger g gives "within g tokens, in order".

**Boolean queries.** A recursive-descent parser accepts AND, OR, NOT (upper case only), parentheses, quoted phrases and terms with "*". Precedence is NOT, AND, OR; adjacent terms mean AND; NOT is the complement within all passages. The unranked retriever used in the comparison orders matches by the summed counts of the query terms, ties by passage number.

**Wildcards.** For one star, X*Y, a permuterm index (all rotations of term + "$", sorted) answers the prefix query Y$X by binary search. For several stars, a 2-gram index over "$term$" intersects the posting sets of the pattern's bigrams and filters the survivors with a regular expression, since bigrams alone admit false positives. The expanded terms are OR-ed.

The unit tests check that skip and plain intersection both equal set intersection on random lists and that skips use under a fifth of the comparisons on a long-versus-short pair; that Boolean, phrase (including word order and stop-word phrases) and proximity queries return hand-chosen passages; that numbers match with or without thousands separators; and that both wildcard routes agree with brute force. No stored result measures the skip-pointer saving or wildcard latency. The only stored figures here are for the unranked Boolean retriever on the 20 test needs: nDCG@10 0.364 and MAP 0.220, against 0.699 nDCG@10 for BM25 (eval/out/retrieval_test.json).

### B. TF-IDF with cosine, heap top-K and champion lists

System 1 is the classical vector space model (Salton et al. 1975) in the weighting scheme described by Manning et al. 2008. For a term t and passage d, with N passages and document frequency df(t),

    w(t, d) = (1 + log10 tf(t, d)) * log10(N / df(t))      if tf > 0, else 0
    cos(q, d) = sum_t w(t, q) w(t, d) / (||q|| ||d||)

The query is weighted the same way, and query terms absent from the collection are dropped. Document vectors are length-normalised once at build time and stored as postings (term to passage to normalised weight), so a query needs only dot products. Scoring is term-at-a-time: each query term walks its postings and adds w(t, q) w(t, d) to a per-passage accumulator, and the K best are taken with a heap (heapq.nlargest). Ties go to the lower passage number.

**Champion lists.** For each term we precompute the r passages with the highest weight and, at query time, score only the union of the query terms' champion lists. Two details differ from a literal reading of the proposal: the ranking inside a list uses the length-normalised weight, not the raw w(t, d), and if the union holds fewer than K passages the query falls back to the full postings. The index counts these fallbacks, but no stored result reports them, so we cannot say how often they occurred. We build r = 50 and r = 200.

**Result (H7).** H7 states that champion lists cut latency with a small quality loss. On 11,964 passages, median latency was 0.237 ms for full TF-IDF, 0.080 ms for r = 50 and 0.228 ms for r = 200 (95th percentile 0.549, 0.122 and 0.308 ms; eval/out/cost_table.json). On the 20 test needs, nDCG@10 was 0.560 for all three systems and MAP was 0.455, 0.450 and 0.455 respectively (data/results/retrieval_test.json). So r = 50 gives about a threefold median speed-up for a MAP loss of 0.005, consistent with H7, while r = 200 gives no useful speed-up. These are single timing runs at sub-millisecond latencies and no significance test of H7 is stored, so this is supporting evidence, not a confirmed effect. All three remain well below BM25 (H1, mean nDCG@10 difference 0.139; data/results/hypotheses_test.json).

The unit tests check TF-IDF scores against a hand computation on a three-passage collection, that a short passage outranks a long one with the same term, that empty and unknown queries return nothing, and that champion lists shrink the candidate set (5 against 60 passages) and trigger the counted fallback when more results are requested than the lists hold.

All rankers return the same `Hit(passage_id, score)` list, so every one can be evaluated, shown in the Search tab and used by the RAG pipeline without special cases.

### C. BM25 (System 2), implemented from scratch

For query q and passage d, with N passages, df(t) passages containing t, tf(t,d) occurrences, |d| tokens and avgdl the mean length:

    score(q, d) = sum over t in q of  idf(t) * tf(t,d) * (k1 + 1) / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )
    idf(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

The "+1" inside the logarithm keeps idf positive for terms that occur in more than half of the passages, which matters in 10-Ks where words such as "company" are everywhere. Two properties explain why BM25 should beat plain tf-idf (H1): the factor tf / (tf + k1 ...) saturates, so the tenth occurrence of "tariff" adds little to the third, and b scales the penalty for long passages, whereas tf-idf with cosine normalisation uses one fixed length rule.

**Implementation.** Our own inverted index: one NumPy postings array (passage numbers and term frequencies) per term. A query touches only the postings of its terms (term-at-a-time), accumulates scores in an array and selects the top k with `argpartition`; ties are broken by passage number so every run is reproducible. `explain(query, passage)` returns each query term's contribution to a score, which the demo uses to show *why* a passage ranks where it does.

**Analysis choices (RQ2).** Tokens are lower-cased alphanumerics. Numbers keep their value but lose thousands separators, so `383,285` and `383285` match. Defaults: stop words removed (a short 118-word English list), no stemming, numbers kept, no metadata prefix. Each choice is one switch in `AnalyzerConfig` and the 16-way grid is run by `eval/ablation.py`. The default is a hypothesis (H5 says numeric tokens help); it is changed only if the dev split says so, and the choice is reported either way.

**Tuning.** k1 in {0.6, 0.9, 1.2, 1.5, 1.8, 2.0} and b in {0.3, 0.5, 0.75, 0.9, 1.0} are scored by mean nDCG@10 on the 10 development needs only (`eval/tune_bm25.py`); the full grid is printed so the report shows the surface, not just its maximum. With 10 queries the surface is flat and noisy, so ties go to the textbook 1.2 / 0.75. Result (`eval/out/bm25_grid_dev.json`): k1 = 0.6 and b = 0.9, with mean dev nDCG@10 of 0.795, against 0.758 for the textbook 1.2 / 0.75. The surface is flat near its top (the three best cells score 0.795, 0.794 and 0.789; the worst of the 30 cells scores 0.728), and with 10 development needs we do not read the exact optimum as meaningful. The chosen values are stored in `data/judgments/bm25_params.json` and were frozen before the test run.

### D. Dense (System 3), hybrid (System 4), cross-encoder (System 5)

**Dense.** Each passage is encoded once by `all-MiniLM-L6-v2` (Reimers and Gurevych 2019; MiniLM, Wang et al. 2020) into a 384-dimension unit vector, and the query likewise; the score is the cosine, which for unit vectors is a dot product. Exact search over a matrix of a few thousand rows takes milliseconds, so no approximate index is used. Embeddings are cached on disk by a hash of the model and texts. The encoder is an interface: tests use a hashing encoder with no download, and only the real model produces reportable numbers.

**Hybrid by Reciprocal Rank Fusion** (Cormack, Clarke and Buettcher 2009):

    RRF(d) = sum over rankings r of  1 / (60 + rank_r(d))

It uses ranks only, so BM25 scores (unbounded) and cosine scores (between -1 and 1) need no calibration. We also implement CombSUM and CombMNZ on min-max normalised scores (Fox and Shaw 1994) as comparison points; RRF's k is varied in the ablation (10, 60, 100).

**Cross-encoder.** The top 30 hybrid candidates are rescored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, which reads the query and the passage together (Nogueira and Cho 2019) and so can use word-level interactions that two independent embeddings miss. It is too slow to run over the whole corpus, hence the two-stage design; the top 5 go to the RAG prompt. Scores are passed through a sigmoid so the abstention threshold tau is comparable across queries.

Why expect each step to help or not (H2 to H4): BEIR (Thakur et al. 2021) found BM25 a robust baseline that dense retrievers often fail to beat out of domain, and re-ranking and late-interaction models to be best on average, at high computational cost; DPR (Karpukhin et al. 2020) found dense retrieval better than BM25 on most datasets, with BM25 ahead where questions and passages share many words (SQuAD) and dense retrieval weaker on rare salient phrases. Our corpus is entity- and number-heavy, so we predict a tie or BM25 win on entity and terminology needs and a dense win on topical needs.

### E. Pseudo-relevance feedback (Rocchio 1971)

    q_new = alpha * q0 + beta * (1/k) * sum over d in top-k of  d_hat

q0 is the unit-length query vector and d_hat the L2-normalised BM25 term-weight vector of a feedback passage. We take k = 5, alpha = 1, beta = 0.5, keep the m = 10 strongest terms not already in the query, and use q_new's components as term weights inside the BM25 sum (original terms keep weight about 1, expansion terms count for less). The hypothesis (H6) is that PRF helps terminology needs, where the user's words differ from the filing's ("free cash flow" versus "cash provided by operations"), and can hurt entity needs, where the top passages are already about the right company and expansion adds drift. A k x m grid (k in {3, 5, 10}, m in {5, 10, 20}) is run on the dev split to show sensitivity; the headline uses the pre-stated defaults, not the best grid cell.

#### What is not claimed in sections V-C to V-E

No claim of superiority is made before the test run. The tuning above uses the development split only, and the 20 test needs are scored once.

### F. Query processing, entities, section prior, table chunks

#### Company entities with spelling tolerance

Analysts misspell company names ("Telsa", "Micorsoft"), and a misspelt name silently removes the best filter we have. We therefore detect company mentions with an edit-distance matcher built from scratch (no fuzzy-matching library), following the dictionary and edit-distance material of Manning et al. (2008).

*Dictionary.* Each company has a name, a ticker and aliases. Every string is reduced to a key: lower-case letters and digits only ("Coca-Cola" becomes `cocacola`). The legal suffix is also dropped ("Apple Inc." gives `apple`).

*Bigram shortlist.* Every key is padded as `$key$` and split into character bigrams, and an index maps each bigram to the entries containing it. For an input key `q` we collect the entries that share at least one bigram and rank them by Jaccard overlap

J(q, e) = |B(q) ∩ B(e)| / |B(q) ∪ B(e)|,

where B(·) is the set of bigrams. Entries with J < 0.2 are dropped and the best ten are kept.

*Levenshtein confirmation.* The shortlist is confirmed with the unit-cost edit distance computed by the standard two-row dynamic program. A candidate is accepted if the distance is at most 2, and at most 1 when either string has four characters or fewer (this covers tickers, where two edits would map almost any string onto some other ticker). Ties are broken by distance, then by Jaccard overlap; if two different companies tie on both, we refuse to guess.

*Finding mentions in a question.* We slide a window of one to four words over the question, longest first, and accept the first window that matches. Three guards limit false positives: a word that already occurs in the corpus vocabulary is never "corrected" (so "apply" does not become "Apple"); a window may not start or end with a function word; and tickers of three letters or fewer count only when typed in capitals ("KO", "BA").

*Use.* The detected companies are removed from the question to give the *topic query*, and retrieval is restricted to those companies (entity filter). The interface tells the user when a correction was made ("Did you mean Tesla?").

*Evaluation.* On 40 typo'd company names and tickers the matcher reaches **39/40 = 0.975** correct (0 wrong company, 1 no match); the bigram shortlist agrees with a brute-force Levenshtein scan over the whole dictionary on all 40 inputs (`eval/entity_eval.py`, output in `eval/out/entity_eval.json`). The one failure is the ticker typo `MSTF` for `MSFT`: a transposition costs two edits in plain Levenshtein distance, which the one-edit limit for short strings rejects. A Damerau variant that counts a transposition as one edit would fix it; we kept plain Levenshtein, as planned, and report the failure. The typo set was written by us, so this measures the matcher on typo types we thought of, not on real user typos. The result is the same with the proposal company list and with the final `data/companies.json` (39/40 both times), and the 40 inputs were not changed between the two runs.

*[Fig.: fig_abstention.png]* Precision, recall and F1 of the score-only abstention rule and the model-verdict abstention rule on the 26 test items, with the positive class being "should abstain"; true/false positive and negative counts are printed below the plot. Sources: `eval/out/abstention_scoreonly.json` and `eval/out/abstention_eval.json` (the `test` entries).

#### Section prior

Filings differ in how often a section holds what analysts ask about: risk factors (Item 1A) and management's discussion (Item 7) answer many questions, while mine-safety disclosures or exhibit lists rarely do. This is a static quality score in the sense of Manning et al. (2008). We add it to the retrieval score:

final(q, d) = base(q, d) + λ · g(section(d)),  g ∈ [0, 1].

`base` is min-max normalised over the candidate list first, so λ means the same for BM25, dense and hybrid scores. `g` is set by hand per Item (for example 1.0 for Items 1A and 7, 0.8 for Items 1 and 8, 0.1 for Items 1B, 4 and 9). `tune_lambda` in `query/section_prior.py` can choose λ from {0, 0.05, 0.1, 0.2, 0.3, 0.5} by mean nDCG@10 on the 10 development queries, with ties going to the smaller λ. We implemented and unit-tested the prior but did not run that tuning or evaluate its effect: λ = 0 in every reported run, so the section prior is not part of any result in this report and we make no claim about it.

#### Table chunks, as they relate to querying

Each table is one passage (caption, header row and rows as text), so a figure stays attached to its row and column labels. This matters for querying in two ways. First, a question such as "What was Microsoft's revenue by segment?" can be answered from a single retrieved chunk. Second, the verifier knows the scale of a table: if a chunk says "(in millions)", its plain numbers also count at that scale, so "383,285" in the table supports "383,285 million" in an answer (section V-H).

### G. Company mention graph (PageRank and HITS)

The graph links the 20 companies by how their filings talk about each other. There is an edge A -> B if at least m = 2 passages of A's filing mention company B. Mentions are found by strict matching, not by the typo-tolerant entity matcher of section V-F: company names and aliases of at least four characters, matched case-sensitively as whole words, never tickers (which collide with ordinary words such as BA or KO) and never a company's own names. Fuzzy matching would create false edges in running prose. The edge weight is the number of mentioning passages.

PageRank (Brin and Page 1998) is computed by power iteration with damping d = 0.85:

    PR(v) = (1 - d)/|V| + d * sum over u -> v of PR(u) / outdeg(u)

Companies with no outgoing edge (dangling nodes) spread their rank uniformly over all nodes, so the scores always sum to one. Iteration starts from the uniform vector and stops when the L1 change falls below 1e-12 or after 1000 iterations. HITS (Kleinberg 1999) alternates authority a = A^T h and hub h = A a from all-ones vectors, L2-normalising both vectors at every step, with the same stopping rule. In-degree is the plain count of incoming edges. There is no ground truth for importance, so we make no business claims.

**Result (H10).** H10 predicts that PageRank ranks companies differently from in-degree. The stored graph has 20 nodes and 14 edges; ten companies have no outgoing edge and nine have no incoming edge (data/graph/company_graph.json). PageRank and in-degree are strongly correlated (Spearman 0.933, Kendall 0.600), as are authority and in-degree (Spearman 0.957), while PageRank and authority agree less (Spearman 0.801). The Kendall value is computed without tie correction and in-degree has many ties, which lowers it. On this graph H10 is therefore not supported: PageRank adds little beyond counting mentions. The clearest local difference is XOM, which has in-degree 1 yet ranks second by PageRank (0.0978, just behind AAPL at 0.0984), because its single in-link comes from CVX, which itself has an in-link. HITS is degenerate on a graph this sparse and fragmented: only three companies receive a non-zero authority score and four a non-zero hub score, so its scores should not be read as a full ranking.

The unit tests check PageRank against a hand-solved two-node graph with a dangling node, uniform scores on a symmetric cycle, that a star's centre ranks first and scores sum to one, that HITS separates hubs from authorities with unit-length vectors, and that mention matching is whole-word, case-sensitive, excludes self-mentions and applies the threshold m.


*[Fig.: fig_pagerank.png]* PageRank (bars, left axis) and in-degree (diamonds, right axis) of the 12 highest-ranked companies in the company mention graph, which has 20 companies and 14 edges; the title gives the Spearman correlations stored with the graph. Source: `data/graph/company_graph.json` (nodes and `correlations` entries).

### H. RAG, verifier and abstention

Retrieval-augmented generation lets a language model condition on retrieved text (Lewis et al. 2020). A fluent model can still invent a figure or a citation, and question answering over SEC filings is hard even with retrieval (Islam et al. 2023). We therefore wrap generation in a deterministic check and give the system a way to say that it does not know.

#### Context and prompt

The context is the top five passages, labelled `[P1]` to `[P5]`, each with a header of the form `TICKER | FY2025 | Item 1A | text`. In compare mode (two or more companies named) each company contributes its best three passages, grouped by company, so up to six passages are shown. The system prompt tells the model to use only the passages, to put the label of the supporting passage after every claim, to copy numbers exactly as written, to answer in one to four sentences, and, if the passages are not enough, to output exactly `INSUFFICIENT EVIDENCE`. The model runs at temperature 0 and every call is cached on disk by the hash of (model, prompt, parameters), so a run is reproducible and the demo works offline. The **closed-book baseline** uses the same model and question with no passages and no citations.

#### Verifier

The verifier uses no language model. For an answer `a` and context `P1..Pn` it checks:

1. **Citations exist.** Every `[Pi]` in `a` has 1 ≤ i ≤ n.
2. **Citations are present.** `a` cites at least one passage; sentences with no citation are listed. Sentences are split after `.`, `!` or `?` followed by a space, except after abbreviations such as Mr., Dr., U.S. and e.g.; a fragment that is only a citation (`Sales rose. [P1]`) belongs to the previous sentence.
3. **Numbers are grounded.** Every number in `a` occurs in a cited passage.

Numbers are normalised before comparison: `$` and thousands separators are removed, `%` and "percent" are ignored, and the words *thousand, million, billion, trillion* (and `bn`, `mn`) scale the value, so `$1.2 billion` and `1,200 million` are the same number. If a passage says "(in thousands)", "(in millions)" or "(in billions)", its plain numbers also count at that scale. The fiscal year in the passage header counts as present. Item numbers, "10-K" and the labels themselves are not treated as claims. Matching is exact on the normalised value, because the prompt asks for verbatim figures; a rounded figure ("$383.3 billion" for 383,285 million) is therefore flagged. This is strict by design and is listed under limitations.

A failed check is **flagged in the trust panel and never hidden or repaired**. For each number the panel shows whether it was found in a cited passage, found only in an uncited passage, or not found.

#### Abstention

The system abstains if the model outputs `INSUFFICIENT EVIDENCE`, or if the best score is below a threshold τ:

abstain ⇔ conf(q) < τ  or  answer = INSUFFICIENT EVIDENCE.

Until the cross-encoder is available the score is conf(q) = s₁ / |T(q)|, where s₁ is the best first-stage retrieval score and |T(q)| the number of distinct content words in the query (so long queries are not favoured); when the reranker is installed its best score is used instead and the rule is unchanged. τ is tuned on the development queries only: candidates are 0 and the midpoints between consecutive observed confidences, we maximise F1 with the positive class "should abstain", and ties go to the smaller τ. A score below τ skips the model call, which also saves cost.

#### Evaluation protocol

The test needs (20) are run once, after the freeze, in three configurations: closed-book, BM25 context and the best retriever with the reranker. Two of us (Shashwat Rajan and Ruchir Sharma) each wrote a reference answer for each question from the filings and then rated each answer 0, 1 or 2 against it, blind and independently (an earlier two-pass model-assisted rating is kept for comparison; the two humans are compared with each other, and each with the model, by Cohen's kappa): the order of the three answers is shuffled separately for every question and the system names are hidden (presentation bias, as in the reputation-systems lecture). We report mean rating, the share rated 2, Cohen's κ between the two humans and between each human and the model-assisted ratings, **citation precision** (share of cited passages that the raters judge to support the claim; a citation counts only if every rater who checked it agrees) and **number grounding** (share of answers with numbers whose numbers are all found in cited passages, computed automatically). For abstention we use 20 answerable and 10 unanswerable questions with the positive class "should abstain" and report precision, recall, F1 and accuracy; τ is fixed on the dev items, so the headline numbers are on the test items. Because the test collection is small we report per-question outcomes next to means and make no significance claims without support.

Everything is implemented in `src/trademesh/rag/` and `eval/rag_*.py`, `eval/abstention_eval.py`.

### I. Page-image retrieval (gated vision track): scoped out

The proposal contained an optional track that retrieves 10-K pages directly from page images, for questions whose answer sits in a chart or a laid-out table that text extraction flattens. It was gated: the track would be attempted only if a small go/no-go check passed, and cancelled and reported otherwise. **The track was not built, and no result for it (RQ5, H11) exists.** The Pages tab of the application and the readiness checklist state this.

**What is implemented.** The late-interaction scoring of ColPali-style retrieval (Faysse et al. 2024; the MaxSim operator of ColBERT, Khattab and Zaharia 2020). A query is a set of n token vectors q_1..q_n and a page a set of m patch vectors p_1..p_m, and

    MaxSim(q, page) = sum over i = 1..n of max over j = 1..m of <q_i, p_j>

Pages are ranked by this score by brute force (a few hundred pages need no vector database), ties broken by page id. The scoring function, a ranking helper and a writer for the results file the application reads are in `src/trademesh/vision/maxsim.py` and are covered by unit tests on hand-computed vectors.

**What is not implemented.** The page images, the vision-language encoder that turns pages and queries into vectors (`ColPaliEncoder` is a stub that says so) and the go/no-go check itself (embed five pages and test whether MaxSim returns the right page for three of five hand-written questions). The encoder needs a GPU for practical indexing and a multi-gigabyte model download; we did not set that up, so the check was not run (and so did not fail either).

**Why this is a sound omission.** (1) The track was optional and gated in the proposal, and the gate was not passed (the check was never run), so leaving it out is within the plan. (2) The text pipeline already indexes every table as a chunk, and the table-lookup questions are covered by the text-based experiments (with the weakness reported in section VI-L). (3) Adding an untested vision component, or reporting a number for it, would have been less honest than leaving it out; the claims in this report are limited to what was run. What the omission costs: the report cannot say anything about charts and images in 10-Ks, and H11 and RQ5 stay open.

### J. Evaluation protocol

This section states, before any result exists, how retrieval is scored, so that a reader can see nothing was tuned on the test set.

**Collection.** 30 information needs (10 entity, 10 topical, 10 terminology; 5 of the terminology needs require a table), each with a short query. 10 are development needs, used for every choice; 20 are test needs, scored once after the feature freeze. The test run is enforced by a lock file in the scripts.

**Pooling.** For every need, the top 8 of every retrieval system are merged, duplicates removed, and shuffled with a fixed seed. The judging sheet shows the need and the passage (ticker, year, Item, text) and nothing else: no system name, no rank, no score. The mapping to systems is stored separately (`pool_key.json`) and judges do not open it. Grades: 2 = answers the need, 1 = partly relevant, 0 = not relevant. All 589 pooled pairs were judged by an LLM (Claude) working from the sheet alone, so it saw no system names or ranks and did not open `pool_key.json`. A second, independent LLM pass judged a stratified subset of 91 pairs spread over all 30 needs. No human judged any pair. This replaces the original plan of three human judges.

**Agreement.** Linearly weighted Cohen's kappa over the 91 doubly judged pairs (Cohen 1960, 1968), reported with the number of pairs. Because both passes are LLM passes, this measures consistency between two LLM passes, not agreement with a human, and the report says so. No adjudication was done: where the two passes disagree, the lower grade is used. The audit trail, including a truncation error that was found and corrected, is in `docs/report_B/audit/`.

**Metrics.** P@5, P@10, R@5, R@10, F1@5, F1@10, MAP, MRR, nDCG@5, nDCG@10 (binary gain, as in section VI-D), a graded nDCG@10 as a robustness check, and the 11-point interpolated precision-recall curve. Recall is relative to the pool. All metrics are implemented in `trademesh/evaluation/metrics.py` from the definitions of Lecture 8 and checked by hand-computed tests, including the lecture's own nDCG example.

**Significance.** With 20 queries, differences are summarised by the paired bootstrap over queries (10,000 resamples, fixed seed): mean difference, 95 percent interval, and win/loss/tie counts per query. A hypothesis is "supported" only if the interval excludes zero; the rules are fixed in `eval/hypotheses.py` before the test run and applied mechanically.

**Is the pool fair? (a threat we measure rather than assume away).** Pooled judgments favour the systems that built the pool: a passage no pooled system retrieved is treated as not relevant (Zobel 1998). Three checks are reported. (1) judged@10: the share of each system's top 10 that has a judgment; a system with a low value is scored pessimistically. (2) bpref (Buckley and Voorhees 2004), which uses only judged documents and so is robust to incomplete judgments. (3) leave-one-system-out: remove the passages that only system S contributed, re-score all systems, and report how S's score and the ordering of all systems change (Kendall tau). Our pool is built from the same systems we compare, which is the standard setting for this check.

**Honest limits.** The judge is an LLM, and the relevance labels have not been validated by a human, so any systematic LLM preference (for example for passages that read like answers, which may favour neural rankers) is a threat to the comparison; the LLM judge is also a language model like the generator in the RAG study; 20 test needs give wide intervals; the 10-K corpus covers about 20 companies, so findings may not transfer to other filings. These limits go into the report's Limitations section.

## VI. Experimentation and results

All numbers below are read from the locked test-split files (`eval/out/*_test.json`, `pool_bias.json`, `cost_table.json`). The test split was run once, after the freeze; the lock files are `eval/out/.testrun_*`. Regenerate this page with `python eval/fill_report_results.py`.

### A. Datasets

The project uses four datasets, all built by us from public sources: the filing corpus, the information needs with their relevance judgments, the unanswerable questions, and the misspelt company names. No external training data is used; the dense encoder and the cross-encoder are pre-trained models applied unchanged.

#### Filing corpus

**Source.** The most recent Form 10-K of each of 20 large US companies, downloaded from SEC EDGAR (the SEC's public filing database) with a crawler that identifies itself and waits between requests. The companies span technology (Apple, Microsoft, NVIDIA, Intel), consumer and media (Tesla, Amazon, Netflix, Walmart, Coca-Cola, PepsiCo, Disney), banks (JPMorgan, Goldman Sachs, Bank of America), energy (Exxon Mobil, Chevron), health (Pfizer, Johnson & Johnson, UnitedHealth) and aerospace (Boeing). The filing list, with accession numbers and URLs, is in `data/filings.json`.

**Processing.** Each filing's HTML is parsed, split by 10-K Item, and cut into passages; every table is kept whole as one chunk (caption, header row and rows as text). Each passage records its ticker, fiscal year, Item, kind (text or table) and page.

| Ticker | Company | Fiscal year | Period end | Passages | of which tables | Words |
|---|---|---|---|---|---|---|
| AAPL | Apple Inc. | 2026 | 2026-09-26 | 286 | 16 | 30,665 |
| MSFT | Microsoft Corporation | 2026 | 2026-06-30 | 373 | 20 | 43,973 |
| NVDA | NVIDIA Corporation | 2026 | 2026-01-25 | 406 | 25 | 48,736 |
| TSLA | Tesla, Inc. | 2025 | 2025-12-31 | 460 | 20 | 56,642 |
| AMZN | Amazon.com, Inc. | 2025 | 2025-12-31 | 334 | 19 | 39,204 |
| NFLX | Netflix, Inc. | 2025 | 2025-12-31 | 372 | 30 | 41,347 |
| JPM | JPMorgan Chase & Co. | 2025 | 2025-12-31 | 1,464 | 152 | 179,700 |
| GS | The Goldman Sachs Group, Inc. | 2025 | 2025-12-31 | 1,295 | 109 | 161,315 |
| BAC | Bank of America Corporation | 2025 | 2025-12-31 | 997 | 94 | 130,460 |
| XOM | Exxon Mobil Corporation | 2025 | 2025-12-31 | 540 | 110 | 68,667 |
| CVX | Chevron Corporation | 2025 | 2025-12-31 | 641 | 104 | 78,567 |
| WMT | Walmart Inc. | 2026 | 2026-01-31 | 394 | 34 | 49,433 |
| KO | The Coca-Cola Company | 2025 | 2025-12-31 | 634 | 23 | 81,658 |
| PEP | PepsiCo, Inc. | 2025 | 2025-12-27 | 512 | 55 | 61,348 |
| PFE | Pfizer Inc. | 2025 | 2025-12-31 | 671 | 34 | 84,618 |
| JNJ | Johnson & Johnson | 2025 | 2025-12-28 | 451 | 33 | 56,449 |
| UNH | UnitedHealth Group Incorporated | 2025 | 2025-12-31 | 355 | 24 | 44,660 |
| BA | The Boeing Company | 2025 | 2025-12-31 | 493 | 33 | 57,809 |
| DIS | The Walt Disney Company | 2025 | 2025-09-27 | 637 | 75 | 67,797 |
| INTC | Intel Corporation | 2025 | 2025-12-27 | 649 | 42 | 78,073 |
| **Total** | 20 companies | | | **11,964** | **1,052** | **1,461,121** |

Fiscal years are 2025 for 16 companies and 2026 for four (Apple, Microsoft, NVIDIA and Walmart, whose fiscal years end between January and September 2026), so every question is about the latest filing of each company only. Of the 11,964 passages, 10,912 are text and 1,052 are tables. The distribution over 10-K Items is Item 8 (financial statements) 2,717, Item 7 (management's discussion) 2,409, Item 1A (risk factors) 1,738, Item 15 (exhibits and financial statement schedules) 1,427, Item 1 (business) 1,132, unlabelled or other material 2,014, and the remaining Items (2, 5, 9A, 1C, 7A, 3) together 527. A passage has 122 words on average (median 131, longest 713); the corpus holds 1,461,121 words. The three banks (JPMorgan, Goldman Sachs, Bank of America) contribute 3,756 passages (31%), so a topical question has many similar bank passages to compete with.

#### Information needs and relevance judgments

**Needs.** 30 information needs, each a natural-language question and a short keyword query, in three types: entity (about one named company), topical (a theme across companies) and terminology (a defined term or a figure that sits in a table). There are 10 of each type; 10 needs are development needs, used for every tuning choice (3 entity, 3 topical, 4 terminology) and 20 are test needs (7, 7 and 6), scored once after the feature freeze. Five needs, all terminology, require a table. The needs are in `data/needs/needs.csv`.

**Pooling and judgments.** For every need the top 8 results of every system were merged, deduplicated and shuffled; the judging sheet shows the need and the passage and nothing else. The pools hold 11 to 30 passages per need (mean 19.6). Pooled passages were graded 0 (not relevant), 1 (partly) or 2 (answers the need) by a language model working from the sheet alone, and 91 pairs spread over all 30 needs were graded a second time by an independent pass. The relevance file `data/judgments/qrels.csv` has 680 rows (253 of grade 2, 179 of grade 1, 248 of grade 0). **No human graded a relevance pair**; a planned human spot-check was not done. Agreement between the two passes is reported in section VI-B.

**Size.** 30 needs, 20 of them test needs, is small. The cost is wide intervals, which is why every comparison is reported with a bootstrap interval and why several hypotheses are inconclusive.

#### Unanswerable questions

10 questions written by us whose answer is not in the corpus (4 development, 6 test): a future fiscal year, personal information not in a 10-K, a prediction, a company outside the corpus, a detail a 10-K does not give (for example a headcount for one country), and a question about an earlier year than the latest filing. They test abstention (section VI-L). They are easy, in the sense that none is a near miss of an answerable question; this is a limitation (section VII).

#### Misspelt company names

40 misspelt company names written by us, covering deletion (15), substitution (10), transposition (4), insertion (2), spacing errors (3) and six ticker errors; they test the entity matcher (section VI-L). Because we wrote them, they measure the typo types we thought of, not real user typing.

#### Why the corpus was not made larger

More filings or companies would not help answer any of our research questions, and would invalidate results already produced: the test run is locked, the 680 judgments pool from this corpus, the human ratings and the cached demo answers are tied to it. The corpus already exceeds the proposal's target of 6,000 to 10,000 passages. A corpus of several filing years per company would be the natural extension (section VII).

### B. Judgments and agreement

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

### C. Main comparison (RQ1), 20 test needs

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


*[Fig.: fig_ndcg_systems.png]* Mean nDCG@10 of the ten retrieval systems over the 20 test questions, as horizontal bars sorted in descending order; the column on the right gives each system's paired bootstrap difference from BM25 with its 95% interval. No per-system error bars are drawn because `retrieval_test.json` stores only paired differences against BM25 (source: `eval/out/retrieval_test.json`, identical to `data/results/retrieval_test.json`).


*[Fig.: fig_pr_curves.png]* 11-point interpolated precision-recall curves (recall 0.0 to 1.0 in steps of 0.1) for TF-IDF, BM25, dense, hybrid and hybrid + rerank, each averaged over the 20 test questions. Source: the `pr_11pt` entry of `eval/out/retrieval_test.json`.

### D. By need type (nDCG@10)

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


*[Fig.: fig_ndcg_by_type.png]* Mean nDCG@10 of BM25, hybrid and hybrid + rerank for each of the three need types (entity, topical, terminology), computed from the 20 test questions. Source: the `by_type` entry of `eval/out/retrieval_test.json`.

### E. Hypotheses H1 to H4, H6

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

### F. Ablations (RQ2, H5)

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

H5 (table chunks and numeric tokens matter for table needs) was run over all 20 needs only: dropping table chunks changes nDCG@10 by -0.001 (CI -0.002 to +0.000) and dropping numeric tokens by +0.009 (CI -0.009 to +0.030), so over all needs we find no effect of either. The analysis restricted to the five table needs, where an effect should show, was not computed: per-need ablation values were not stored, and the test run is locked. H5 is therefore neither supported nor refuted. One preprocessing result worth noting: adding the metadata prefix (company, year, section) to passages improved nDCG@10 by +0.083 (CI +0.011 to +0.170) in the best variant (no stemming, numbers off), the largest preprocessing effect we measured.

### G. Speed and cost (H7)

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

Reranking the top 30 with the cross-encoder adds p50 106.3 ms and p95 211.0 ms per query. This figure comes from the console output of `eval/cost_table.py` and is not stored in `cost_table.json`; hybrid+rerank is therefore about the hybrid latency plus this. It was measured on one laptop, once.

**Storage and start-up.** The processed passages file is 11 MB. The dense embeddings of all 11,964 passages (384 dimensions, 32-bit) take 18 MB; they are computed on the first start of the application with the ML packages installed and reused from disk afterwards. The BM25 index is built in memory in 0.64 s. The application starts in about 15 s with BM25 only and about 50 s with the dense model on a warm laptop; a cold first start took about four minutes. End-to-end answer latency was not logged, so we give no figure: a live language-model call took from a few seconds to over a minute (the client waits out the provider's rate limits), and a cached answer returns at once.


*[Fig.: fig_speed_quality.png]* Median (p50) query latency in milliseconds, with a bar extending to the p95 latency, against test-set nDCG@10 for plain TF-IDF and TF-IDF with champion lists of 200 and 50 passages per term, over an index of 11,964 passages; nDCG@10 is the mean over the 20 test questions. Sources: `eval/out/cost_table.json` (latency; the query set it was timed on is not recorded in the file) and `eval/out/retrieval_test.json` (nDCG@10).

### H. Pool-bias checks

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

### I. Does better retrieval give better answers? (RQ3, second half)

Spearman correlation between retrieval quality (nDCG@5 of the retriever behind each RAG configuration, from our LLM-assisted judgments) and the answer rating (0-2, mean of the two LLM rating passes), over the 20 test needs and the two RAG configurations (BM25-RAG and best-RAG): **rho = 0.007 over 40 (need, configuration) pairs** (`python eval/retrieval_vs_answer.py --split test`, `eval/out/retrieval_vs_answer.json`). Recomputed with the later blind human answer ratings (same nDCG@5 values, `python eval/human_score.py`): **rho = +0.000** with the first rater alone (approximate 95% interval -0.33 to +0.33) and **rho = -0.138** with the mean of both raters (approximate 95% interval -0.45 to +0.20); the conclusion is the same.

This is no detectable relationship. The approximate 95% interval is about -0.31 to +0.31 (Fisher transform, treating the 40 pairs as independent; they are not, since each need appears twice, so the true interval is wider). The result therefore does not show that retrieval quality is unrelated to answer quality. It shows only that on these 20 questions nDCG@5 did not predict the rating. Two likely reasons, both untested: the rating scale has three values with many ties, and in many cases a retrieved set containing a few good passages is enough for the generator, so differences below the top few ranks do not change the answer. The comparison that does carry information is the paired one in section VI-L (the RAG study): replacing BM25 with hybrid + rerank raised the mean answer rating by +0.30 under the two-rater human ratings (95% interval +0.125 to +0.500) and by +0.375 (interval +0.150 to +0.650) under the earlier model-assisted ratings.

### J. Reading the results

Findings, each with its interval; where the interval contains zero it is called inconclusive.

- **Hybrid with reranking is the strongest system.** nDCG@10 0.828 against 0.699 for bm25, a difference of +0.129 (95% CI +0.052 to +0.213), winning 13 needs and losing 3.
- **H1 supported:** bm25 beats tf-idf by +0.139 nDCG@10 (CI +0.064 to +0.223). Champion lists did not change this.
- **H3a supported, H3b tied:** hybrid beats bm25 (+0.085, CI +0.019 to +0.155) but is indistinguishable from dense alone (+0.002, CI -0.059 to +0.051).
- **H4 inconclusive:** the reranker's gain over hybrid is +0.044 (CI -0.012 to +0.113). The gain of hybrid+rerank over bm25 is significant; over hybrid it is not.
- **H2 and H6 inconclusive.** The per-type subsets hold 6 or 7 needs each; no conclusion about dense versus lexical retrieval by need type is supported.
- **Pseudo-relevance feedback did not help** on test in any of the 9 settings tried. On the 10 dev needs it looked slightly positive; the test result is the one that counts.
- **Stemming was not adopted.** With no metadata prefix, stemming lowers nDCG@10 by 0.066 to 0.086 on test (the interval excludes zero in three of four cases) and every stemmed dev configuration also scored below the default. With the metadata prefix stemming makes no measurable difference. The choice was made on dev before the test run.
- **Table chunks and the entity filter showed no measurable effect over all 20 test needs** (nDCG@10 differences -0.001 and -0.010). On dev they looked mildly helpful; the test data do not confirm it. The five-table-need subset was not analysed (see VI-F).
- **Pooling limits.** The pool was built from these same systems and judged by an LLM. Removing a system's unique finds changes MAP by at most 0.097 (dense, the most pool-dependent) and the system ordering is stable (tau 0.75 to 0.96), but the absolute scores should not be compared with other collections.
- **Judgments are LLM-assisted with no human validation.** All conclusions are conditional on that.

### K. Comparison with published findings

The proposal requires a table of our observed ordering against the published one. Corpora differ (BEIR and DPR use Wikipedia and web text, FinanceBench uses questions about public filings, ColPali uses ViDoRe page-image tasks), so we compare the direction of each effect, not absolute numbers.

| Claim | Published (source) | Ours | Same direction? |
|---|---|---|---|
| BM25 is a robust baseline that dense retrievers often fail to beat out of domain | BEIR (Thakur et al. 2021) | Dense is ahead of BM25 overall (nDCG@10 0.782 against 0.699; descriptive, not separately tested). BM25 minus dense on entity needs -0.092 (95% CI -0.208 to +0.019) and on terminology needs -0.091 (CI -0.324 to +0.079): both inconclusive (H2a) | No. We do not find BM25 ahead of dense, though the per-type intervals include 0 |
| Dense retrieval wins where questions paraphrase the text; BM25 wins where selective keywords or rare phrases matter (lexical overlap) | DPR (Karpukhin et al. 2020) | Topical needs: dense minus BM25 +0.068 (CI -0.055 to +0.201; 4 wins, 3 losses), inconclusive (H2b). On entity and terminology needs, where exact strings should favour BM25, dense is nominally ahead (see above) | Partly. The sign for paraphrase-like questions agrees but is inconclusive; BM25 does not win on exact-string needs |
| Fusing lexical and dense rankings helps | RRF (Cormack et al. 2009); BEIR | Hybrid minus BM25 +0.085 (CI +0.019 to +0.155; 13 wins, 4 losses), supported (H3a). Hybrid minus dense +0.002 (CI -0.059 to +0.051), tied (H3b) | Yes against BM25 (the lexical side); no gain over dense alone |
| Cross-encoder reranking improves the top of the ranking | Nogueira and Cho 2019; BEIR | Hybrid + rerank minus hybrid +0.044 (CI -0.012 to +0.113; 9 wins, 4 losses, 7 ties), inconclusive (H4); hybrid + rerank minus BM25 +0.129 (CI +0.052 to +0.213) | Same sign, but the interval includes 0, so not shown here |
| Answer quality on filings depends on retrieval quality (a weak retrieval setup left most questions wrong); chunking choices matter | FinanceBench (Islam et al. 2023: GPT-4-Turbo with a retrieval system answered incorrectly or refused 81% of 150 sampled questions); Jimeno Yepes et al. 2024 | The RAG study (H8, 20 questions, rated blind by two humans, mean of the two): best-RAG minus closed-book +0.625 (CI +0.350 to +0.900); BM25-RAG minus closed-book +0.325 (CI +0.050 to +0.575, carried by one rater). The earlier model-rated pass gave +0.525 and +0.150. Table-chunk ablation (H5, all 20 needs): dropping table chunks -0.001 nDCG@10 (CI -0.002 to +0.000), dropping numeric tokens +0.009 (CI -0.009 to +0.030); the table-need subset was not computed | Yes for retrieval that is strong enough (best-RAG); only weakly for BM25-RAG. Chunking: no effect found over all needs; the table-need subset was not analysed |
| Page-image retrieval beats text pipelines on visually rich documents | ColPali / ViDoRe (Faysse et al. 2024) | Not evaluated: the vision track is gated (only the MaxSim scoring is implemented and tested; there is no page-image encoder or results file) | Cannot compare |

Where our ordering differs from the published one, the report gives the most likely reasons (domain, query style, corpus size, 20 queries) and does not hide the difference.

### L. RAG study, abstention and entity correction

**Status:** the RAG study and the abstention study are run and filled below (test split, run once). Every number comes from a file in `eval/out/`. **Answer ratings and citation checks were made twice: first by a language model in two passes, then, blind, by two human raters; the human ratings are the primary result and the model-assisted ones are kept for comparison.** The retrieval relevance judgments remain model-assisted with no human validation, and the abstention items were written by us. See `docs/report_C/audit/README.md` for how the model-assisted rating was done and `eval/human/README.md` for the human protocol. The section prior was not evaluated (stated at the end).

#### Entity correction (run)

| Metric | Value |
|---|---|
| Typo'd inputs | 40 |
| Mapped to the right company | 39 (0.975) |
| Wrong company / no match | 0 / 1 |
| Agreement with brute-force Levenshtein | 40/40 |

By error type: deletion 15/15, insertion 2/2, spacing 3/3, substitution 10/10, transposition 4/4, ticker 5/6.

Source: `python eval/entity_eval.py` (`eval/out/entity_eval.json`), dictionary = `data/companies.json` (the same 39 of 40 as with the earlier proposal list). Failure: `MSTF` for `MSFT` (a transposition is two edits; short strings allow one). The 40 typo'd inputs were written by us, so this measures the matcher on typo types we thought of, not on real user typos.

*[Fig.: fig_entity_accuracy.png]* Share of misspelled company names resolved to the correct ticker, by typo kind, over 40 misspelled-name queries (39 correct overall); the number above each bar is correct/total for that kind. Source: the `by_kind` entry of `eval/out/entity_eval.json`.

#### RAG study: closed-book vs BM25-RAG vs best-RAG (20 test questions)

All three configurations use the same generator, `gemini-3.5-flash-lite` (a small, cheap model chosen because the free tier of larger models allows only 20 requests a day). Closed-book gets the question and no passages. BM25-RAG retrieves with our BM25 (k1 = 0.6, b = 0.9, tuned on dev). Best-RAG retrieves with the hybrid (BM25 + dense, reciprocal rank fusion) and reranks with the cross-encoder. The section prior and the score threshold were off in these runs (lambda = 0, tau = 0); the only abstention path was the model answering INSUFFICIENT EVIDENCE.

#### Human ratings (the primary result)

**Who rated.** Two project members rated independently, each blind to which system wrote an answer: Shashwat Rajan (folder `Shashwat`) and Ruchir Sharma (folder `Ruchir`). Each wrote a reference answer for each of the 20 test questions from the filings (Shashwat's average 85 words and Ruchir's 43; we had planned one to two sentences), rated all 60 answers 0, 1 or 2 against their own reference, and checked citations. Shashwat checked all 95 citations. Ruchir checked 48 of them, on 11 of the 20 questions (E02, E04, E05, E06, E08, E09, E10, T02, T04, T05 and T06; his checks for E08 and T06 are incomplete), and did not do the optional retrieval-judging step. Both saw the anonymous labels A/B/C (shuffled with a new seed, so the order differs from the model-assisted sheets) and no system names, scores or earlier ratings. Closed-book answers are still recognisable by their lack of citations, so the rating was not fully blind to the system. The headline number for each answer is the mean of the two raters, and the questions are then averaged. The scale is the one below.

**A correction that was made after the first scoring (Shashwat's T10 reference).** For question T10 Shashwat first wrote a reference stating that the filings contain no answer and rated the three answers accordingly (closed-book 0, BM25-RAG 2 for its refusal, best-RAG 0). We then found that the corpus does contain an answer: Amazon's own 10-K (passages `AMZN-2025-1-0010` and `AMZN-2025-1A-0009`) lists cloud and IT-infrastructure providers among its competitors and says that cloud competition will intensify, and Microsoft's Item 1 describes Azure's competition explicitly. He rewrote the reference from those passages and re-rated the three answers (now closed-book 0, BM25-RAG 0, best-RAG 1). **This revision was made after the first scoring had been seen**, so it is a researcher degree of freedom. When Shashwat alone had rated, it changed best-RAG minus BM25-RAG from +0.250 with interval [-0.100, +0.550] (not established) to +0.400 with interval [+0.200, +0.600]. The revision was prompted by a factual error in the reference (a filing passage he had not found), not by the ratings, and no other reference or rating was changed. Ruchir wrote his own T10 reference (Microsoft's Azure competition; it says the passages shown contain nothing about Amazon) and rated closed-book 0, BM25-RAG 0 and best-RAG 1, the same as the revised ratings. The tool, sheets and scorer are in `eval/human/`, `eval/human_app.py` and `eval/human_score.py`; the results are in `eval/out/rag_scores_human.json`.

| Configuration | Mean rating (0-2) | Questions rated 2 by both | Citation precision (citations both raters checked) | Number grounding | Verifier pass |
|---|---|---|---|---|---|
| Closed-book | 0.575 | 0 of 20 | n/a (no citations) | n/a | n/a |
| BM25-RAG | 0.900 | 0 of 20 | 16 of 16 (1.000) | 1.000 | 1.000 |
| Best-RAG (hybrid + rerank) | 1.200 | 2 of 20 (10%) | 31 of 32 (0.969) | 1.000 | 1.000 |

Each rater on their own:

| Rater | Closed-book | BM25-RAG | Best-RAG | Ratings 0 / 1 / 2 (of 60) | Citations checked |
|---|---|---|---|---|---|
| Shashwat | 0.700 | 0.800 | 1.200 | 10 / 46 / 4 | 95 (4 judged not to support their sentence) |
| Ruchir | 0.450 | 1.000 | 1.200 | 17 / 33 / 10 | 48 (all judged to support) |

The two raters agree exactly on 41 of the 60 answers (68%), are never more than one point apart (9 answers where Ruchir is higher, 10 where he is lower), and have Cohen's kappa 0.391 (unweighted). Both give best-RAG exactly 1.200, and both order the configurations best-RAG, then BM25-RAG, then closed-book. They differ most on closed-book (0.450 against 0.700) and BM25-RAG (1.000 against 0.800).

Citation precision counts a citation as supporting only if every rater who checked it said so. The 48 citations that both checked agree on 47; the one disagreement is T05 (best-RAG, P4), which Shashwat judged not to support its sentence. Shashwat's three other unsupported citations (R05 twice for BM25-RAG, T08 once for best-RAG) were not among Ruchir's 48. Ruchir's 48 cover only some of the questions, so the combined precision describes that subset; Shashwat alone found 33 of 35 (0.943) for BM25-RAG and 58 of 60 (0.967) for best-RAG. Number grounding and the verifier are automatic and do not depend on the rater.

**Paired bootstrap over the 20 questions** (10,000 resamples, seed 358; difference in the two-rater mean rating, 95% interval, questions better / worse / tied):

| Comparison | Difference | 95% interval | Better / worse / tied | Leave-one-question-out range |
|---|---|---|---|---|
| Best-RAG minus closed-book | +0.625 | [+0.350, +0.900] | 14 / 1 / 5 | +0.553 to +0.684 |
| Best-RAG minus BM25-RAG | +0.300 | [+0.125, +0.500] | 9 / 1 / 10 | +0.263 to +0.342 |
| BM25-RAG minus closed-book | +0.325 | [+0.050, +0.575] | 11 / 2 / 7 | +0.263 to +0.395 |

Each rater alone gives the same sign for every comparison. Shashwat: best-RAG minus closed-book +0.50, best-RAG minus BM25-RAG +0.40, BM25-RAG minus closed-book +0.10. Ruchir: +0.75, +0.20 and +0.55. The one comparison that depends on who rated is BM25-RAG against closed-book: the pooled interval excludes 0, but that comes from Ruchir (+0.55), while Shashwat alone found +0.10 (interval [-0.150, +0.350], which includes 0). We therefore call it weak evidence.

By question type (mean of the two raters; n = 7 entity, 6 terminology, 7 topical, so descriptive only):

| Type | Closed-book | BM25-RAG | Best-RAG |
|---|---|---|---|
| Entity | 0.43 | 1.14 | 1.43 |
| Terminology | 0.42 | 0.92 | 1.00 |
| Topical | 0.86 | 0.64 | 1.14 |

**Human against model-assisted ratings.** On the same 60 answers each human agrees exactly with the model on 47% to 58% of answers (Cohen's kappa 0.23 to 0.38; Shashwat 47% and 53%, kappa 0.23 and 0.27, Ruchir 53% and 58%, kappa 0.33 and 0.38, against pass 1 and pass 2). The human means over the three configurations are 0.90 (Shashwat) and 0.88 (Ruchir) against 1.43 (pass 1) and 1.37 (pass 2) for the model, and neither human rated higher than the model on any answer (lower on 25 to 32). Most of the gap is a human 1 where the model gave a 2. The three configurations come out in the same order in both studies. The verdicts are no longer all the same: in the model-assisted study BM25-RAG minus closed-book is not established (+0.150, interval [-0.250, +0.550]), while the two-rater human interval excludes 0. Possible reasons for the level difference, none tested: the human references were long, so "complete" was a higher bar; the model rater was lenient with fluent, cited text. Both human raters belong to the project, so the human ratings are not independent of the system's authors. The retrieval relevance judgments were still not validated by a human: nobody did the optional retrieval-judging step.

**Does better retrieval give better answers (RQ3), with human ratings.** The Spearman correlation between retrieval quality (nDCG@5 of the retriever behind each RAG configuration, from the model-assisted relevance judgments) and the two-rater mean answer rating is -0.138 over 40 (need, configuration) pairs, approximate 95% interval [-0.45, +0.20] (Fisher z; the pairs are not independent, so the true interval is wider). With Shashwat's ratings alone it was +0.000, and with the model ratings +0.007; no relation can be claimed.

*[Fig.: fig_rag_ratings.png]* Mean answer rating (scale 0 to 2) for closed-book, BM25-RAG and best-RAG over the same 20 test questions, comparing the human ratings (mean of the two raters, Shashwat Rajan and Ruchir Sharma) with the model-assisted ratings; the text box lists the human paired differences with 95% bootstrap intervals. Sources: `eval/out/rag_scores_human.json` (human ratings and paired differences) and `eval/out/rag_scores_test.json` (model-assisted ratings).

#### Model-assisted ratings (first pass, kept for comparison)

Rating scale per answer: 2 = correct and complete, 1 = partly correct or incomplete, 0 = wrong, invented, or no answer (an abstention on an answerable question counts as 0). Each answer was rated twice by the same language model in two passes (the second in shuffled order with a stricter checklist), and the two ratings were averaged. Raters saw anonymous labels A/B/C, not system names.

| Configuration | Mean rating (0-2) | Rated 2 | Citation precision | Number grounding | Verifier pass |
|---|---|---|---|---|---|
| Closed-book | 1.175 | 5 of 20 (25%) | n/a (no citations) | n/a | n/a |
| BM25-RAG | 1.325 | 10 of 20 (50%) | 35 of 35 (1.000) | 4 of 4 answers with numbers | 20 of 20 |
| Best-RAG (hybrid + rerank) | 1.700 | 14 of 20 (70%) | 58 of 60 (0.967) | 5 of 5 answers with numbers | 20 of 20 |

Citation precision is the share of cited passages that state the main claim they are cited for. Number grounding is the share of answers containing numbers in which every number appears in a cited passage; only 4 and 5 answers contain numbers, so it is nearly uninformative. The verifier passed all 20 answers in both configurations. Its first run on the test split reported 19 of 20, because its sentence splitter cut "Walmart U.S." in question R09 at the abbreviation and counted the fragment as a sentence without a citation. We found the bug when the app flagged a fully cited answer ("Mr. Musk"), fixed the splitter so that it no longer splits after abbreviations such as Mr., Dr., U.S. and e.g., and re-verified the stored answers from the cache with no new model calls. The fix changed 6 of the 55 answered outputs on dev and test (4 on dev, 2 on test); we read all six and every sentence in them carries a citation. The fix came after the test results were seen, and it changes only the verifier, not any rating. Two limits follow. After the fix the verifier flagged none of the 55 answers, so on this data it did not demonstrate catching an error (its unit tests do); and it checks that citations exist, that sentences are cited and that numbers occur in cited passages, not that a citation supports its sentence, so it cannot catch what we found in R09, where both RAG answers omitted a segment. BM25-RAG abstained on 4 of the 20 questions (T02, T04, T10, R07); best-RAG on none. Both human raters scored those four refusals 0.

**Agreement.** On the 60 double-rated answers, Cohen's kappa is 0.886 (unweighted; 0.906 linear-weighted, 0.930 quadratic-weighted). Both passes were made by the same model, and the second pass knew the first existed, so this agreement is probably optimistic and is not human validation. Citation checks agreed on 94 of 95 passages.

**Paired bootstrap over the 20 questions** (10,000 resamples, seed 358; difference in mean rating, 95% interval, questions better / worse / tied):

| Comparison | Difference | 95% interval | Better / worse / tied |
|---|---|---|---|
| Best-RAG minus closed-book | +0.525 | [+0.225, +0.850] | 10 / 2 / 8 |
| Best-RAG minus BM25-RAG | +0.375 | [+0.150, +0.650] | 7 / 0 / 13 |
| BM25-RAG minus closed-book | +0.150 | [-0.250, +0.550] | 9 / 8 / 3 |

By question type (mean rating; n = 7 entity, 6 terminology, 7 topical, so descriptive only):

| Type | Closed-book | BM25-RAG | Best-RAG |
|---|---|---|---|
| Entity | 1.21 | 1.79 | 1.86 |
| Terminology | 0.75 | 1.17 | 1.67 |
| Topical | 1.50 | 1.00 | 1.57 |

**Hypotheses fixed in advance.**
- **H8** (RAG beats closed-book on correctness and number grounding). Human ratings (mean of two raters): supported for best-RAG on correctness (+0.625, interval [+0.350, +0.900] excludes 0; better on 14 questions, worse on 1). BM25-RAG minus closed-book is +0.325 with interval [+0.050, +0.575], which excludes 0 (11 better, 2 worse, 7 tied), but only Ruchir's ratings carry it (+0.55; Shashwat alone +0.10, interval including 0), so we call it weak support. The model-assisted ratings show best-RAG better and BM25-RAG not established (+0.150, interval [-0.250, +0.550]). Number grounding could not be compared: closed-book has no citations to ground against, and only 4 to 5 answers per RAG configuration contain numbers.
- **H9** (hybrid + rerank RAG is at least as good as BM25-RAG). Human ratings (mean of two raters): **supported**: best-RAG was better on 9 questions and worse on 1 (difference +0.300, interval [+0.125, +0.500]), and it holds when any single question is dropped (+0.263 to +0.342). Each rater alone agrees in sign (Shashwat +0.40, Ruchir +0.20). The verdict of the first, single-rater scoring changed when one reference was corrected (see the T10 note above): with Shashwat's original T10 rating the same comparison was +0.250 with interval [-0.100, +0.550] and was not established. The model-assisted ratings gave the same verdict as the corrected ones. The second half of H9, that abstention raises citation precision at some cost in answered questions, was not tested as a separate comparison: the threshold tau tuned to 0, so no configuration used a score-based abstention.

**What the numbers do not show.** In the model-assisted study BM25-RAG is worse than closed-book on topical questions (1.00 against 1.50), and in the two-rater human ratings too (0.64 against 0.86), because it abstained on three of the seven topical questions it was asked to answer. Closed-book answers are often fluent but stale (for example, figures from older fiscal years than the filings in the corpus), which the rubric treats as incomplete. Table-lookup questions are a weakness of retrieval: in R09 both RAG answers left out one of the segments asked about. R07 and R09 also ran without the company filter (a defect with possessive names such as "Microsoft's", see the limitations), which may have contributed. Limits: 20 test questions; one generator, and a weak one; the human ratings come from two raters who both belong to the project, one of whom checked only about half of the citations, and the earlier model-assisted ratings from one model family; closed-book answers are recognisable by their lack of citations, so the rating was not fully blind to the system.

#### Abstention (dev items tune tau; test items give the headline)

Positive class = should abstain. The unanswerable questions are 10 questions written by the team (4 dev, 6 test); the answerable items are the 30 needs. Three rows per rule: dev (tau tuned on it), test (headline), all items (partly tuned on).

**Rule 1, retrieval score only** (best hybrid score divided by the number of content words in the query, no LLM call):

| Item set | tau | Precision | Recall | F1 | Accuracy | tp / fp / fn / tn |
|---|---|---|---|---|---|---|
| Dev: 10 needs + 4 unanswerable (tuned on) | 0.0063 | 0.400 | 1.000 | 0.571 | 0.571 | 4 / 6 / 0 / 4 |
| Test: 20 needs + 6 unanswerable | frozen | 0.385 | 0.833 | 0.526 | 0.654 | 5 / 8 / 1 / 12 |
| All 30 needs + 10 unanswerable | frozen | 0.391 | 0.900 | 0.545 | 0.625 | 9 / 14 / 1 / 16 |

**Rule 2, model verdict (INSUFFICIENT EVIDENCE) combined with the score threshold.** Tuned on dev, tau came out as 0, which turns the score threshold off, so this rule is the model verdict alone:

| Item set | tau | Precision | Recall | F1 | Accuracy | tp / fp / fn / tn |
|---|---|---|---|---|---|---|
| Dev: 10 needs + 4 unanswerable (tuned on) | 0.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 4 / 0 / 0 / 10 |
| Test: 20 needs + 6 unanswerable | frozen | 1.000 | 1.000 | 1.000 | 1.000 | 6 / 0 / 0 / 20 |
| All 30 needs + 10 unanswerable | frozen | 1.000 | 1.000 | 1.000 | 1.000 | 10 / 0 / 0 / 30 |

Command: `python eval/abstention_eval.py --retriever hybrid --reranker` (the score-only baseline: add `--score-only`; its output is `eval/out/abstention_scoreonly.json`).

**Why the score rule fails.** The score comes from the hybrid retriever before reranking. Reciprocal rank fusion scores depend on rank positions, not on how well a passage matches, so the confidences of the unanswerable questions (0.0023 to 0.0105) are mixed in with those of the answerable ones; the question about Pfizer's 1995 revenue (U09) has the highest confidence of all 40 items, tied with the answerable question R03 at the precision shown. A threshold on this score cannot separate the two groups.

**Reading the perfect score with care.** 6 of 6 correct abstentions on test has a 95% exact lower bound of about 0.54, and 26 of 26 correct decisions about 0.87. The unanswerable questions are easy: they are clearly outside the corpus (a future fiscal year, a home address, a stock forecast, a secret recipe, employees in one country), written by us. None is a near-miss such as a company and topic that the filings cover with the specific figure missing. The first run of the abstention script (the score-only rule above) was seen on the test items before we turned to the model verdict, so the comparison of the two rules was not fixed in advance; both rules are reported in full. The verdict behaviour belongs to the weaker generator used here.

#### Section prior

The prior adds lambda times a hand-set weight g(Item) to each candidate's min-max normalised score (section V-F). Following the proposal, lambda was tuned on the 10 development needs only (`python eval/tune_section_prior.py`, `eval/out/section_prior_dev.json`), with the company filter applied as in the pipeline. Mean dev nDCG@10 by lambda (0, 0.05, 0.1, 0.2, 0.3, 0.5): BM25 0.795, 0.803, 0.814, 0.787, 0.779, 0.747; hybrid 0.800, 0.806, 0.797, 0.805, 0.799, 0.760. The best values are lambda = 0.1 for BM25 (+0.019) and 0.05 for hybrid (+0.006). At lambda = 0.1 the prior changed the ranking for 4 of the 10 dev needs for BM25 (3 better, 1 worse) and 6 for hybrid (3 better, 3 worse), and large values hurt (lambda = 0.5: BM25 better on 2 needs, worse on 6). These gains are within what 10 needs can produce by chance, so we read the result as no demonstrated benefit. The test split was frozen and run with lambda = 0 before this check, so no test result depends on the prior and we did not run it on the test needs; the dev check is reported for completeness and cannot be taken as evidence on the test needs.

## VII. Conclusions and limitations

### A. Conclusions

TradeMesh answers questions over the latest 10-K filings of 20 companies with a retrieval stack written largely from the lecture formulas and a verified answer layer on top. We state the findings in the order of the research questions.

**Retrieval (RQ1, RQ2).** On the 20 test needs, hybrid retrieval with cross-encoder reranking had the highest nDCG@10 (0.828), ahead of dense retrieval (0.782), hybrid (0.784), BM25 with reranking (0.776), BM25 (0.699), pseudo-relevance feedback (0.679), TF-IDF (0.560) and Boolean search (0.364). The differences whose 95% interval excludes 0 are that BM25 beats TF-IDF (+0.139), that hybrid beats BM25 (+0.085) and that hybrid plus reranking beats BM25 (+0.129). Hybrid is tied with dense retrieval alone (+0.002), the gain of reranking over hybrid is inconclusive (+0.044, interval -0.012 to +0.113), and pseudo-relevance feedback does nothing measurable. The strongest preprocessing effect we found is the metadata prefix (company, year and Item prepended to a passage), +0.083 in its best variant; stemming hurt. Champion lists with r = 50 cut median TF-IDF latency about threefold at a MAP cost of 0.005. These findings rest on 20 test needs and model-assisted relevance judgments, so we read them as the direction of effects, not as exact magnitudes. The published ordering (BEIR) of BM25 and dense retrieval is not reproduced here: dense retrieval is nominally ahead of BM25, though the per-type intervals include 0.

**Answers (RQ3).** The query side detects misspelt company names with a bigram shortlist and Levenshtein confirmation (39 of 40 resolved) and uses the result to filter retrieval and to split comparison questions. The answer side gives a language model five labelled passages, requires citations, and checks them with a deterministic verifier; the system abstains when the model says the evidence is insufficient. On 20 test questions rated blind by two humans (the mean of the two raters), answers built on hybrid retrieval with reranking scored 1.20 on a 0 to 2 scale, against 0.575 for closed-book (difference +0.625, 95% interval +0.350 to +0.900) and 0.90 for BM25-based RAG (best-RAG minus BM25-RAG +0.300, interval +0.125 to +0.500). Best-RAG was reliably better than both. BM25-based RAG was better than closed-book only weakly (+0.325, interval +0.050 to +0.575), because only one of the two raters carries it (Ruchir +0.55, Shashwat +0.10), and it was worse than closed-book on topical questions because it abstained on three of seven. The two raters agree exactly on 41 of 60 answers (kappa 0.39), never differ by more than one point, and order the three systems the same way. A first, model-assisted rating of the same answers gave the same order at a higher level (1.70, 1.325, 1.175), with BM25-RAG over closed-book not established; each human agrees exactly with the model on only 47% to 58% of answers (kappa 0.23 to 0.38), the humans being stricter. In the first, single-rater scoring the best-RAG over BM25-RAG verdict changed when one question (T10) was corrected after the first scoring (see the limitations); with the correction, and with both raters, it holds under leave-one-question-out (+0.26 to +0.34). The model's own insufficient-evidence verdict separated all 6 unanswerable test questions from 20 answerable ones, while a threshold on the retrieval score did not (precision 0.385, recall 0.833), so the final rule is the model's verdict. Retrieval quality (nDCG@5) did not predict the answer rating (Spearman -0.14 with the two-rater human ratings, interval -0.45 to +0.20, and +0.007 with the model ratings, n = 40). Both human raters belong to the project; the relevance judgments behind the retrieval results are still model-assisted, with no human validation.

**Graph (RQ4).** The company mention graph has 20 nodes and 14 edges. PageRank correlates 0.93 with in-degree (Spearman), so it adds little beyond counting mentions (H10 not supported); HITS is degenerate on a graph this sparse.

**Vision (RQ5).** Not answered: the page-image track was gated and not built (section V-I).

**Overall.** The most useful practical result is the combination: a strong retriever matters (best-RAG against BM25-RAG), the model's own refusal is a better abstention signal than the retrieval score, and a checker that only tests presence of citations and numbers is cheap to run and honest about what it cannot catch. The most useful methodological result is negative: with 20 test needs, most pairwise retrieval differences are inconclusive, and a conclusion that the data cannot carry is better reported as inconclusive.

### B. Limitations

* **Small, hand-made test sets.** The entity test has 40 typos written by us, the RAG study has 20 questions (6 to 7 per question type) and the abstention test has 6 unanswerable test questions, all clearly outside the corpus. We report paired bootstrap intervals and call a difference reliable only when its interval excludes 0; per-type numbers are descriptive. A perfect 6 of 6 on abstention has a 95% lower bound of about 0.54.
* **Two human raters, who are not independent.** The headline answer ratings come from two project members, Shashwat Rajan and Ruchir Sharma, each blind to the system except that closed-book answers can be recognised by having no citations. Both belong to the team, so the ratings may carry their preferences. They agree exactly on 41 of 60 answers (kappa 0.39). Shashwat checked all 95 citations and Ruchir 48 (on 11 of the 20 questions), so the combined citation precision covers only those 48. The reference answers averaged 85 words (Shashwat) and 43 words (Ruchir), against a plan of one to two sentences; the longer references probably made "complete" a higher bar, and 46 of Shashwat's 60 ratings are a 1, against 33 of Ruchir's.
* **A reference was corrected after the first scoring.** For T10 Shashwat's first reference said that the filings contain no answer; we then found that Amazon's and Microsoft's own 10-Ks do answer it, and he rewrote the reference and re-rated that question's three answers. In his ratings alone the correction changed best-RAG minus BM25-RAG from +0.25 (interval -0.10 to +0.55) to +0.40 (interval +0.20 to +0.60), so the earlier single-rater verdict on that comparison hinged on one judgment. Ruchir wrote his own T10 reference independently and gave the same ratings as the corrected ones (0, 0, 1). With both raters the comparison is +0.30 (interval +0.125 to +0.50) and holds under leave-one-question-out (+0.26 to +0.34). No other rating was changed. Other references were not re-checked against the full filings, so a similar error elsewhere cannot be ruled out.
* **Model-assisted ratings and judgments remain, and disagree with the human ones.** Answers were first rated by one language model in two passes; those ratings are kept for comparison but agree with each human on only 47% to 58% of answers (kappa 0.23 to 0.38), so they should not be cited as validated. The retrieval relevance judgments are model-assisted and were not validated by a human (the optional human validation sample was not done). The audit trail for the model-assisted ratings is in `docs/report_C/audit/`; the human protocol and sheets are in `eval/human/`.
* **One small language model, one encoder.** All answers come from `gemini-3.5-flash-lite`, chosen because the free tier of larger models allows about 20 requests a day. Conclusions may not transfer to stronger models. Cached answers fix the demo, but they are tied to this model version.
* **Strict number matching.** The verifier requires a number to appear exactly (after unit normalisation). A correct but rounded or recomputed figure is flagged, and a number that is present but attached to the wrong claim is not caught. The verifier checks *presence*, not *entailment*; a citation can be real yet not support the sentence, which is why citation precision is judged separately (by LLM passes). The verifier flagged none of the 55 answered outputs on dev and test after a sentence-splitting bug was fixed, so on this data it did not demonstrate catching an error; and it cannot catch an omission such as the missing segment in question R09.
* **The retrieval score is not a usable confidence.** The score rule uses the hybrid retriever's score before reranking divided by query length. Rank fusion makes that score depend on rank positions, so unanswerable and answerable questions overlap (the 1995 Pfizer revenue question tied for the highest confidence of all 40 items). τ tuned on dev came out as 0, which turns the rule off, so abstention here is the model's own verdict. We saw the score-only numbers on the test items before choosing the verdict rule, so that choice was not fixed in advance. We did not test the cross-encoder score as a confidence.
* **Entity matcher.** Plain Levenshtein counts a transposition as two edits, so a swapped-letter ticker such as `MSTF` is missed. Names unknown to the dictionary are never detected, and a corpus word that is also a company alias can be ignored or accepted wrongly. Possessive forms of company names are also handled inconsistently: a form such as "Microsoft's" is skipped when it already occurs as a word in the filings ("Tesla's" is recognised, "Microsoft's" and "Apple's" are not), so 3 of the 16 company-naming needs (R06 on dev, R07 and R09 on test) and 3 of the 10 unanswerable questions ran without the company filter. We found this after the study and did not change the code, so that the reported results stay reproducible; it may have contributed to the weak results on the table questions R07 and R09.
* **Hand-set section prior.** The weights g(Item) were set by judgement, not learned. λ was tuned on the 10 dev needs only and gave at most +0.019 nDCG@10 there (no demonstrated benefit); it is 0 in all reported test runs, so the section prior is not part of the reported results and was never evaluated on the test needs.
* **Scope.** English, US filings, the latest 10-K of about 20 large companies, text and tables only (charts only through the gated vision track). Questions about trends over several years are out of scope.
* **Not a financial product.** The system makes no investment claims; answers are only as good as the retrieved passages and must be checked against the filing.
* **The vision track was not built.** RQ5 and H11 are unanswered (section V-I); the system cannot retrieve by chart or page image.
* **Several hypotheses are inconclusive, not refuted.** H2, H4 and H6 have intervals that include 0, and H5 was only computed over all needs, not over the five table needs, because per-need ablation values were not stored and the test run is locked. H7 is a single timing run without a significance test, and the champion-list fallback count was not stored, so we do not know how often the fallback to full postings occurred (the r = 200 latency is nearly that of full TF-IDF, which is consistent with frequent fallback).
* **Pooled judgments favour the pooled systems.** We measure this (section VI-H) but cannot remove it; a relevant passage that no system retrieved is treated as not relevant.
* **A small, sparse graph.** The company graph has 14 edges, and ten companies have no outgoing edge, so PageRank and HITS have little to work with. The Kendall correlation is computed without tie correction.
* **Untested edge cases.** Boolean OR and NOT queries over real data, wildcards inside phrases, proximity queries with three or more terms, and the skip-pointer saving and wildcard latency on the real index are covered only by small unit tests or not at all. A query containing an upper-case AND, OR or NOT is treated as a Boolean query.
* **Single-year, single-language corpus.** Only the latest 10-K of each company is indexed, so trend questions across years cannot be answered.

### C. Future work

1. A second human rater for the answer study (giving a human-to-human kappa) and a human check of a sample of the relevance judgments.
2. A larger and harder test collection: more needs, unanswerable questions that are near misses of answerable ones, and questions that need several filing years.
3. Fix the possessive-name defect in the entity matcher (a transposition-aware edit distance would also catch swapped-letter tickers such as MSTF) and re-run the affected needs.
4. Test the cross-encoder score as an abstention confidence, and stronger generators than the small model used here.
5. Evaluate the section prior on test-sized judged data (on the 10 dev needs it gained at most +0.019 nDCG@10) and learn its weights instead of setting them by hand.
6. Build and evaluate the page-image track on a GPU machine, starting with the five-page go/no-go check.

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
