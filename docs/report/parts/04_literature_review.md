## 4. Literature review

We review the work our design rests on in six groups: classical ranking and query expansion, neural and hybrid retrieval, retrieval-augmented generation and its evaluation, financial question answering, link analysis, and evaluation methodology. Every paper cited here was checked against the publisher, the ACL Anthology, arXiv or at least two bibliographic databases; where only the abstract was read, the text below claims no more than the abstract does.

### 4.1 Classical ranking and query expansion

Salton, Wong and Yang (1975) represent documents and queries as weighted term vectors and rank by their similarity; our TF-IDF system is this vector space model with cosine similarity in the logarithmic weighting taught in the course (Manning et al. 2008). Robertson and Zaragoza (2009) survey the probabilistic relevance framework behind BM25, which adds term-frequency saturation (k1) and document-length normalisation (b) to an inverse-document-frequency weight; BM25 remains the standard strong lexical baseline and is the reference system in our experiments. Rocchio (1971) introduced relevance feedback by moving the query vector towards judged relevant documents; with the top results assumed relevant it becomes pseudo-relevance feedback, which we evaluate (it does not help on our collection).

### 4.2 Neural and hybrid retrieval

Karpukhin et al. (2020) show that a dense bi-encoder trained for question answering beats BM25 on most open-domain datasets, while noting that lexical overlap favours BM25 on one of them and that dense models struggle with rare, salient phrases. Reimers and Gurevych (2019) make sentence embeddings practical with siamese BERT networks, and Wang et al. (2020) compress such encoders with deep self-attention distillation; our dense retriever uses a MiniLM-based sentence encoder of this family. Khattab and Zaharia (2020) propose late interaction (ColBERT), encoding query and document separately and scoring them by a cheap token-level interaction. Nogueira and Cho (2019) show that a BERT cross-encoder re-ranking a BM25 candidate list improves passage ranking, which motivates our reranking stage. Fox and Shaw (1994) study how to combine several searches (CombSUM, CombMNZ), and Cormack, Clarke and Buettcher (2009) show that reciprocal rank fusion, which uses only ranks, outperforms Condorcet fusion and individual rank-learning methods; our hybrid system fuses BM25 and dense rankings this way, and we test the alternatives. Thakur et al. (2021) assemble BEIR, a zero-shot benchmark over heterogeneous domains, and find that BM25 is a robust baseline, that re-ranking and late-interaction models give the best average results at a high computational cost, and that dense and sparse retrievers are cheaper but often under-perform; this is the published ordering we compare against in section 8.11.

### 4.3 Retrieval-augmented generation and its evaluation

Lewis et al. (2020) combine a pre-trained generator with a non-parametric memory that is retrieved at question time, the formulation that our answer stage follows: retrieve passages, then generate from them. Es et al. (2023) propose RAGAS, a reference-free set of metrics for scoring a RAG pipeline (retrieval quality, faithfulness of the answer to the context, relevance of the answer). Metrics of that kind use a language model as judge. We chose a different route for the two properties that matter most for financial answers: a deterministic checker that tests whether every cited passage exists and every number occurs in the cited text, and blind human ratings of answer quality against a reference written from the filings, kept separate from the model-assisted ratings so that the two can be compared.

### 4.4 Question answering over financial filings

Islam et al. (2023) introduce FinanceBench, questions about public filings with reference answers, and report that 16 model configurations on a 150-case sample fail often: GPT-4-Turbo with a retrieval system answered incorrectly or refused 81% of the questions. This is the motivation for our checkable answers, although our corpus, questions and models differ, so we cannot compare absolute numbers. Jimeno Yepes et al. (2024) study how financial reports should be chunked for retrieval-augmented generation and argue for chunks that follow the structure of the report rather than a fixed size; we follow the same idea by splitting by 10-K Item and keeping every table as one chunk. Faysse et al. (2024) present ColPali, which retrieves document pages directly from page images with a vision-language model and late interaction, and a benchmark (ViDoRe) on which it outperforms text pipelines for visually rich documents. We implement the MaxSim scoring from this line of work but did not build the page-image track (section 7.9).

### 4.5 Link analysis

Brin and Page (1998) describe PageRank, a ranking based on the link structure of a graph, computed by power iteration with a damping factor, and Kleinberg (1999) gives HITS, which assigns each node a hub and an authority score. We apply both to a graph in which companies are linked by mentions in each other's filings. The graph is small and sparse, and our result is that PageRank adds little to simple in-degree there (section 7.7).

### 4.6 Evaluation methodology

Järvelin and Kekäläinen (2002) define cumulated-gain measures for graded relevance, of which nDCG is the one we report as the headline. Zobel (1998) examines pooled test collections and finds that relative system performance appears reliable but that recall is overestimated because relevant documents outside the pool are missed; Buckley and Voorhees (2004) propose bpref, which uses only judged documents and so is robust to incomplete judgments. Both motivate the pool-bias checks in section 8.8. Cohen (1960, 1968) gives kappa and weighted kappa for agreement between judges, and Efron and Tibshirani (1993) the bootstrap, which we use over questions because 20 test questions are too few for conventional tests.

### 4.7 What is new here

None of the components is new, and we do not claim that they are. What this project adds is their combination and the way it is evaluated, in a setting where the literature mostly reports one component at a time:

* a single system in which a from-scratch classic core (index, TF-IDF, BM25, spelling-tolerant company matching) sits beside the neural stages, so that all ten retrieval configurations are compared on the same financial passages with the same judgments;
* a deterministic verifier that checks citations and numbers against the cited text, with abstention based on the model's own verdict, and an explicit negative result: a score threshold on the retrieval confidence does not separate answerable from unanswerable questions;
* an evaluation discipline new to most course projects: hypotheses and decision rules fixed before the test run, paired bootstrap intervals, pool-bias checks, and a blind human rating whose disagreement with the model-assisted rating is reported instead of hidden;
* a documented defect and a documented correction (a reference answer that was wrong and was fixed after the first scoring), with both results reported.

The limits on this claim are in section 9: the collections are small, the relevance judgments are model-assisted, and the human ratings come from a single rater.
