# Abstract and Introduction (Role B draft)

Numbers in angle brackets are filled from the locked test run; until then they stay `TODO(run)`.

## Abstract (draft, about 150 words)

Analysts who ask questions about company filings need exact figures and the passage that supports them. We present TradeMesh, a search and question-answering system over the latest Form 10-K filings of about 20 US companies. It combines a classic IR core built from scratch (positional index, TF-IDF, BM25, query processing with spelling correction) with dense retrieval, rank fusion, cross-encoder reranking, pseudo-relevance feedback, a company mention graph, and retrieval-augmented answers that cite their sources, are checked by a verifier and abstain when the evidence is thin. We evaluate on a test collection of 30 information needs built by pooling and blind judging by three people, with paired bootstrap intervals over queries, and we state hypotheses before the test run. Main findings: TODO(run: best system, size of its gain over BM25 with interval; effect of reranking; effect of RAG over closed-book; abstention). We also report where our results agree and disagree with BEIR, DPR and FinanceBench.

## 1. Introduction (draft)

**Problem.** A 10-K is a long, structured document in which the answer to a question is usually one sentence or one table row inside a few hundred pages. Keyword search finds the right words but not always the right company or the right section; a language model answers fluently but may invent a figure. Both failures are costly when the user is reading about money.

**What we built.** TradeMesh answers questions over the most recent 10-K of about 20 companies. A user can run the classic search engine and compare retrievers side by side, or ask a question and receive an answer in which every claim carries a citation to a labelled passage, a verifier checks that the citations exist and that every number occurs in the cited text, and the system says "insufficient evidence" instead of guessing.

**Why this is a retrieval project.** The quality of the answer is bounded by the quality of the five passages given to the language model. We therefore treat retrieval as the object of study: we implement the classic rankers ourselves, add the modern ones, and ask which combination works on financial text and why.

**Research questions.** RQ1 compares Boolean/phrase, TF-IDF, BM25, dense, hybrid and hybrid with reranking. RQ2 asks which preprocessing and structural choices help on financial text (stop words, stemming, numeric tokens, table chunks, entity filter, section prior, pseudo-relevance feedback) and what champion lists trade for speed. RQ3 asks whether retrieval-grounded answers beat closed-book answers, whether better retrieval gives better answers, and whether abstention improves trustworthiness. RQ4 and RQ5 concern the filing graph and the gated vision track.

**Contributions.** (1) A working, tested retrieval stack in which every ranker is written from the lecture formulas and its behaviour can be explained term by term. (2) A pooled test collection with blind judging, agreement measured by kappa, and three explicit checks for pooling bias. (3) Hypotheses and decision rules fixed before the test run, applied mechanically, so that failed hypotheses are reported as findings. (4) A comparison of our ordering with published results (BEIR, DPR, FinanceBench, ColPali). (5) A verified-RAG pipeline that abstains, evaluated by blind ratings and automatic grounding checks (Role C).

**Related work.** See section 6 of the proposal; the references are in `references.md`.

**Outline.** Section 2 describes the data and model; sections 9.x the methods; section 10 the experiments; section 11 conclusions and limitations.
