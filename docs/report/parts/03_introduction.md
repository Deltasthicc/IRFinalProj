## 3. Introduction

### 3.1 Background

Every listed US company files an annual report on Form 10-K with the Securities and Exchange Commission. A 10-K is a long, semi-structured document, divided into numbered Items (business description, risk factors, management's discussion, financial statements and so on), and the answer to a typical question is one sentence or one table row inside a few hundred pages. The filings of the 20 companies in this project add up to about 1.46 million words. Two familiar tools serve the reader badly here. Keyword search finds the right words but not always the right company or section, and it returns passages, not answers. A language model answers fluently, but without access to the filing it may quote an outdated figure or invent one, and nothing in its answer tells the reader which sentence of which filing to check.

### 3.2 Importance

Both failures are costly when the reader is making a decision about money. Financial question answering is also a demanding test for retrieval: the vocabulary is repetitive (every filing talks about "risk factors", "revenue" and "competition"), the same sentence pattern appears in all companies so the company name carries most of the signal, figures sit in tables that plain text indexing handles badly, and the correct answer must be exact, not approximately right. Published work on this setting reports that retrieval-augmented systems still fail often on filings (FinanceBench, Islam et al. 2023, reports that GPT-4-Turbo with a retrieval system answered incorrectly or refused 81% of 150 sampled questions).

### 3.3 Motivation

The quality of a retrieval-augmented answer is bounded by the five or so passages handed to the language model. We therefore treat retrieval as the object of study. We implement the classic rankers ourselves from the lecture formulas, so that their behaviour can be explained term by term, add the modern ones (dense retrieval, rank fusion, cross-encoder reranking), and ask which combination works on financial text and why. On top of the retrieval layer we ask a second question: can an answer be made checkable, so that a reader can see which passage supports each claim, a program can test that every number in the answer occurs in the cited text, and the system can decline when the evidence is thin?

### 3.4 What we built

TradeMesh answers questions over the most recent 10-K of 20 companies. A user can run the search engine and compare retrievers side by side, or ask a question and receive an answer in which every claim carries a citation to a labelled passage, a verifier checks that the citations exist and that every number occurs in the cited text, and the system says "insufficient evidence" instead of guessing. A misspelt company name ("Telsa") is corrected before retrieval, and a question that names two companies is split into one search per company. A six-tab web application (Ask, Search, Graph, Pages, Evaluation, System) and a Docker image make the system runnable and demonstrable.

### 3.5 Contributions

1. A working, tested retrieval stack in which every ranker is written from the lecture formulas: positional index with skip pointers, Boolean, phrase and wildcard search, TF-IDF with champion lists, BM25, pseudo-relevance feedback, and our own spelling and company-name matcher. Dense encoder and cross-encoder are pre-trained models used as components.
2. A pooled test collection of 30 information needs with blind, LLM-assisted relevance judgments, consistency between two LLM passes measured by Cohen's kappa, and three explicit checks for pooling bias.
3. Hypotheses and decision rules fixed before the test run and applied mechanically, so that failed hypotheses are reported as findings and not explained away.
4. A comparison of our ordering of methods with published results (BEIR, DPR, RRF, BERT reranking, FinanceBench, ColPali), reported by direction of effect because the corpora differ.
5. A verified-RAG pipeline with a deterministic number and citation checker and a rule for abstaining, evaluated by blind ratings from one human and, for comparison, two model-assisted passes; the report states where the two disagree and discloses a reference answer that was corrected after the first scoring.
6. A company mention graph with PageRank and HITS, and an honest finding that on this corpus it adds little beyond counting mentions.

### 3.6 Organisation of the report

The rest of the paper is organised as follows. Section 4 reviews related work and states what is new. Section 5 gives the objectives, research questions and hypotheses. Section 6 presents the proposed model. Section 7 describes the methods and formulas (7.1 to 7.10). Section 8 describes the datasets and reports the experiments, results and a comparison with published findings (8.1 to 8.12). Section 9 gives the conclusions and limitations, and the references follow.
