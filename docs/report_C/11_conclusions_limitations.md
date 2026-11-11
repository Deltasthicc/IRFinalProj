# 9. Conclusions and limitations

## Conclusions

TradeMesh combines a search engine built from scratch with a verified question-answering layer. The query side detects misspelt company names with a bigram shortlist and Levenshtein confirmation (39 of 40 typo'd names resolved correctly in our test) and uses the result to filter retrieval and to split comparison questions. The answer side gives a language model five labelled passages, requires citations, and checks them with a deterministic verifier that flags invented citations and numbers that do not occur in the cited text; the system abstains when the model says the evidence is insufficient or when the best score is below a threshold tuned on development queries. Whether grounding beats closed-book answering on this corpus, and whether abstention improves trustworthiness, are answered by the RAG study (section 8); the results are TODO(run) until the study has been executed and rated.

## Limitations

* **Small, hand-made test sets.** The entity test has 40 typos written by us, the RAG study has 20 questions and the abstention test has 10 unanswerable questions. Differences between configurations can be noise, so we report per-question outcomes and avoid significance claims.
* **Human ratings by the authors.** Answers are rated by two of us, blind to the system, but we are not independent of the system. Agreement is measured with Cohen's κ on a double-rated subset only.
* **One language model, one encoder.** Conclusions may not transfer to other models. Cached answers fix the demo, but they are tied to one model version.
* **Strict number matching.** The verifier requires a number to appear exactly (after unit normalisation). A correct but rounded or recomputed figure is flagged, and a number that is present but attached to the wrong claim is not caught. The verifier checks *presence*, not *entailment*; a citation can be real yet not support the sentence, which is why citation precision is judged by people.
* **Abstention score is crude.** Until the reranker score is used, the confidence is the best first-stage score divided by query length, which is not calibrated across retrievers. τ is tuned on only 10 development needs and a few unanswerable questions.
* **Entity matcher.** Plain Levenshtein counts a transposition as two edits, so a swapped-letter ticker such as `MSTF` is missed. Names unknown to the dictionary are never detected, and a corpus word that is also a company alias can be ignored or accepted wrongly.
* **Hand-set section prior.** The weights g(Item) were set by judgement, not learned, and λ is tuned on 10 queries.
* **Scope.** English, US filings, the latest 10-K of about 20 large companies, text and tables only (charts only through the gated vision track). Questions about trends over several years are out of scope.
* **Not a financial product.** The system makes no investment claims; answers are only as good as the retrieved passages and must be checked against the filing.
