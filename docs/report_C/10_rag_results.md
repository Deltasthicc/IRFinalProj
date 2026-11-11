# 8. Experiments and results: RAG study, abstention and entity correction (Role C)

**Status: the RAG and abstention numbers below have NOT been run yet.** They need Role A's corpus, Role B's retrievers and reranker, a real LLM key, and the human ratings. Every `TODO(run)` is filled by the script named next to it; no number here is estimated or invented.

## Entity correction (run)

| Metric | Value |
|---|---|
| Typo'd inputs | 40 |
| Mapped to the right company | 39 (0.975) |
| Wrong company / no match | 0 / 1 |
| Agreement with brute-force Levenshtein | 40/40 |

Source: `python eval/entity_eval.py` (`eval/out/entity_eval.json`), dictionary = proposal company list with our aliases. Failure: `MSTF` for `MSFT` (a transposition is two edits; short strings allow one). TODO(run): repeat with Role A's `data/companies.json`.

## RAG study: closed-book vs BM25-RAG vs best-RAG (20 test questions)

| Configuration | Mean rating (0-2) | Rated 2 | Citation precision | Number grounding |
|---|---|---|---|---|
| Closed-book | TODO(run) | TODO(run) | n/a (no citations) | n/a |
| BM25-RAG | TODO(run) | TODO(run) | TODO(run) | TODO(run) |
| Best-RAG (hybrid + rerank) | TODO(run) | TODO(run) | TODO(run) | TODO(run) |

Cohen's κ between the two raters on the double-rated answers: TODO(run). Commands: `python eval/rag_run.py --split test`, `python eval/rag_sheets.py --split test`, then the humans fill `eval/out/rating_sheet.csv` and `eval/out/citation_sheet.csv`, then `python eval/rag_score.py --split test`.

Hypotheses fixed in advance: **H8** RAG beats closed-book on correctness and number grounding; **H9** hybrid + rerank RAG ≥ BM25 RAG, and abstention raises citation precision at some cost in answered questions. Failed hypotheses will be reported as findings.

## Abstention (dev items tune τ; test items give the headline)

| Item set | τ | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Dev: 10 dev needs + 4 dev unanswerable (tuned on) | TODO(run) | TODO(run) | TODO(run) | TODO(run) | TODO(run) |
| Test: 20 test needs + 6 test unanswerable | frozen | TODO(run) | TODO(run) | TODO(run) | TODO(run) |
| All 30 needs + 10 unanswerable (partly tuned on) | frozen | TODO(run) | TODO(run) | TODO(run) | TODO(run) |

Command: `python eval/abstention_eval.py`. Positive class = should abstain.

## Section prior

λ tuned on the 10 dev queries by nDCG@10: TODO(run) (`tune_lambda` in `query/section_prior.py`, needs Role B's judgments file).
