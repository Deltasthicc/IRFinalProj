# 8. Experiments and results: RAG study, abstention and entity correction (Role C)

**Status:** the RAG study and the abstention study are run and filled below (test split, run once). Every number comes from a file in `eval/out/`. **All answer ratings, citation checks and retrieval judgments in this project are LLM-assisted; no human validated them.** See `docs/report_C/audit/README.md` for how the rating was done and its limits. The section prior was not evaluated (stated at the end).

## Entity correction (run)

| Metric | Value |
|---|---|
| Typo'd inputs | 40 |
| Mapped to the right company | 39 (0.975) |
| Wrong company / no match | 0 / 1 |
| Agreement with brute-force Levenshtein | 40/40 |

By error type: deletion 15/15, insertion 2/2, spacing 3/3, substitution 10/10, transposition 4/4, ticker 5/6.

Source: `python eval/entity_eval.py` (`eval/out/entity_eval.json`), dictionary = Role A's `data/companies.json` (the same 39 of 40 as with the earlier proposal list). Failure: `MSTF` for `MSFT` (a transposition is two edits; short strings allow one). The 40 typo'd inputs were written by us, so this measures the matcher on typo types we thought of, not on real user typos.

## RAG study: closed-book vs BM25-RAG vs best-RAG (20 test questions)

All three configurations use the same generator, `gemini-3.5-flash-lite` (a small, cheap model chosen because the free tier of larger models allows only 20 requests a day). Closed-book gets the question and no passages. BM25-RAG retrieves with our BM25 (k1 = 0.6, b = 0.9, tuned on dev). Best-RAG retrieves with the hybrid (BM25 + dense, reciprocal rank fusion) and reranks with the cross-encoder. The section prior and the score threshold were off in these runs (lambda = 0, tau = 0); the only abstention path was the model answering INSUFFICIENT EVIDENCE.

Rating scale per answer: 2 = correct and complete, 1 = partly correct or incomplete, 0 = wrong, invented, or no answer (an abstention on an answerable question counts as 0). Each answer was rated twice by the same language model in two passes (the second in shuffled order with a stricter checklist), and the two ratings were averaged. Raters saw anonymous labels A/B/C, not system names.

| Configuration | Mean rating (0-2) | Rated 2 | Citation precision | Number grounding | Verifier pass |
|---|---|---|---|---|---|
| Closed-book | 1.175 | 5 of 20 (25%) | n/a (no citations) | n/a | n/a |
| BM25-RAG | 1.325 | 10 of 20 (50%) | 35 of 35 (1.000) | 4 of 4 answers with numbers | 20 of 20 |
| Best-RAG (hybrid + rerank) | 1.700 | 14 of 20 (70%) | 58 of 60 (0.967) | 5 of 5 answers with numbers | 20 of 20 |

Citation precision is the share of cited passages that state the main claim they are cited for. Number grounding is the share of answers containing numbers in which every number appears in a cited passage; only 4 and 5 answers contain numbers, so it is nearly uninformative. The verifier passed all 20 answers in both configurations. Its first run on the test split reported 19 of 20, because its sentence splitter cut "Walmart U.S." in question R09 at the abbreviation and counted the fragment as a sentence without a citation. We found the bug when the app flagged a fully cited answer ("Mr. Musk"), fixed the splitter so that it no longer splits after abbreviations such as Mr., Dr., U.S. and e.g., and re-verified the stored answers from the cache with no new model calls. The fix changed 6 of the 55 answered outputs on dev and test (4 on dev, 2 on test); we read all six and every sentence in them carries a citation. The fix came after the test results were seen, and it changes only the verifier, not any rating. Two limits follow. After the fix the verifier flagged none of the 55 answers, so on this data it did not demonstrate catching an error (its unit tests do); and it checks that citations exist, that sentences are cited and that numbers occur in cited passages, not that a citation supports its sentence, so it cannot catch what we found in R09, where both RAG answers omitted a segment. BM25-RAG abstained on 4 of the 20 questions (T02, T04, T10, R07); best-RAG on none.

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
- **H8** (RAG beats closed-book on correctness and number grounding): supported for best-RAG on correctness (the interval excludes 0). Not shown for BM25-RAG (interval includes 0). Number grounding could not be compared: closed-book has no citations to ground against, and only 4 to 5 answers per RAG configuration contain numbers.
- **H9** (hybrid + rerank RAG is at least as good as BM25-RAG): supported. Best-RAG was never rated below BM25-RAG on any question. The second half of H9, that abstention raises citation precision at some cost in answered questions, was not tested as a separate comparison: the threshold tau tuned to 0, so no configuration used a score-based abstention.

**What the numbers do not show.** BM25-RAG is worse than closed-book on topical questions (1.00 against 1.50) because it abstained on three of the seven topical questions it was asked to answer. Closed-book answers are often fluent but stale (for example, figures from older fiscal years than the filings in the corpus), which the rubric treats as incomplete. Table-lookup questions are a weakness of retrieval: in R09 both RAG answers left out one of the segments asked about. Limits: 20 test questions; one generator, and a weak one; LLM-only ratings from one model family; closed-book answers are recognisable by their lack of citations, so the rating was not fully blind to the system.

## Abstention (dev items tune tau; test items give the headline)

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

## Section prior

Not evaluated. The mechanism is implemented and unit-tested (`query/section_prior.py`), but we did not tune or test λ; λ = 0 in every reported run, so no result in this report depends on it.
