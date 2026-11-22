# 10.11 Experiments and results: RAG study, abstention and entity correction (Role C)

**Status:** the RAG study and the abstention study are run and filled below (test split, run once). Every number comes from a file in `eval/out/`. **Answer ratings and citation checks were made twice: first by a language model in two passes, then, blind, by two human raters; the human ratings are the primary result and the model-assisted ones are kept for comparison.** The retrieval relevance judgments (Report B) remain model-assisted with no human validation, and the abstention items were written by us. See `docs/report_C/audit/README.md` for how the model-assisted rating was done and `eval/human/README.md` for the human protocol. The section prior was not evaluated (stated at the end).

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

### Human ratings (the primary result)

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

### Model-assisted ratings (first pass, kept for comparison)

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

The prior adds lambda times a hand-set weight g(Item) to each candidate's min-max normalised score (section 7.6). Following the proposal, lambda was tuned on the 10 development needs only (`python eval/tune_section_prior.py`, `eval/out/section_prior_dev.json`), with the company filter applied as in the pipeline. Mean dev nDCG@10 by lambda (0, 0.05, 0.1, 0.2, 0.3, 0.5): BM25 0.795, 0.803, 0.814, 0.787, 0.779, 0.747; hybrid 0.800, 0.806, 0.797, 0.805, 0.799, 0.760. The best values are lambda = 0.1 for BM25 (+0.019) and 0.05 for hybrid (+0.006). At lambda = 0.1 the prior changed the ranking for 4 of the 10 dev needs for BM25 (3 better, 1 worse) and 6 for hybrid (3 better, 3 worse), and large values hurt (lambda = 0.5: BM25 better on 2 needs, worse on 6). These gains are within what 10 needs can produce by chance, so we read the result as no demonstrated benefit. The test split was frozen and run with lambda = 0 before this check, so no test result depends on the prior and we did not run it on the test needs; the dev check is reported for completeness and cannot be taken as evidence on the test needs.
