# 9.10 Evaluation protocol (Role B)

This section states, before any result exists, how retrieval is scored, so that a reader can see nothing was tuned on the test set.

**Collection.** 30 information needs (10 entity, 10 topical, 10 terminology; 5 of the terminology needs require a table), each with a short query. 10 are development needs, used for every choice; 20 are test needs, scored once after the feature freeze. The test run is enforced by a lock file in the scripts.

**Pooling.** For every need, the top 8 of every retrieval system are merged, duplicates removed, and shuffled with a fixed seed. The judging sheet shows the need and the passage (ticker, year, Item, text) and nothing else: no system name, no rank, no score. The mapping to systems is stored separately (`pool_key.json`) and judges do not open it. Grades: 2 = answers the need, 1 = partly relevant, 0 = not relevant. Each person judges about 300 passages; 10 needs are judged by two people.

**Agreement.** Linearly weighted Cohen's kappa over the doubly judged passages (Cohen 1960, 1968), reported with the number of pairs. Disagreements are adjudicated in a short meeting and the result is stored in `adjudicated.csv`; without adjudication the lower of the two grades is used, and the report says which was done.

**Metrics.** P@5, P@10, R@5, R@10, F1@5, F1@10, MAP, MRR, nDCG@5, nDCG@10 (binary gain, as in section 10.3), a graded nDCG@10 as a robustness check, and the 11-point interpolated precision-recall curve. Recall is relative to the pool. All metrics are implemented in `trademesh/evaluation/metrics.py` from the definitions of Lecture 8 and checked by hand-computed tests, including the lecture's own nDCG example.

**Significance.** With 20 queries, differences are summarised by the paired bootstrap over queries (10,000 resamples, fixed seed): mean difference, 95 percent interval, and win/loss/tie counts per query. A hypothesis is "supported" only if the interval excludes zero; the rules are fixed in `eval/hypotheses.py` before the test run and applied mechanically.

**Is the pool fair? (a threat we measure rather than assume away).** Pooled judgments favour the systems that built the pool: a passage no pooled system retrieved is treated as not relevant (Zobel 1998). Three checks are reported. (1) judged@10: the share of each system's top 10 that has a judgment; a system with a low value is scored pessimistically. (2) bpref (Buckley and Voorhees 2004), which uses only judged documents and so is robust to incomplete judgments. (3) leave-one-system-out: remove the passages that only system S contributed, re-score all systems, and report how S's score and the ordering of all systems change (Kendall tau). Our pool is built from the same systems we compare, which is the standard setting for this check.

**Honest limits.** The three judges are also the system builders; the sheets hide system names and order, but a judge may still guess; 20 test needs give wide intervals; the 10-K corpus covers about 20 companies, so findings may not transfer to other filings. These limits go into the report's Limitations section.
