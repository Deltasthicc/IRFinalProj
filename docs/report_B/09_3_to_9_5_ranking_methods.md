# 9.3 to 9.5 Ranking methods

All rankers return the same `Hit(passage_id, score)` list, so every one can be evaluated, shown in the Search tab and used by the RAG pipeline without special cases.

## 9.3 BM25 (System 2), implemented from scratch

For query q and passage d, with N passages, df(t) passages containing t, tf(t,d) occurrences, |d| tokens and avgdl the mean length:

    score(q, d) = sum over t in q of  idf(t) * tf(t,d) * (k1 + 1) / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )
    idf(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

The "+1" inside the logarithm keeps idf positive for terms that occur in more than half of the passages, which matters in 10-Ks where words such as "company" are everywhere. Two properties explain why BM25 should beat plain tf-idf (H1): the factor tf / (tf + k1 ...) saturates, so the tenth occurrence of "tariff" adds little to the third, and b scales the penalty for long passages, whereas tf-idf with cosine normalisation uses one fixed length rule.

**Implementation.** Our own inverted index: one NumPy postings array (passage numbers and term frequencies) per term. A query touches only the postings of its terms (term-at-a-time), accumulates scores in an array and selects the top k with `argpartition`; ties are broken by passage number so every run is reproducible. `explain(query, passage)` returns each query term's contribution to a score, which the demo uses to show *why* a passage ranks where it does.

**Analysis choices (RQ2).** Tokens are lower-cased alphanumerics. Numbers keep their value but lose thousands separators, so `383,285` and `383285` match. Defaults: stop words removed (a short 118-word English list), no stemming, numbers kept, no metadata prefix. Each choice is one switch in `AnalyzerConfig` and the 16-way grid is run by `eval/ablation.py`. The default is a hypothesis (H5 says numeric tokens help); it is changed only if the dev split says so, and the choice is reported either way.

**Tuning.** k1 in {0.6, 0.9, 1.2, 1.5, 1.8, 2.0} and b in {0.3, 0.5, 0.75, 0.9, 1.0} are scored by mean nDCG@10 on the 10 development needs only (`eval/tune_bm25.py`); the full grid is printed so the report shows the surface, not just its maximum. With 10 queries the surface is flat and noisy, so ties go to the textbook 1.2 / 0.75. Result (`eval/out/bm25_grid_dev.json`): k1 = 0.6 and b = 0.9, with mean dev nDCG@10 of 0.795, against 0.758 for the textbook 1.2 / 0.75. The surface is flat near its top (the three best cells score 0.795, 0.794 and 0.789; the worst of the 30 cells scores 0.728), and with 10 development needs we do not read the exact optimum as meaningful. The chosen values are stored in `data/judgments/bm25_params.json` and were frozen before the test run.

## 9.4 Dense (System 3), hybrid (System 4), cross-encoder (System 5)

**Dense.** Each passage is encoded once by `all-MiniLM-L6-v2` (Reimers and Gurevych 2019; MiniLM, Wang et al. 2020) into a 384-dimension unit vector, and the query likewise; the score is the cosine, which for unit vectors is a dot product. Exact search over a matrix of a few thousand rows takes milliseconds, so no approximate index is used. Embeddings are cached on disk by a hash of the model and texts. The encoder is an interface: tests use a hashing encoder with no download, and only the real model produces reportable numbers.

**Hybrid by Reciprocal Rank Fusion** (Cormack, Clarke and Buettcher 2009):

    RRF(d) = sum over rankings r of  1 / (60 + rank_r(d))

It uses ranks only, so BM25 scores (unbounded) and cosine scores (between -1 and 1) need no calibration. We also implement CombSUM and CombMNZ on min-max normalised scores (Fox and Shaw 1994) as comparison points; RRF's k is varied in the ablation (10, 60, 100).

**Cross-encoder.** The top 30 hybrid candidates are rescored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, which reads the query and the passage together (Nogueira and Cho 2019) and so can use word-level interactions that two independent embeddings miss. It is too slow to run over the whole corpus, hence the two-stage design; the top 5 go to the RAG prompt. Scores are passed through a sigmoid so the abstention threshold tau of Role C is comparable across queries.

Why expect each step to help or not (H2 to H4): BEIR (Thakur et al. 2021) found BM25 a strong baseline that dense retrievers often fail to beat out of domain, and re-ranking with a cross-encoder to be the most consistent improvement; DPR (Karpukhin et al. 2020) found dense retrieval better on paraphrase-heavy questions and BM25 better where exact entity strings matter. Our corpus is entity- and number-heavy, so we predict a tie or BM25 win on entity and terminology needs and a dense win on topical needs.

## 9.5 Pseudo-relevance feedback (Rocchio 1971)

    q_new = alpha * q0 + beta * (1/k) * sum over d in top-k of  d_hat

q0 is the unit-length query vector and d_hat the L2-normalised BM25 term-weight vector of a feedback passage. We take k = 5, alpha = 1, beta = 0.5, keep the m = 10 strongest terms not already in the query, and use q_new's components as term weights inside the BM25 sum (original terms keep weight about 1, expansion terms count for less). The hypothesis (H6) is that PRF helps terminology needs, where the user's words differ from the filing's ("free cash flow" versus "cash provided by operations"), and can hurt entity needs, where the top passages are already about the right company and expansion adds drift. A k x m grid (k in {3, 5, 10}, m in {5, 10, 20}) is run on the dev split to show sensitivity; the headline uses the pre-stated defaults, not the best grid cell.

## What is not claimed

No claim of superiority is made before the test run. The tuning above uses the development split only, and the 20 test needs are scored once.
