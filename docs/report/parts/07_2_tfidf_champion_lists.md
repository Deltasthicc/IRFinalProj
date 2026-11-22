## 7.2 TF-IDF with cosine, heap top-K and champion lists

System 1 is the classical vector space model (Salton et al. 1975) in the weighting scheme described by Manning et al. 2008. For a term t and passage d, with N passages and document frequency df(t),

    w(t, d) = (1 + log10 tf(t, d)) * log10(N / df(t))      if tf > 0, else 0
    cos(q, d) = sum_t w(t, q) w(t, d) / (||q|| ||d||)

The query is weighted the same way, and query terms absent from the collection are dropped. Document vectors are length-normalised once at build time and stored as postings (term to passage to normalised weight), so a query needs only dot products. Scoring is term-at-a-time: each query term walks its postings and adds w(t, q) w(t, d) to a per-passage accumulator, and the K best are taken with a heap (heapq.nlargest). Ties go to the lower passage number.

**Champion lists.** For each term we precompute the r passages with the highest weight and, at query time, score only the union of the query terms' champion lists. Two details differ from a literal reading of the proposal: the ranking inside a list uses the length-normalised weight, not the raw w(t, d), and if the union holds fewer than K passages the query falls back to the full postings. The index counts these fallbacks, but no stored result reports them, so we cannot say how often they occurred. We build r = 50 and r = 200.

**Result (H7).** H7 states that champion lists cut latency with a small quality loss. On 11,964 passages, median latency was 0.237 ms for full TF-IDF, 0.080 ms for r = 50 and 0.228 ms for r = 200 (95th percentile 0.549, 0.122 and 0.308 ms; eval/out/cost_table.json). On the 20 test needs, nDCG@10 was 0.560 for all three systems and MAP was 0.455, 0.450 and 0.455 respectively (data/results/retrieval_test.json). So r = 50 gives about a threefold median speed-up for a MAP loss of 0.005, consistent with H7, while r = 200 gives no useful speed-up. These are single timing runs at sub-millisecond latencies and no significance test of H7 is stored, so this is supporting evidence, not a confirmed effect. All three remain well below BM25 (H1, mean nDCG@10 difference 0.139; data/results/hypotheses_test.json).

The unit tests check TF-IDF scores against a hand computation on a three-passage collection, that a short passage outranks a long one with the same term, that empty and unknown queries return nothing, and that champion lists shrink the candidate set (5 against 60 passages) and trigger the counted fallback when more results are requested than the lists hold.
