## 7. Methodology

This section describes the algorithms and formulas. The retrieval systems are numbered as in the experiments; each is a separate, switchable component with the same interface (query in, ranked list of passage numbers out), so that any of them can be run side by side in the application and in the evaluation scripts.

| System | Name in results | Section |
|---|---|---|
| Unranked Boolean/phrase search | `boolean` | 7.1 |
| TF-IDF with cosine, heap top-K | `tfidf` | 7.2 |
| TF-IDF with champion lists (r = 50, r = 200) | `tfidf-champ-50`, `tfidf-champ-200` | 7.2 |
| BM25 | `bm25` | 7.3 |
| BM25 with pseudo-relevance feedback | `bm25+prf` | 7.5 |
| Dense bi-encoder | `dense` | 7.4 |
| Hybrid: BM25 and dense fused by reciprocal rank fusion | `hybrid` | 7.4 |
| Cross-encoder reranking of BM25 or of hybrid | `bm25+rerank`, `hybrid+rerank` | 7.4 |

Around the retrievers sit the query processor with spelling-tolerant company matching, an optional section prior and table chunks (7.6), the company mention graph (7.7), the retrieval-augmented answer stage with its verifier and abstention rule (7.8), the gated vision track (7.9), and the evaluation protocol (7.10).

**Notation.** N is the number of passages (11,964), tf(t, d) the frequency of term t in passage d, df(t) its document frequency, |d| the length of d in tokens and avgdl the average length. Passages are the retrieval unit throughout: a text chunk of a 10-K Item or one whole table.

**Passage construction.** Each filing is split by 10-K Item into passages (mean 122 words, median 131, longest 713); every table is kept as one chunk (caption, header row and rows as text), and each passage carries its ticker, fiscal year, Item and page. Section 8.1 describes the resulting corpus.

**Reproducibility.** The test split is run once, after the feature freeze, and a lock file in the scripts prevents a second run from overwriting the result; the BM25 parameters k1 and b were tuned on the 10 development needs only, the other settings are standard defaults, and variations of them are reported as ablations (section 8.6), not used to choose the system. Language-model calls use temperature 0 and a disk cache keyed by the model, prompt and parameters.
