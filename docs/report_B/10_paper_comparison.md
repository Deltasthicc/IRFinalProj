# 10.10 Comparison with published findings

The proposal requires a table of our observed ordering against the published one. Corpora differ (BEIR and DPR use Wikipedia and web text, FinanceBench uses questions about public filings, ColPali uses ViDoRe page-image tasks), so we compare the direction of each effect, not absolute numbers.

| Claim | Published (source) | Ours | Same direction? |
|---|---|---|---|
| BM25 is a strong baseline that dense retrievers often fail to beat out of domain | BEIR (Thakur et al. 2021) | TODO(run: H2a, H2b) | TODO(run) |
| Dense retrieval wins where questions paraphrase the text; BM25 wins where exact strings matter | DPR (Karpukhin et al. 2020) | TODO(run: by-type table) | TODO(run) |
| Fusing lexical and dense rankings helps | RRF (Cormack et al. 2009); BEIR | TODO(run: H3) | TODO(run) |
| Cross-encoder reranking improves the top of the ranking | Nogueira and Cho 2019; BEIR | TODO(run: H4) | TODO(run) |
| Retrieval improves answers to questions about filings; chunking choices matter | FinanceBench (Islam et al. 2023); Jimeno Yepes et al. 2024 | Role C's RAG study (H8); our table-chunk ablation (H5) | TODO(run) |
| Page-image retrieval beats text pipelines on visually rich documents | ColPali / ViDoRe (Faysse et al. 2024) | Role A's vision track (H11, gated) | TODO(run / gated) |

Where our ordering differs from the published one, the report gives the most likely reasons (domain, query style, corpus size, 20 queries) and does not hide the difference.
