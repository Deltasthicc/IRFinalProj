# TradeMesh

Search and cited question answering over the latest 10-K filings of about 20 companies
(CSD358 Information Retrieval, Monsoon 2026). This is the project skeleton: the shared
contract, the folder layout and the workflow. Each role adds its code on its own branch.

## Team and roles

| Role | Owner | Scope |
|---|---|---|
| A | Tanish Shitanshu | data, crawler/parser, positional index, Boolean/phrase/wildcard, TF-IDF, company graph, vision track |
| B | Ruchir Sharma | BM25, dense, hybrid (RRF), cross-encoder, test collection, metrics, report lead |
| C | Shashwat Rajan | query and entities, RAG with verifier and abstention, Streamlit app, Docker |

## Layout

```
src/trademesh/types.py   shared Passage, Hit, Retriever, Reranker (the contract between roles)
data/                    passages.jsonl, companies.json (from A), data/needs (from C)
eval/                    evaluation scripts
docs/                    report drafts and figures
tests/                   pytest, offline
requirements.txt         runtime dependencies (pinned)
.env.example             settings; copy to .env (git-ignored)
```

## Contract between roles

* `data/passages.jsonl`: one `Passage` (see `types.py`) per line. `data/companies.json`: a list of `{ticker, name, aliases}`.
* Role A may provide `trademesh.index.build_retrievers(passages) -> {name: Retriever}`.
* Role B may provide `trademesh.ranking.build_retrievers(passages)` and `trademesh.ranking.build_reranker(passages)`.

## Workflow

* Work on a branch named `<role letter>/<topic>` (for example `a/positional-index`) and open a pull request into `main`.
* Keep commits for meaningful changes, not every edit.
* Never commit secrets or `.env`. Run `python -m pytest -q` before opening a pull request.
