# TradeMesh

Search and cited question answering over the latest 10-K filings of about 20 companies (CSD358 Information Retrieval, Monsoon 2026).
This repository currently holds the **query, RAG and app** part (Role C). The data, indexes and vision track (Role A) and the rankers and evaluation (Role B) plug in through the interfaces below.

## Run it

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows Git Bash; on Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # then choose an LLM provider and add a key
streamlit run src/trademesh/app/streamlit_app.py
```

or with Docker (packaging only):

```bash
docker compose up --build          # http://localhost:8501
docker compose --profile local-llm up --build    # also starts a local Ollama
```

Without Role A's `data/passages.jsonl` and `data/companies.json` the app runs on a tiny **test fixture of invented text** and says so in a warning banner. Without a key (`LLM_PROVIDER=stub`) answers are extractive quotes. Neither can produce a reportable number: the eval scripts refuse to run on them unless you pass `--allow-stub` (dry runs go to `eval/out/dryrun/`).

## Tests

```bash
python -m pytest -q        # offline, no network, no key
```

## What is here

| Path | What |
|---|---|
| `src/trademesh/types.py` | shared `Passage`, `Hit`, `Retriever`, `Reranker` (the contract between A, B, C) |
| `src/trademesh/query/entities.py` | company names with spelling tolerance: bigram shortlist (Jaccard) + Levenshtein, from scratch |
| `src/trademesh/query/section_prior.py` | `final = base + lambda * g(Item)` and dev tuning of lambda |
| `src/trademesh/rag/llm.py` | provider-agnostic LLM client (Anthropic, OpenAI-compatible incl. Ollama), temperature 0, disk cache, call log |
| `src/trademesh/rag/prompt.py` | context builder `[P1]..[P5]`, prompt, closed-book prompt |
| `src/trademesh/rag/verifier.py` | cited ids exist, every number occurs in a cited passage |
| `src/trademesh/rag/abstain.py` | abstention rule, tau tuning, P/R/F1/accuracy |
| `src/trademesh/rag/pipeline.py` | entities -> retrieve -> context -> LLM -> verify -> abstain; compare mode |
| `src/trademesh/app/` | Streamlit app (Ask, Search, Filing graph, Pages) and its testable back end |
| `src/trademesh/dev_stub.py` | **dev-only** in-memory BM25, to be dropped when B's BM25 lands |
| `data/needs/` | 30 information needs (10 dev, 20 test), 10 unanswerable questions, 40 typo'd company names |
| `eval/` | `entity_eval.py`, `rag_run.py`, `rag_sheets.py`, `rag_score.py`, `abstention_eval.py` |
| `docs/report_C/` | report drafts for Role C's sections and the model figure |
| `tests/` | pytest suite and the clearly marked fixture |

## Interfaces for the other roles

* Types: `src/trademesh/types.py`. Field names match the `data/passages.jsonl` lines (one `Passage` per line).
* **Role A** may provide `trademesh.index.build_retrievers(passages) -> {name: Retriever}`. **Role B** may provide `trademesh.ranking.build_retrievers(passages)` and `trademesh.ranking.build_reranker(passages)`. They are discovered automatically (`src/trademesh/retrievers.py`); the app and eval scripts then stop using the stub.
* Optional app data: `data/graph/company_graph.json` (nodes with `ticker, pagerank, hub, authority, in_degree`; edges `source -> target`) and `data/vision/results.json`. Formats are documented in `src/trademesh/app/backend.py`.

## LLM settings (Gemini, Anthropic, OpenAI-compatible, Ollama)

Using Gemini: get a key at https://aistudio.google.com/apikey, then in `.env` set
`LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...` (optionally `LLM_MODEL=`, default `gemini-2.5-flash`).
The key stays in `.env`, which is git-ignored. See `.env.example` for all options. Every call is cached under `LLM_CACHE_DIR` by the hash of (model, prompt, parameters) and logged to `calls.jsonl`, so the demo can run offline from the cache (`LLM_OFFLINE=1`). The eval scripts print the number of live (uncached) calls. Keys live only in `.env`, which is git-ignored.

## Evaluation (Role C)

```bash
python eval/entity_eval.py                          # 40 typo'd names -> eval/out/entity_eval.json
python eval/abstention_eval.py                      # tau tuned on dev, headline on test
python eval/rag_run.py --split test                 # closed-book vs BM25-RAG vs best-RAG
python eval/rag_sheets.py --split test              # blind rating sheet + separate key + citation sheet
python eval/rag_score.py --split test               # scores from the human-filled sheets
```

Human judgments and ratings are filled by people; the scripts only prepare sheets and compute scores.
