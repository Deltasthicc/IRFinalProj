# TradeMesh

Search and cited question answering over the latest 10-K filings of about 20 companies (CSD358 Information Retrieval, Monsoon 2026).
This repository holds the whole project: data, indexes, graph and vision track (Role A), rankers and evaluation (Role B), and query processing, RAG and the app (Role C).

## Team and roles

| Member | Role | What they own |
|---|---|---|
| Tanish Shitanshu | **Role B** (lead) and most of **Role A** | **Role B:** BM25 from scratch with tuning, dense retrieval, rank fusion (RRF), cross-encoder reranking, pseudo-relevance feedback (`src/trademesh/ranking/`); the evaluation protocol, pooling, judgments, paired bootstrap, hypotheses and ablations (`src/trademesh/evaluation/`, `data/judgments/`, `data/results/`, and the retrieval scripts in `eval/`: `run_retrieval`, `make_pools`, `tune_bm25`, `hypotheses`, `ablation`, `pool_bias`, `speed_quality`, `cost_table`); the Report B sections (`docs/report_B/`). **Role A:** corpus download, parsing and chunking (`src/trademesh/ingest/`); TF-IDF, champion lists, positional index and wildcard search (`src/trademesh/index/`); the company mention graph with PageRank and HITS (`src/trademesh/graph/`); the vision track (`src/trademesh/vision/`); the Report A notes (`docs/report_A/`). |
| Ruchir Sharma | **Role A** (one part) | The Boolean retriever (`src/trademesh/index/boolean.py`) |
| Shashwat Rajan | **Role C** | Query processing, RAG pipeline, verifier, abstention, the Streamlit app and the Report C sections (`src/trademesh/query/`, `src/trademesh/rag/`, `src/trademesh/app/`, `docs/report_C/`, and the RAG scripts in `eval/`: `entity_eval`, `rag_run`, `rag_sheets`, `rag_score`, `abstention_eval`) |

## Run it

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows Git Bash; on Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # then choose an LLM provider and add a key
streamlit run src/trademesh/app/streamlit_app.py
```

The first load builds the search indexes for the 11,964 passages (about 15 seconds, once per session).

**Best retrieval setup (optional).** Dense retrieval, the hybrid retriever and the cross-encoder reranker need `pip install -r requirements-ml.txt` (about 1 GB with PyTorch; the two small models download on first use). Without it the app still runs on TF-IDF, Boolean, BM25 and BM25 with pseudo-relevance feedback, and the System tab shows the reranker as not installed. The reported best-RAG numbers and the committed demo cache come from the hybrid retriever with the reranker, so install it to reproduce them.

With Docker (packaging only):

```bash
docker compose up --build                       # http://localhost:8501
INSTALL_ML=1 docker compose up --build          # also install dense retrieval and the reranker
docker compose --profile local-llm up --build   # also start a local Ollama
```

Without Role A's `data/passages.jsonl` and `data/companies.json` the app runs on a tiny **test fixture of invented text** and says so in a warning. Without a key (`LLM_PROVIDER=stub`) answers are extractive quotes. Neither can produce a reportable number: the eval scripts refuse to run on them unless you pass `--allow-stub` (dry runs go to `eval/out/dryrun/`).

### The app

Six tabs: **Ask** (cited answer, clickable evidence, a trust panel with the verifier's checks, closed-book toggle, compare mode), **Search** (several retrievers side by side, snippets, per-term score explanations), **Graph** (company mention graph, PageRank), **Pages** (page-image retrieval; the vision track was gated and not run, so this tab says so), **Evaluation** (the locked retrieval, RAG and abstention results) and **System** (what is wired up, headline numbers, and a spelling-correction playground). The look is a dark glass theme; its markup is in `src/trademesh/app/ui.py` and its stylesheet in `src/trademesh/app/cinematic.css`.

### Offline demo

Every LLM call is cached by the hash of (model, prompt). To check that the demo works with no network, run `LLM_OFFLINE=1 python eval/demo_warm.py` (a question that is not cached stops the run), then start the app with `LLM_OFFLINE=1`. The example buttons use the questions that `eval/demo_warm.py` caches. The cache was built with `gemini-3.5-flash-lite` and the hybrid retriever plus reranker, so the offline demo needs that setup (see above).

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
| `src/trademesh/app/` | Streamlit app and its testable back end: `streamlit_app.py` (layout), `ui.py` (HTML blocks), `cinematic.css` (look), `icons.py` (Phosphor icons), `static/` (photographs), `backend.py` (data loading, no Streamlit) |
| `src/trademesh/dev_stub.py` | **dev-only** in-memory BM25 fallback, used only when Role B's rankers are not installed |
| `src/trademesh/ingest/`, `index/`, `graph/`, `vision/` | Role A: corpus, TF-IDF / Boolean / positional / wildcard indexes, mention graph, page-image scoring |
| `src/trademesh/ranking/`, `evaluation/` | Role B: BM25, dense, RRF fusion, reranker, PRF; metrics, pooling, qrels, bootstrap |
| `data/judgments/`, `data/results/` | judged pools, qrels, test-run result files |
| `docs/report_A/`, `docs/report_B/` | report notes for Role A and the Report B sections |
| `data/needs/` | 30 information needs (10 dev, 20 test), 10 unanswerable questions, 40 typo'd company names |
| `eval/` | `entity_eval.py`, `rag_run.py`, `rag_sheets.py`, `rag_score.py`, `abstention_eval.py` |
| `eval/human_*.py`, `eval/human/` | the blind **human rating** package: tool, blank sheets, scorer and step-by-step guide (`eval/human/README.md`) |
| `docs/report_C/` | report drafts for Role C's sections and the model figure |
| `tests/` | pytest suite and the clearly marked fixture |

## Interfaces for the other roles

* Types: `src/trademesh/types.py`. Field names match the `data/passages.jsonl` lines (one `Passage` per line).
* **Role A** may provide `trademesh.index.build_retrievers(passages) -> {name: Retriever}`. **Role B** may provide `trademesh.ranking.build_retrievers(passages)` and `trademesh.ranking.build_reranker(passages)`. They are discovered automatically (`src/trademesh/retrievers.py`); the app and eval scripts then stop using the stub.
* Optional app data: `data/graph/company_graph.json` (nodes with `ticker, pagerank, hub, authority, in_degree`; edges `source -> target`) and `data/vision/results.json`. Formats are documented in `src/trademesh/app/backend.py`.

## LLM settings (Gemini, Anthropic, OpenAI-compatible, Ollama)

Using Gemini: get a key at https://aistudio.google.com/apikey, then in `.env` set
`LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...` (optionally `LLM_MODEL=`, default `gemini-3.5-flash-lite`, the model the RAG study and the committed demo cache used).
The key stays in `.env`, which is git-ignored. See `.env.example` for all options. Every call is cached under `LLM_CACHE_DIR` by the hash of (model, prompt, parameters) and logged to `calls.jsonl`, so the demo can run offline from the cache (`LLM_OFFLINE=1`). The eval scripts print the number of live (uncached) calls. Keys live only in `.env`, which is git-ignored.

## Evaluation (Role C)

```bash
python eval/entity_eval.py                          # 40 typo'd names -> eval/out/entity_eval.json
python eval/abstention_eval.py                      # tau tuned on dev, headline on test
python eval/rag_run.py --split test                 # closed-book vs BM25-RAG vs best-RAG
python eval/rag_sheets.py --split test              # blind rating sheet + separate key + citation sheet
python eval/rag_score.py --split test               # scores from the human-filled sheets
```

The scripts prepare the sheets and compute the scores; they never invent a rating. In the reported study the answer ratings, citation checks and retrieval judgments were made by a language model in two passes (disclosed in the report, audit files in `docs/report_C/audit/`), and no human validated them. A blind human re-rating is prepared in `eval/human/` (run `streamlit run eval/human_app.py`; the guide is `eval/human/README.md`); once it is done, `python eval/human_score.py` writes the human scores and the app's Evaluation tab shows them.

## Credits

* Photographs (served from `src/trademesh/app/static/`, details in `credits.json` and the app footer): Berenice Abbott, "Wall Street, from roof of Irving Trust Co. Building" (public domain); Billie Grace Ward, "Sign of the New York Stock Exchange, Broad Street" (CC0); "Manhattan 1931" (public domain). All from Wikimedia Commons.
* Icons: [Phosphor Icons](https://phosphoricons.com), bold weight, MIT licence.
* Fonts: Outfit and JetBrains Mono from Google Fonts (loaded by the browser; the page falls back to system fonts offline).
