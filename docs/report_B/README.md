# Role B: ranking, evaluation, report lead

Everything in this folder is a DRAFT written before any real result exists. Every number that has to come from a run is written
`TODO(run)` and names the command that produces it. Nothing here may be filled in by hand-estimating.

## What is done (code, tested on the fixture only)

| Part | Where | Status |
|---|---|---|
| BM25 from scratch, explain(), k1/b tuning | `src/trademesh/ranking/bm25.py`, `eval/tune_bm25.py` | code + tests done |
| Dense retriever (pluggable encoder, cached) | `ranking/dense.py` | code done; needs `requirements-ml.txt` + model download |
| RRF, CombSUM, CombMNZ, hybrid | `ranking/fusion.py` | done |
| Cross-encoder rerank (top 30, calibrated scores) | `ranking/rerank.py` | code done; needs model download |
| Rocchio PRF | `ranking/prf.py` | done |
| Metrics, qrels, pooling, bootstrap, kappa, bpref | `src/trademesh/evaluation/` | done, hand-checked tests |
| Eval scripts | `eval/{make_pools,judge_app,run_retrieval,ablation,pool_bias,speed_quality,cost_table,retrieval_vs_answer,hypotheses}.py` | done, smoke-tested end to end on the fixture with SYNTHETIC judgments |

## What only people (or a networked machine) can do, in this order

1. Real corpus from Role A (`data/passages.jsonl`). Install `pip install -r requirements-ml.txt` on a machine with Hugging Face access.
2. `make pools` -> blind sheets. DONE differently from the plan: an LLM judged all 589 pairs and a second LLM pass judged 91 (see `audit/`); no human judged. If people later grade the spare 91-row sheet with `make judge`, the human-vs-LLM kappa can replace the LLM-vs-LLM one. Nobody opens `data/judgments/pool_key.json` while judging.
3. `make merge`, then `make tune` (dev split only, writes `bm25_params.json`).
4. `make retrieval-dev`, `make ablation-dev` while building.
5. After the feature freeze, once: `make retrieval-test`, then `python eval/hypotheses.py`. The test run is locked after its first execution.
6. Copy numbers from `eval/out/*.json` into `10_retrieval_results.md`; replace each `TODO(run)`.

Dry runs on the fixture go to `eval/out/dryrun/` and are refused unless `--allow-stub` is passed.

## Things to decide or check

* The README names Ruchir Sharma for B; confirm who actually owns B so work is not duplicated.
* `references.md` marks which citations were checked against a source and which still need a check.
* Role A's systems (TF-IDF, champion lists, Boolean/phrase) join the comparison automatically when `trademesh.index.build_retrievers` exists; H1 needs a retriever named `tfidf`.
