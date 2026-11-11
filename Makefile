.PHONY: test run entity-eval abstention rag-study docker pools judge tune retrieval-dev retrieval-test ablation-dev pool-bias cost

test:
	python -m pytest -q

run:
	streamlit run src/trademesh/app/streamlit_app.py

entity-eval:
	python eval/entity_eval.py

abstention:
	python eval/abstention_eval.py

rag-study:
	python eval/rag_run.py --split test
	python eval/rag_sheets.py --split test

docker:
	docker compose up --build

# ---- Role B: ranking and retrieval evaluation (see docs/report_B/README.md for the order) ----
pools:            ## 1. build blind judging sheets from all systems' top 8
	python eval/make_pools.py --judges Tanish,Shashwat,Ruchir

judge:            ## 2. judge in the browser (each person picks their name)
	streamlit run eval/judge_app.py

merge:            ## 3. fold the filled sheets into data/judgments/qrels.csv
	python eval/make_pools.py --merge

tune:             ## 4. BM25 k1/b on the dev split only
	python eval/tune_bm25.py

retrieval-dev:
	python eval/run_retrieval.py --split dev

ablation-dev:
	python eval/ablation.py --split dev

retrieval-test:   ## ONCE, after the feature freeze
	python eval/run_retrieval.py --split test
	python eval/ablation.py --split test
	python eval/pool_bias.py
	python eval/cost_table.py

.PHONY: merge
