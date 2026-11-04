.PHONY: test run entity-eval abstention rag-study docker

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
