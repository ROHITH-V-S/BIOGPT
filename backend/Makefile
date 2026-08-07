.PHONY: dev ingest test

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	python -m app.rag.embedder

test:
	pytest tests/ -v

lint:
	python -m py_compile app/main.py && echo 'OK'

compare:
	python scripts/retrieval_comparison.py

eval:
	python scripts/run_eval.py
