.PHONY: dev ingest test

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	python -m app.rag.embedder

test:
	pytest tests/ -v
