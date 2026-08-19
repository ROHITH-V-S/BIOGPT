.PHONY: dev ingest test lint ner corpus eval eval-hard eval-weak judge compare redis cache-clear

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	python -m app.rag.embedder

test:
	pytest tests/ -v

lint:
	python -m py_compile app/main.py && echo 'OK'

# Redis only — enough to develop against the cache without the full stack.
# Everything still works if you skip this; the cache just stays inactive.
redis:
	docker compose -f ../docker-compose.yml up -d redis

# Drop cached embeddings and PubMed responses. Needed only when you want to
# force a genuine cold-path measurement.
cache-clear:
	docker compose -f ../docker-compose.yml exec -T redis redis-cli --scan --pattern 'biogpt:*' | xargs -r docker compose -f ../docker-compose.yml exec -T redis redis-cli del

# Install the scispaCy biomedical NER model (see scripts/install_ner_model.py
# for why this can't live in requirements.txt).
ner:
	python scripts/install_ner_model.py

# Build both evaluation corpora from live PubMed abstracts.
corpus:
	python scripts/build_eval_corpus.py --suite general
	python scripts/build_eval_corpus.py --suite hard

# Baseline vector search vs entity-aware re-ranking.
eval:
	python scripts/run_eval.py --suite general

eval-hard:
	python scripts/run_eval.py --suite hard

# Same comparison against a deliberately weaker encoder. Tests whether the
# "dense retrieval already saturates" explanation actually holds. No API quota.
eval-weak:
	python scripts/run_eval.py --suite general --backend local
	python scripts/run_eval.py --suite hard --backend local

# Generation metrics (faithfulness / answer relevancy) via a locally served
# model. ~48 calls, which does not fit OpenRouter's 50-request daily free tier.
# Requires: ollama serve  +  ollama pull llama3.2  +  ollama pull gemma3n:e4b
judge:
	python scripts/run_eval.py --suite general --judge --llm ollama --judge-model gemma3n:e4b
	python scripts/run_eval.py --suite hard --judge --llm ollama --judge-model gemma3n:e4b

# General-purpose vs biomedical embedding models. First run downloads
# PubMedBERT (~440 MB); the OpenRouter arm reuses the corpus index.
compare:
	python scripts/retrieval_comparison.py --suite general
	python scripts/retrieval_comparison.py --suite hard
