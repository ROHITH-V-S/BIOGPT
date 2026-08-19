from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LOG_LEVEL: str = 'INFO'
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 30
    API_KEY_REQUIRED: bool = False
    API_KEY: str = ''
    # Defaulted rather than required: the app must still import without a
    # .env (CI, test collection, and the EMBEDDING_BACKEND=local path, which
    # needs no key at all). Missing credentials surface at call time instead.
    OPENROUTER_API_KEY: str = ''
    OPENROUTER_BASE_URL: str = 'https://openrouter.ai/api/v1'
    LLM_MODEL_FALLBACK_LIST: list[str] = [
        'google/gemma-4-31b-it:free',
        'nvidia/nemotron-3-ultra-550b-a55b:free',
        'poolside/laguna-s-2.1:free'
    ]
    # 'openrouter' (hosted, 50 requests/day on the free tier) or 'ollama'
    # (local, unmetered). Ollama exposes an OpenAI-compatible API, so both
    # backends run through the same client code — only the base URL, key and
    # model list differ. See docs/eval_report.md for why the eval harness
    # defaults to Ollama while the app defaults to OpenRouter.
    LLM_BACKEND: str = 'openrouter'
    OLLAMA_BASE_URL: str = 'http://localhost:11434/v1'
    # Ollama needs no credential; the OpenAI SDK requires a non-empty string.
    OLLAMA_API_KEY: str = 'ollama'
    OLLAMA_MODEL_FALLBACK_LIST: list[str] = [
        'llama3.2:latest',
        'gemma3n:e4b'
    ]
    EMBEDDING_MODEL: str = 'nvidia/nemotron-3-embed-1b:free'
    EMBEDDING_BACKEND: str = 'openrouter'  # 'openrouter', 'local', or 'biomedical'
    LOCAL_EMBEDDING_MODEL: str = 'sentence-transformers/all-MiniLM-L6-v2'
    BIOMEDICAL_EMBEDDING_MODEL: str = 'pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb'
    # Redis is an optional accelerator, never a requirement: it caches
    # embeddings (which cost API quota) and PubMed responses, and backs the
    # rate limiter so counters survive a restart and coordinate across workers.
    # When it is absent every caller falls back to the uncached path.
    REDIS_ENABLED: bool = True
    REDIS_URL: str = 'redis://localhost:6379/0'
    REDIS_TIMEOUT_SECONDS: float = 2.0
    # After a failure the cache is skipped for this long, then re-probed once.
    # Retrying every request would add a timeout to every request; never
    # retrying would let a brief blip disable caching until the next restart.
    REDIS_RETRY_COOLDOWN_SECONDS: float = 30.0
    # Embeddings are deterministic per (model, text) — a long TTL is safe.
    CACHE_TTL_EMBEDDINGS: int = 604800  # 7 days
    # PubMed results change as papers are indexed, so expire them daily.
    CACHE_TTL_PUBMED: int = 86400  # 1 day
    FAISS_INDEX_PATH: str = 'data/vector.index'
    CHUNK_DATA_PATH: str = 'data/chunk_data.pkl'
    CORS_ORIGINS: list[str] = ['http://localhost:3000', 'http://localhost:5173']
    ENTITY_AWARE_DEFAULT: bool = False
    NER_MODEL: str = 'en_core_sci_sm'

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()
