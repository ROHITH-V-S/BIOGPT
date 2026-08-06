from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = 'https://openrouter.ai/api/v1'
    LLM_MODEL_FALLBACK_LIST: list[str] = [
        'google/gemma-4-27b-it:free',
        'nvidia/nemotron-3-ultra-550b-a55b:free',
        'poolside/laguna-s-2.1:free'
    ]
    EMBEDDING_MODEL: str = 'nvidia/nemotron-3-embed-1b:free'
    EMBEDDING_BACKEND: str = 'openrouter'  # 'openrouter' or 'local'
    LOCAL_EMBEDDING_MODEL: str = 'sentence-transformers/all-MiniLM-L6-v2'
    FAISS_INDEX_PATH: str = 'data/vector.index'
    CHUNK_DATA_PATH: str = 'data/chunk_data.pkl'
    CORS_ORIGINS: list[str] = ['http://localhost:3000', 'http://localhost:5173']

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()
