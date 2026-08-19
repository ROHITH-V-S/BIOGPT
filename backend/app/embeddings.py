"""
Embedding service — supports OpenRouter API and local sentence-transformers.

The active backend is selected via the EMBEDDING_BACKEND config flag.
"""

import logging
from typing import List

import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

from app import cache
from app.config import settings
from app.retry import retry_if_transient

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings via OpenRouter or a local model."""

    def __init__(self) -> None:
        self.backend = settings.EMBEDDING_BACKEND
        self._local_model = None  # lazy-loaded

        if self.backend == "openrouter":
            self._client = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
            )
            self._model_name = settings.EMBEDDING_MODEL
            logger.info("Embedding backend: OpenRouter (%s)", self._model_name)
        else:
            backend_model = settings.LOCAL_EMBEDDING_MODEL if self.backend == "local" else settings.BIOMEDICAL_EMBEDDING_MODEL
            logger.info(
                "Embedding backend: %s (%s) — will load on first use.",
                self.backend,
                backend_model,
            )

    def _get_local_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            backend_model = settings.LOCAL_EMBEDDING_MODEL if self.backend == "local" else settings.BIOMEDICAL_EMBEDDING_MODEL
            logger.info("Loading local embedding model: %s …", backend_model)
            self._local_model = SentenceTransformer(backend_model)
            logger.info("Local embedding model loaded.")
        return self._local_model

    # ------------------------------------------------------------------
    # OpenRouter path
    # ------------------------------------------------------------------
    @retry(
        retry=retry_if_transient,
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _embed_openrouter(self, texts: List[str]) -> np.ndarray:
        """Embed via OpenRouter's embeddings endpoint with retry."""
        logger.debug("Embedding %d texts via OpenRouter …", len(texts))
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model_name,
            # The OpenAI SDK defaults to base64; OpenRouter's embedding
            # providers reject it, so request plain floats explicitly.
            encoding_format="float",
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    # ------------------------------------------------------------------
    # Local path
    # ------------------------------------------------------------------
    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Embed via a local sentence-transformers model."""
        logger.debug("Embedding %d texts locally …", len(texts))
        model = self._get_local_model()
        vectors = model.encode(texts, show_progress_bar=False)
        return np.array(vectors, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _cache_model_id(self) -> str:
        """Cache identity of the active model — dimensionality differs per backend."""
        if self.backend == "openrouter":
            return f"openrouter/{self._model_name}"
        model = (
            settings.LOCAL_EMBEDDING_MODEL
            if self.backend == "local"
            else settings.BIOMEDICAL_EMBEDDING_MODEL
        )
        return f"{self.backend}/{model}"

    async def _embed_uncached(self, texts: List[str]) -> np.ndarray:
        if self.backend == "openrouter":
            batch_size = 10  # stay under rate limits
            batches = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batches.append(await self._embed_openrouter(batch))
            return np.vstack(batches)
        return self._embed_local(texts)

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts, reusing any cached vectors.

        Caching is per *text*, not per batch, so a partially-changed corpus only
        pays for the texts that actually changed. Re-running the eval harness
        over an unchanged corpus becomes free — which matters because embedding
        ~300 chunks was the largest single draw on a 50-request daily quota.

        Returns an ndarray of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model_id = self._cache_model_id()
        cached: dict[int, np.ndarray] = {}
        missing: list[int] = []

        for i, text in enumerate(texts):
            vector = await cache.get_embedding(model_id, text)
            if vector is None:
                missing.append(i)
            else:
                cached[i] = vector

        if cached:
            logger.info(
                "Embedding cache: %d/%d hit (%d to compute).",
                len(cached), len(texts), len(missing),
            )

        if not missing:
            return np.vstack([cached[i] for i in range(len(texts))])

        fresh = await self._embed_uncached([texts[i] for i in missing])
        for slot, i in enumerate(missing):
            cached[i] = fresh[slot]
            await cache.set_embedding(model_id, texts[i], fresh[slot])

        return np.vstack([cached[i] for i in range(len(texts))]).astype(np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns an ndarray of shape ``(1, embedding_dim)``.
        """
        model_id = self._cache_model_id()
        vector = await cache.get_embedding(model_id, query)
        if vector is not None:
            return vector.reshape(1, -1).astype(np.float32)

        result = await self._embed_uncached([query])
        await cache.set_embedding(model_id, query, result[0])
        return result
