"""
Embedding service — supports OpenRouter API and local sentence-transformers.

The active backend is selected via the EMBEDDING_BACKEND config flag.
"""

import logging
from typing import List

import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

from app.config import settings

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
            logger.info(
                "Embedding backend: local (%s) — will load on first use.",
                settings.LOCAL_EMBEDDING_MODEL,
            )

    def _get_local_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedding model: %s …", settings.LOCAL_EMBEDDING_MODEL)
            self._local_model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
            logger.info("Local embedding model loaded.")
        return self._local_model

    # ------------------------------------------------------------------
    # OpenRouter path
    # ------------------------------------------------------------------
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
    )
    async def _embed_openrouter(self, texts: List[str]) -> np.ndarray:
        """Embed via OpenRouter's embeddings endpoint with retry."""
        logger.debug("Embedding %d texts via OpenRouter …", len(texts))
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model_name,
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
    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Returns an ndarray of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self.backend == "openrouter":
            batch_size = 10  # stay under rate limits
            batches = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_emb = await self._embed_openrouter(batch)
                batches.append(batch_emb)
            return np.vstack(batches)
        else:
            return self._embed_local(texts)

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns an ndarray of shape ``(1, embedding_dim)``.
        """
        if self.backend == "openrouter":
            return await self._embed_openrouter([query])
        else:
            return self._embed_local([query])
