"""
RAG query engine — orchestrates retrieval + answer generation.
"""

import logging
from typing import AsyncGenerator

from app.config import settings
from app.embeddings import EmbeddingService
from app.rag.embedder import load_index
from app.llm import generate_answer, stream_answer
from app.schemas import QueryResponse, PaperSummary

logger = logging.getLogger(__name__)


class RAGEngine:
    """Encapsulates the FAISS index, chunk store, and embedding service."""

    def __init__(self) -> None:
        self.index_path = settings.FAISS_INDEX_PATH
        self.chunk_path = settings.CHUNK_DATA_PATH
        self.embedder = EmbeddingService()
        self.index = None
        self.chunks: list[str] = []
        self._initialized = False

    def initialize(self) -> None:
        """Load the FAISS index and chunk data from disk (if available)."""
        if not self._initialized:
            self.index, self.chunks = load_index(self.index_path, self.chunk_path)
            self._initialized = True
            if self.index is not None:
                logger.info(
                    "FAISS index loaded: %d vectors, %d chunks",
                    self.index.ntotal,
                    len(self.chunks),
                )
            else:
                logger.warning(
                    "No existing FAISS index found at %s — run /ingest first.",
                    self.index_path,
                )

    def reload(self) -> None:
        """Re-load the index from disk (e.g. after ingestion)."""
        self._initialized = False
        self.initialize()

    async def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Retrieve the top-k most similar chunks for a query."""
        self.initialize()

        if self.index is None or len(self.chunks) == 0:
            logger.warning("No index available for retrieval.")
            return []

        logger.info("Retrieving top-%d chunks …", k)
        query_embedding = await self.embedder.embed_query(query)

        # Clamp k to actual index size
        actual_k = min(k, self.index.ntotal)
        _, indices = self.index.search(query_embedding, actual_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])

        logger.info("Retrieved %d chunks.", len(results))
        return results

    async def query(
        self,
        question: str,
        sources: list[PaperSummary] | None = None,
        k: int = 5,
    ) -> QueryResponse:
        """Full RAG pipeline: retrieve context → generate answer."""
        context_chunks = await self.retrieve(question, k)

        if not context_chunks:
            return QueryResponse(
                answer=(
                    "No relevant information found in the knowledge base. "
                    "Try ingesting documents first."
                ),
                sources=sources or [],
                chunks=[],
            )

        answer = await generate_answer(question, context_chunks)
        return QueryResponse(
            answer=answer,
            sources=sources or [],
            chunks=context_chunks,
        )

    async def stream_query(
        self, question: str, k: int = 5
    ) -> AsyncGenerator[str, None]:
        """Stream answer tokens after retrieval."""
        context_chunks = await self.retrieve(question, k)

        if not context_chunks:
            yield "No relevant information found in the knowledge base."
            return

        async for token in stream_answer(question, context_chunks):
            yield token
