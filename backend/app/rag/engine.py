"""
RAG query engine — orchestrates retrieval + answer generation.
"""

import logging
import math
from typing import AsyncGenerator

from app.config import settings
from app.embeddings import EmbeddingService
from app.rag.embedder import load_index
from app.llm import generate_answer, stream_answer
from app.schemas import QueryResponse, PaperSummary
from app.ner import extract_entities, extract_key_terms

logger = logging.getLogger(__name__)


class RAGEngine:
    """Encapsulates the FAISS index, chunk store, and embedding service."""

    def __init__(
        self, index_path: str | None = None, chunk_path: str | None = None
    ) -> None:
        # Paths are overridable so an experiment (e.g. the eval harness) can
        # build its own corpus without disturbing the live index.
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.chunk_path = chunk_path or settings.CHUNK_DATA_PATH
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

    #: Weight on the vector score in the fused ranking. Entity evidence should
    #: adjust the ordering, not replace it — at 0.0 the re-ranker discards
    #: similarity entirely, which measurably hurt precision.
    VECTOR_WEIGHT = 0.7

    async def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Vector search returning (chunk, distance) pairs, nearest first."""
        self.initialize()

        if self.index is None or len(self.chunks) == 0:
            logger.warning("No index available for retrieval.")
            return []

        logger.info("Retrieving top-%d chunks …", k)
        query_embedding = await self.embedder.embed_query(query)

        # Clamp k to actual index size
        actual_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, actual_k)

        results: list[tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))

        logger.info("Retrieved %d chunks.", len(results))
        return results

    async def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Retrieve the top-k most similar chunks for a query."""
        return [chunk for chunk, _ in await self.search(query, k)]

    async def retrieve_entity_aware(self, query: str, k: int = 5) -> list[str]:
        """
        Retrieve 2k candidates by vector similarity, then re-rank by fusing
        similarity with IDF-weighted biomedical term overlap.

        Three properties matter here, each fixing a measured failure:

        * **IDF weighting.** A term present in every candidate carries no
          information — when all six candidates mention "TP53", counting it
          scores them identically and the re-rank is a no-op. Weighting by
          inverse document frequency *within the candidate set* makes common
          terms contribute ~0 and rare, discriminating ones dominate.
        * **Score fusion, not replacement.** Raw match counts let a weakly
          similar chunk leapfrog a strong one; blending with the normalised
          vector score keeps similarity in control while letting entity
          evidence break near-ties.
        * **Untyped terms.** Uses every biomedical span, not just those that
          fit a gene/disease/drug bucket, so "metformin" and "RNA polymerase
          II" still contribute.
        """
        scored = await self.search(query, k=k * 2)
        if not scored:
            return []

        terms = extract_key_terms(query)
        if not terms:
            logger.info("No biomedical terms in query — using vector order.")
            return [chunk for chunk, _ in scored[:k]]

        chunks_lower = [chunk.lower() for chunk, _ in scored]
        n = len(scored)

        # IDF over the candidate set: log(N / (1 + df)), floored at 0 so a term
        # appearing in every candidate contributes nothing.
        weights: dict[str, float] = {}
        for term in terms:
            df = sum(1 for c in chunks_lower if term in c)
            weights[term] = max(0.0, math.log(n / (1 + df)))

        max_weight = sum(weights.values())

        # Convert L2 distances to similarities, then min-max normalise so both
        # components live on [0, 1] and the weighting is meaningful.
        sims = [1.0 / (1.0 + dist) for _, dist in scored]
        lo, hi = min(sims), max(sims)
        span = (hi - lo) or 1.0

        ranked = []
        for i, (chunk, _) in enumerate(scored):
            vector_score = (sims[i] - lo) / span
            entity_score = (
                sum(w for t, w in weights.items() if t in chunks_lower[i]) / max_weight
                if max_weight > 0
                else 0.0
            )
            fused = (
                self.VECTOR_WEIGHT * vector_score
                + (1 - self.VECTOR_WEIGHT) * entity_score
            )
            ranked.append((fused, i, chunk))

        # Sort by fused score; ties fall back to original vector rank.
        ranked.sort(key=lambda t: (-t[0], t[1]))
        return [chunk for _, _, chunk in ranked[:k]]

    async def query(
        self,
        question: str,
        sources: list[PaperSummary] | None = None,
        k: int = 5,
        entity_aware: bool = False,
    ) -> QueryResponse:
        """Full RAG pipeline: retrieve context → generate answer."""
        if entity_aware:
            context_chunks = await self.retrieve_entity_aware(question, k)
        else:
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
        self, question: str, k: int = 5, entity_aware: bool = False
    ) -> AsyncGenerator[str, None]:
        """Stream answer tokens after retrieval."""
        if entity_aware:
            context_chunks = await self.retrieve_entity_aware(question, k)
        else:
            context_chunks = await self.retrieve(question, k)

        if not context_chunks:
            yield "No relevant information found in the knowledge base."
            return

        async for token in stream_answer(question, context_chunks):
            yield token
