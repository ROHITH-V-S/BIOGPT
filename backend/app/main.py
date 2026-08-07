"""
BioGPT Explorer — FastAPI Application

Provides REST and SSE endpoints for biomedical RAG queries,
document ingestion, and system health checks.
"""

import json
import logging
import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas import (
    QueryRequest, QueryResponse, IngestRequest, IngestResponse,
    HealthResponse, ErrorResponse, PaperSummary,
)
from app.rag.engine import RAGEngine
from app.rag.embedder import embed_and_store
from app.rag.loader import load_pdf_chunks
from app.ner import extract_entities

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application-scoped singletons (populated during lifespan)
# ---------------------------------------------------------------------------
rag_engine: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI application."""
    global rag_engine

    logger.info("Starting BioGPT Explorer API …")

    # Ensure data directory exists
    os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH) or "data", exist_ok=True)

    rag_engine = RAGEngine()
    rag_engine.initialize()
    logger.info("RAG engine initialized.")

    yield  # ← app is running

    logger.info("Shutting down BioGPT Explorer API …")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BioGPT Explorer API",
    description=(
        "A retrieval-augmented generation (RAG) backend for biomedical "
        "literature search and question answering, powered by OpenRouter "
        "free-tier models and FAISS vector search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# PubMed helper (async)
# ---------------------------------------------------------------------------
async def fetch_pubmed(query: str, max_results: int = 5) -> list[PaperSummary]:
    """Fetch paper metadata from NCBI PubMed E-utilities."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            search_resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": max_results,
                },
            )
            search_resp.raise_for_status()
            ids = (
                search_resp.json()
                .get("esearchresult", {})
                .get("idlist", [])
            )
            if not ids:
                return []

            summary_resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "json",
                },
            )
            summary_resp.raise_for_status()
            result_data = summary_resp.json().get("result", {})

            papers: list[PaperSummary] = []
            for pid in ids:
                item = result_data.get(pid, {})
                papers.append(
                    PaperSummary(
                        title=item.get("title", "No Title"),
                        summary=item.get("sorttitle", "PubMed abstract"),
                        link=f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                    )
                )
            return papers
    except Exception as exc:
        logger.error("PubMed fetch error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Return system health and configuration summary."""
    return HealthResponse(
        status="ok",
        embedding_backend=settings.EMBEDDING_BACKEND,
        llm_models=settings.LLM_MODEL_FALLBACK_LIST,
        index_loaded=rag_engine is not None and rag_engine.index is not None,
    )


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """
    Run a RAG query.

    When ``stream=True`` the response is a ``text/event-stream`` SSE stream
    with the following event types:

    * ``event: chunk``  — retrieved context chunks + PubMed sources
    * ``event: token``  — a single generated token
    * ``event: done``   — generation complete (includes full answer + model id)
    * ``event: error``  — something went wrong
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    logger.info("Query received: %s (stream=%s)", req.query, req.stream)

    if req.stream:
        return StreamingResponse(
            _stream_rag_response(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Non-streaming path ---------------------------------------------------
    try:
        sources_task = asyncio.create_task(fetch_pubmed(req.query, req.max_results))
        
        entities = extract_entities(req.query) if req.entity_aware else None
        
        if req.entity_aware:
            context_chunks = await rag_engine.retrieve_entity_aware(req.query, k=req.max_results)
        else:
            context_chunks = await rag_engine.retrieve(req.query, k=req.max_results)
            
        sources = await sources_task

        if not context_chunks:
            return QueryResponse(
                answer="No relevant information found in the knowledge base. Try ingesting documents first.",
                sources=sources,
                chunks=[],
                entities=entities
            )

        from app.llm import generate_answer
        answer = await generate_answer(req.query, context_chunks)

        return QueryResponse(answer=answer, sources=sources, chunks=context_chunks, entities=entities)
    except Exception as exc:
        logger.exception("Query error")
        raise HTTPException(status_code=500, detail=str(exc))


async def _stream_rag_response(req: QueryRequest) -> AsyncGenerator[str, None]:
    """
    SSE generator that matches the frontend event contract:

    event: chunk   → { chunks: [...], sources: [...] }
    event: token   → { token: "..." }
    event: done    → { answer: "...", model: "..." }
    event: error   → { error: "..." }
    """
    try:
        # 1. Retrieve context + PubMed sources concurrently
        sources_task = asyncio.create_task(fetch_pubmed(req.query, req.max_results))
        
        entities = extract_entities(req.query) if req.entity_aware else None
        
        if req.entity_aware:
            context_chunks = await rag_engine.retrieve_entity_aware(req.query, k=req.max_results)
        else:
            context_chunks = await rag_engine.retrieve(req.query, k=req.max_results)
            
        sources = await sources_task

        # 2. Send retrieved chunks + sources first
        chunk_event = {
            "chunks": context_chunks,
            "sources": [s.model_dump() for s in sources],
            "entities": entities
        }
        yield f"event: chunk\ndata: {json.dumps(chunk_event)}\n\n"

        if not context_chunks:
            no_data_msg = "No relevant information found in the knowledge base."
            yield f'event: token\ndata: {json.dumps({"token": no_data_msg})}\n\n'
            yield f'event: done\ndata: {json.dumps({"answer": no_data_msg, "model": "none"})}\n\n'
            return

        # 3. Stream answer tokens
        from app.llm import stream_answer, _get_successful_model
        full_answer = ""
        model_used = settings.LLM_MODEL_FALLBACK_LIST[0]

        async for token in stream_answer(req.query, context_chunks):
            full_answer += token
            yield f'event: token\ndata: {json.dumps({"token": token})}\n\n'

        # 4. Send done event
        yield f'event: done\ndata: {json.dumps({"answer": full_answer, "model": model_used})}\n\n'

    except Exception as exc:
        logger.exception("SSE streaming error")
        yield f'event: error\ndata: {json.dumps({"error": str(exc)})}\n\n'


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest):
    """Ingest a PDF file into the vector index."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    if not req.pdf_path:
        raise HTTPException(status_code=400, detail="Must provide pdf_path")

    if not os.path.isfile(req.pdf_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.pdf_path}")

    try:
        chunks = load_pdf_chunks(req.pdf_path)
        logger.info("Loaded %d chunks from %s", len(chunks), req.pdf_path)

        # Ensure output directories exist
        os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH) or "data", exist_ok=True)

        await embed_and_store(
            chunks=chunks,
            index_path=settings.FAISS_INDEX_PATH,
            chunk_path=settings.CHUNK_DATA_PATH,
            embedder=rag_engine.embedder,
        )

        # Reload the engine index
        rag_engine.reload()

        return IngestResponse(status="success", chunks_indexed=len(chunks))
    except Exception as exc:
        logger.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(exc))
