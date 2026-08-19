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
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import settings
from app.schemas import (
    QueryRequest, QueryResponse, IngestRequest, IngestResponse,
    HealthResponse, ErrorResponse, PaperSummary,
)
from app.rag.engine import RAGEngine
from app.rag.embedder import embed_and_store
from app.rag.loader import load_pdf_chunks, split_text
from app import cache
from app.llm import (
    AllModelsFailedError, active_backend, active_models, generate_answer,
    stream_answer, _get_successful_model,
)
from app.ner import extract_entities
from app.pubmed import PubMedArticle, search_abstracts, search_pmids, fetch_abstracts
from app.logging_config import setup_logging, request_id_var
from app.middleware import RequestIDMiddleware
from app.rate_limiter import rate_limit_dependency
from app.auth import verify_api_key

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging(settings.LOG_LEVEL)
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

    # Probe the cache at startup so its state is known before the first request
    # and a misconfigured REDIS_URL shows up in the logs immediately rather than
    # as a surprise latency spike. A failure here is non-fatal by design.
    await cache.get_client()

    yield  # ← app is running

    logger.info("Shutting down BioGPT Explorer API …")
    await cache.close()


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

app.add_middleware(RequestIDMiddleware)
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
SUMMARY_SNIPPET_CHARS = 280

#: Shown when every fallback model returns 429. OpenRouter's free tier is
#: capped per *day* (reset at 00:00 UTC), so "try again shortly" would be
#: misleading — say what actually happened and what resolves it.
QUOTA_MESSAGE = (
    "OpenRouter free-tier quota exhausted — all fallback models returned 429. "
    "The daily allowance (50 requests) resets at 00:00 UTC. Retrieval, entity "
    "extraction and PubMed search still work; only answer generation is "
    "unavailable. To keep generating without waiting, set LLM_BACKEND=ollama "
    "to serve a model locally."
)


async def fetch_pubmed(query: str, max_results: int = 5) -> list[PubMedArticle]:
    """
    Fetch PubMed articles *with their abstracts* for a query.

    Uses efetch rather than esummary so the abstract text is available — both
    to show a real snippet in the UI and to hand the model something it can
    actually ground on. Network failures degrade to an empty list: PubMed is
    an enrichment, never a hard dependency of answering.
    """
    try:
        return await search_abstracts(query, max_results)
    except Exception as exc:
        logger.error("PubMed fetch error: %s", exc)
        return []


def to_summaries(articles: list[PubMedArticle]) -> list[PaperSummary]:
    """Project articles onto the wire format the frontend renders."""
    summaries: list[PaperSummary] = []
    for art in articles:
        snippet = art.abstract[:SUMMARY_SNIPPET_CHARS].rstrip()
        if len(art.abstract) > SUMMARY_SNIPPET_CHARS:
            snippet += "…"
        summaries.append(
            PaperSummary(
                title=art.title or "Untitled",
                summary=snippet or art.citation,
                link=art.url,
            )
        )
    return summaries


def chunks_from_articles(articles: list[PubMedArticle]) -> list[str]:
    """
    Turn PubMed abstracts into indexable chunks.

    Each chunk keeps its "[PubMed <pmid>] <title>" header so provenance
    survives retrieval — without it, an indexed abstract becomes anonymous
    text and the model can no longer cite where a claim came from. Long
    abstracts are split with the same splitter used for PDFs.
    """
    chunks: list[str] = []
    for art in articles:
        header = f"[PubMed {art.pmid}] {art.title}"
        for part in split_text(art.abstract):
            chunks.append(f"{header}\n{part}")
    return chunks


def build_context(
    chunks: list[str], articles: list[PubMedArticle], use_pubmed: bool
) -> list[str]:
    """
    Assemble the context blocks handed to the LLM.

    Local corpus chunks come first (they are the retrieved, ranked evidence);
    PubMed abstracts follow, each tagged with its PMID so the model can cite
    provenance. Previously the abstracts were fetched and displayed but never
    passed to generation, so the UI credited sources the model had not read.
    """
    context = [f"[Local corpus] {c}" for c in chunks]
    if use_pubmed:
        context.extend(art.as_context() for art in articles)
    return context


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "detail": None},
        headers=getattr(exc, "headers", None)
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation Error", "detail": str(exc.errors())},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": f"Request ID: {request_id_var.get()}"},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Return system health and configuration summary."""
    return HealthResponse(
        status="ok",
        embedding_backend=settings.EMBEDDING_BACKEND,
        llm_backend=active_backend(),
        llm_models=active_models(),
        index_loaded=rag_engine is not None and rag_engine.index is not None,
        cache_active=cache.is_active(),
    )


@app.post("/query", dependencies=[Depends(rate_limit_dependency), Depends(verify_api_key)])
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
            
        articles = await sources_task
        sources = to_summaries(articles)
        context = build_context(context_chunks, articles, req.use_pubmed_context)

        if not context:
            return QueryResponse(
                answer="No relevant information found in the knowledge base. Try ingesting documents first.",
                sources=sources,
                chunks=[],
                entities=entities
            )

        answer = await generate_answer(req.query, context)

        return QueryResponse(answer=answer, sources=sources, chunks=context_chunks, entities=entities)
    except AllModelsFailedError as exc:
        logger.error("Generation unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=QUOTA_MESSAGE if exc.rate_limited else "All LLM models failed.",
        )
    except HTTPException:
        raise
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
            
        articles = await sources_task
        sources = to_summaries(articles)
        context = build_context(context_chunks, articles, req.use_pubmed_context)

        # 2. Send retrieved chunks + sources first
        chunk_event = {
            "chunks": context_chunks,
            "sources": [s.model_dump() for s in sources],
            "entities": entities
        }
        yield f"event: chunk\ndata: {json.dumps(chunk_event)}\n\n"

        if not context:
            no_data_msg = "No relevant information found in the knowledge base."
            yield f'event: token\ndata: {json.dumps({"token": no_data_msg})}\n\n'
            yield f'event: done\ndata: {json.dumps({"answer": no_data_msg, "model": "none"})}\n\n'
            return

        # 3. Stream answer tokens
        full_answer = ""

        async for token in stream_answer(req.query, context):
            full_answer += token
            yield f'event: token\ndata: {json.dumps({"token": token})}\n\n'

        # Report the model that actually served the request — the first entry
        # in the fallback list may well have been skipped.
        model_used = _get_successful_model() or active_models()[0]

        # 4. Send done event
        yield f'event: done\ndata: {json.dumps({"answer": full_answer, "model": model_used})}\n\n'

    except AllModelsFailedError as exc:
        # The stream has already returned 200, so the failure has to travel as
        # an SSE event rather than an HTTP status.
        logger.error("Generation unavailable mid-stream: %s", exc)
        message = QUOTA_MESSAGE if exc.rate_limited else "All LLM models failed."
        yield f'event: error\ndata: {json.dumps({"error": message})}\n\n'
    except Exception as exc:
        logger.exception("SSE streaming error")
        yield f'event: error\ndata: {json.dumps({"error": str(exc)})}\n\n'


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(rate_limit_dependency), Depends(verify_api_key)])
async def ingest_endpoint(req: IngestRequest):
    """Ingest a PDF file into the vector index."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    if not req.pdf_path and not req.pubmed_query:
        raise HTTPException(
            status_code=400, detail="Must provide either pdf_path or pubmed_query"
        )

    if req.pdf_path and not os.path.isfile(req.pdf_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.pdf_path}")

    try:
        if req.pdf_path:
            chunks = load_pdf_chunks(req.pdf_path)
            logger.info("Loaded %d chunks from %s", len(chunks), req.pdf_path)
        else:
            articles = await search_abstracts(req.pubmed_query, req.max_papers)
            if not articles:
                raise HTTPException(
                    status_code=404,
                    detail=f"No PubMed abstracts found for: {req.pubmed_query}",
                )
            chunks = chunks_from_articles(articles)
            logger.info(
                "Loaded %d chunks from %d PubMed abstracts for '%s'",
                len(chunks), len(articles), req.pubmed_query,
            )

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
