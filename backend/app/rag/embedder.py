"""
Document embedding and FAISS index management.
"""

import logging
import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np

from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


async def embed_and_store(
    chunks: List[str],
    index_path: str,
    chunk_path: str,
    embedder: EmbeddingService,
) -> None:
    """
    Embed text chunks and store them in a FAISS index + pickle file.

    If an index already exists at *index_path*, new vectors are appended.
    """
    if not chunks:
        logger.warning("No chunks to embed — skipping.")
        return

    # Ensure parent directories exist
    for path in (index_path, chunk_path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    logger.info("Embedding %d chunks …", len(chunks))
    embeddings = await embedder.embed_texts(chunks)

    dimension = embeddings.shape[1]

    if os.path.exists(index_path):
        logger.info("Appending to existing FAISS index at %s", index_path)
        index = faiss.read_index(index_path)
    else:
        logger.info("Creating new FAISS index (dim=%d)", dimension)
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    faiss.write_index(index, index_path)
    logger.info("Saved FAISS index → %s (%d vectors total)", index_path, index.ntotal)

    # Append chunk text to the pickle store
    existing_chunks: List[str] = []
    if os.path.exists(chunk_path):
        with open(chunk_path, "rb") as f:
            existing_chunks = pickle.load(f)

    existing_chunks.extend(chunks)
    with open(chunk_path, "wb") as f:
        pickle.dump(existing_chunks, f)
    logger.info("Saved %d total chunks → %s", len(existing_chunks), chunk_path)


def load_index(
    index_path: str, chunk_path: str
) -> Tuple[faiss.Index | None, List[str]]:
    """
    Load a FAISS index and its associated chunk data from disk.

    Returns (None, []) if the files don't exist yet (first run).
    """
    if not os.path.exists(index_path) or not os.path.exists(chunk_path):
        logger.info("No existing index at %s — starting fresh.", index_path)
        return None, []

    logger.info("Loading FAISS index from %s", index_path)
    index = faiss.read_index(index_path)

    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    logger.info("Loaded index (%d vectors) and %d chunks.", index.ntotal, len(chunks))
    return index, chunks


if __name__ == "__main__":
    """CLI entry point for `make ingest` / `python -m app.rag.embedder`."""
    import asyncio
    import sys

    from app.config import settings
    from app.rag.loader import load_pdf_chunks

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_paper.pdf"

    if not os.path.isfile(pdf_path):
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    chunks = load_pdf_chunks(pdf_path)
    print(f"Loaded {len(chunks)} chunks from {pdf_path}")

    embedder = EmbeddingService()
    asyncio.run(
        embed_and_store(
            chunks=chunks,
            index_path=settings.FAISS_INDEX_PATH,
            chunk_path=settings.CHUNK_DATA_PATH,
            embedder=embedder,
        )
    )
    print("✅ Embedding and indexing complete!")
