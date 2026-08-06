"""
PDF loading and text chunking for the RAG pipeline.
"""

import logging
from typing import List

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_pdf_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Read a PDF file and split its text into overlapping chunks.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Target size (in characters) of each chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    logger.info("Loading PDF: %s", pdf_path)
    full_text = ""

    with open(pdf_path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted.strip() + "\n"

    if not full_text.strip():
        logger.warning("No text extracted from %s", pdf_path)
        return []

    return split_text(full_text, chunk_size, chunk_overlap)


def split_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[str]:
    """Split raw text into chunks using LangChain's recursive splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    logger.info("Split text into %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)
    return chunks
