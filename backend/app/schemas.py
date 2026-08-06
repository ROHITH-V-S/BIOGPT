from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    max_results: int = 5
    stream: bool = False

class PaperSummary(BaseModel):
    title: str
    summary: str
    link: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[PaperSummary]
    chunks: List[str]

class IngestRequest(BaseModel):
    pdf_path: Optional[str] = None
    pubmed_query: Optional[str] = None
    max_papers: int = 5

class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int

class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    llm_models: List[str]
    index_loaded: bool

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
