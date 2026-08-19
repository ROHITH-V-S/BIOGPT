from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_results: int = Field(5, ge=1, le=20)
    stream: bool = False
    entity_aware: bool = False
    # When true, live PubMed abstracts are added to the generation context
    # alongside the local corpus. Set false to ground strictly on the index.
    use_pubmed_context: bool = True

class PaperSummary(BaseModel):
    title: str
    summary: str
    link: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[PaperSummary]
    chunks: List[str]
    entities: Optional[Dict[str, List[str]]] = None

class IngestRequest(BaseModel):
    pdf_path: Optional[str] = None
    pubmed_query: Optional[str] = None
    max_papers: int = 5

    @field_validator('pdf_path')
    def validate_pdf_path(cls, v):
        if v is not None and not v.endswith('.pdf'):
            raise ValueError('pdf_path must end with .pdf')
        return v

class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int

class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    llm_backend: str
    llm_models: List[str]
    index_loaded: bool
    #: False whenever Redis is disabled or unreachable. The API is fully
    #: functional either way — this reports whether caching and cross-worker
    #: rate limiting are active, not whether the service is healthy.
    cache_active: bool

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
