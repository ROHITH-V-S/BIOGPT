import pytest
from pydantic import ValidationError
from app.schemas import QueryRequest, IngestRequest

def test_query_request_validation():
    # Valid
    QueryRequest(query="Valid query", max_results=5)
    
    # Invalid length
    with pytest.raises(ValidationError):
        QueryRequest(query="", max_results=5)
        
    with pytest.raises(ValidationError):
        QueryRequest(query="a" * 2001, max_results=5)
        
    # Invalid max_results
    with pytest.raises(ValidationError):
        QueryRequest(query="query", max_results=0)

def test_ingest_request_validation():
    # Valid
    IngestRequest(pdf_path="test.pdf")
    
    # Invalid extension
    with pytest.raises(ValidationError):
        IngestRequest(pdf_path="test.txt")
