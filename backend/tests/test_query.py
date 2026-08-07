import pytest
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_query_empty_query(async_client):
    response = await async_client.post("/query", json={"query": ""})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"

@pytest.mark.asyncio
async def test_query_valid(async_client):
    with patch("app.main.rag_engine.retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.main.fetch_pubmed", new_callable=AsyncMock) as mock_pubmed, \
         patch("app.main.generate_answer", new_callable=AsyncMock) as mock_generate:
        
        mock_retrieve.return_value = ["chunk1", "chunk2"]
        mock_pubmed.return_value = []
        mock_generate.return_value = "Mock answer"

        response = await async_client.post("/query", json={"query": "test query", "max_results": 5})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Mock answer"
        assert len(data["chunks"]) == 2
