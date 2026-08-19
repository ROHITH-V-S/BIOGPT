import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from app import main
from app.main import app
from app.config import settings

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_REQUIRED", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 100)
    # Never let the suite depend on a live Redis. Without this the first cache
    # probe waits out a connection timeout on any machine that has no Redis,
    # making the suite slow and its result environment-dependent. The
    # Redis-backed paths are tested explicitly in test_cache.py with a fake.
    monkeypatch.setattr(settings, "REDIS_ENABLED", False)


@pytest.fixture(autouse=True)
def reset_cache_state(monkeypatch):
    """
    Reset the cache module's memoised connection state between tests.

    ``cache`` deliberately latches to "unavailable" after a failure so a dead
    Redis is not re-probed on every request. That latch is process-global, so
    without resetting it one test's failure would silently disable the cache for
    every test that follows.
    """
    from app import cache

    monkeypatch.setattr(cache, "_client", None)
    monkeypatch.setattr(cache, "_state", "unprobed")
    monkeypatch.setattr(cache, "_retry_after", 0.0)

@pytest.fixture(autouse=True)
def stub_rag_engine(monkeypatch):
    """
    ASGITransport does not run the lifespan handler, so ``main.rag_engine``
    stays None and every route short-circuits with a 503. Install a stub so
    tests exercise the real request path.
    """
    from app.rag.engine import RAGEngine

    engine = RAGEngine.__new__(RAGEngine)  # skip __init__ (loads an embedder)
    engine.index = None
    engine.chunks = []
    engine._initialized = True
    monkeypatch.setattr(main, "rag_engine", engine)
    return engine

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
