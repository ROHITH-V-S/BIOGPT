"""
Cache and rate-limiter tests.

The central guarantee is that Redis is optional. Most of these tests assert that
things still work when it is missing or broken, because that is the failure mode
that would otherwise take the whole API down.
"""

import numpy as np
import pytest

from app import cache
from app.config import settings
from app.rate_limiter import InProcessLimiter, RedisLimiter


class FakeRedis:
    """Minimal in-memory stand-in for the calls this project makes."""

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, bytes] = {}
        self.fail = fail
        self.sets = 0
        self.gets = 0

    async def ping(self):
        if self.fail:
            raise ConnectionError("no redis")
        return True

    async def get(self, key):
        self.gets += 1
        if self.fail:
            raise ConnectionError("no redis")
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.sets += 1
        if self.fail:
            raise ConnectionError("no redis")
        self.store[key] = value if isinstance(value, bytes) else str(value).encode()

    async def aclose(self):
        return None


@pytest.fixture
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_client", fake)
    monkeypatch.setattr(cache, "_state", "ready")
    return fake


# ---------------------------------------------------------------------------
# Degradation: the point of the whole design
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_disabled_cache_returns_none_not_error(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_ENABLED", False)
    assert await cache.get_client() is None
    assert await cache.get_embedding("m", "text") is None
    assert await cache.get_json("ns", "k") is None
    # Writes must be no-ops rather than raising.
    await cache.set_embedding("m", "text", np.zeros(4, dtype=np.float32))
    await cache.set_json("ns", "k", {"a": 1})
    assert cache.is_active() is False


def _patch_redis_factory(monkeypatch, factory):
    """
    Replace the async client factory that ``cache`` imports.

    ``cache`` does ``from redis import asyncio as aioredis``, which resolves the
    ``asyncio`` *attribute* of the ``redis`` package — patching
    ``sys.modules["redis.asyncio"]`` does not intercept it.
    """
    import redis

    monkeypatch.setattr(redis, "asyncio", factory)


@pytest.mark.asyncio
async def test_unreachable_redis_disables_cache_without_raising(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_ENABLED", True)

    class Boom:
        @staticmethod
        def from_url(*a, **k):
            raise ConnectionError("refused")

    _patch_redis_factory(monkeypatch, Boom)
    monkeypatch.setattr(cache, "_state", "unprobed")
    assert await cache.get_client() is None
    assert cache.is_active() is False


@pytest.mark.asyncio
async def test_runtime_failure_trips_breaker(monkeypatch):
    """A mid-run Redis failure must disable the cache, not retry forever."""
    fake = FakeRedis(fail=True)
    monkeypatch.setattr(cache, "_client", fake)
    monkeypatch.setattr(cache, "_state", "ready")

    assert await cache.get_embedding("m", "t") is None
    assert cache.is_active() is False

    # Subsequent calls short-circuit rather than hitting the dead client again.
    before = fake.gets
    assert await cache.get_embedding("m", "t") is None
    assert fake.gets == before


@pytest.mark.asyncio
async def test_breaker_reprobes_after_cooldown(monkeypatch):
    """
    A tripped breaker must self-heal. Latching the failure permanently would let
    a momentary Redis blip disable caching until the process restarts.
    """
    monkeypatch.setattr(settings, "REDIS_ENABLED", True)
    monkeypatch.setattr(cache, "_client", FakeRedis(fail=True))
    monkeypatch.setattr(cache, "_state", "ready")

    assert await cache.get_embedding("m", "t") is None
    assert cache.is_active() is False

    healthy = FakeRedis()

    class Recovered:
        @staticmethod
        def from_url(*a, **k):
            return healthy

    _patch_redis_factory(monkeypatch, Recovered)

    # Still inside the cooldown: no re-probe.
    assert await cache.get_client() is None

    # Cooldown elapsed.
    monkeypatch.setattr(cache, "_retry_after", 0.0)
    assert await cache.get_client() is healthy
    assert cache.is_active() is True


@pytest.mark.asyncio
async def test_disabled_by_config_never_reprobes(monkeypatch):
    """
    REDIS_ENABLED=false is permanent for the process, so it must not schedule a
    re-probe — otherwise every request pays a pointless state transition.
    """
    monkeypatch.setattr(settings, "REDIS_ENABLED", False)
    monkeypatch.setattr(cache, "_state", "unprobed")

    assert await cache.get_client() is None
    assert cache._retry_after == float("inf")


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_embedding_round_trip_preserves_values(fake_redis):
    vector = np.array([0.5, -1.25, 3.0, 0.0], dtype=np.float32)
    await cache.set_embedding("model-a", "hello", vector)
    restored = await cache.get_embedding("model-a", "hello")

    assert restored is not None
    np.testing.assert_allclose(restored, vector)


@pytest.mark.asyncio
async def test_embeddings_are_namespaced_by_model(fake_redis):
    """Different encoders have different dimensionality — keys must not collide."""
    await cache.set_embedding("model-a", "hello", np.zeros(4, dtype=np.float32))
    assert await cache.get_embedding("model-b", "hello") is None


@pytest.mark.asyncio
async def test_json_round_trip(fake_redis):
    await cache.set_json("pubmed:search", "brca1", ["1", "2", "3"])
    assert await cache.get_json("pubmed:search", "brca1") == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_undecodable_entry_is_discarded(fake_redis):
    fake_redis.store[cache._key("ns", cache.digest("k"))] = b"{not json"
    assert await cache.get_json("ns", "k") is None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_in_process_limiter_allows_then_blocks():
    limiter = InProcessLimiter(requests_per_minute=3)
    assert [await limiter.hit("1.2.3.4") for _ in range(3)] == [None, None, None]

    retry_after = await limiter.hit("1.2.3.4")
    assert retry_after is not None and 1 <= retry_after <= 60


@pytest.mark.asyncio
async def test_in_process_limiter_isolates_clients():
    limiter = InProcessLimiter(requests_per_minute=1)
    assert await limiter.hit("1.1.1.1") is None
    assert await limiter.hit("2.2.2.2") is None
    assert await limiter.hit("1.1.1.1") is not None


@pytest.mark.asyncio
async def test_redis_limiter_uses_a_shared_counter():
    """
    The reason Redis backs the limiter: two limiter instances (standing in for
    two uvicorn workers) must share one counter. With per-process dicts the
    effective limit was multiplied by the worker count.
    """
    calls = {"zcard": 0}

    class FakePipeline:
        def __init__(self, state):
            self.state = state

        def zremrangebyscore(self, *a, **k):
            return self

        def zcard(self, *a, **k):
            return self

        def zadd(self, key, mapping):
            self.state["count"] += 1
            return self

        def expire(self, *a, **k):
            return self

        async def execute(self):
            calls["zcard"] += 1
            # Report the count as it was *before* this request was added.
            return [0, self.state["count"] - 1, 1, True]

    class FakeRedisZ:
        def __init__(self):
            self.state = {"count": 0}

        def pipeline(self):
            return FakePipeline(self.state)

        async def zremrangebyscore(self, *a, **k):
            self.state["count"] -= 1
            return 1

        async def zrange(self, *a, **k):
            return [(b"x", 0.0)]

    shared = FakeRedisZ()
    worker_a = RedisLimiter(requests_per_minute=2)
    worker_b = RedisLimiter(requests_per_minute=2)

    assert await worker_a.hit(shared, "1.2.3.4") is None
    assert await worker_b.hit(shared, "1.2.3.4") is None
    # Third request across either "worker" is refused — one shared budget.
    assert await worker_a.hit(shared, "1.2.3.4") is not None
