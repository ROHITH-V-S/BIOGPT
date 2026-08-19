"""
Optional Redis cache.

Redis is a genuine optimisation here, not decoration. Three things in this
pipeline are expensive in ways a cache fixes:

* **Embeddings** cost OpenRouter quota (50 requests/day on the free tier) and
  are perfectly deterministic for a given (model, text) pair. Re-embedding the
  same corpus across eval runs was the single largest consumer of that budget.
* **PubMed E-utilities** calls take ~1-2s and NCBI asks clients not to hammer
  the endpoint. The same topic query is issued repeatedly during corpus builds.
* **Rate-limit counters** live in a per-process dict today, so they reset on
  restart and cannot coordinate across workers.

Design rule: **Redis is optional and never load-bearing.** If it is not
installed, not configured, or goes away mid-run, every caller falls back to the
uncached path. A cache that can take down the app is worse than no cache, and
requiring Redis for local development would be a bad trade for a project that
otherwise runs with two commands.
"""

import hashlib
import json
import logging
import time
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_client: Any = None
_state: str = "unprobed"  # unprobed | ready | unavailable
#: Monotonic deadline before which a failed Redis is not re-probed.
_retry_after: float = 0.0


def _key(*parts: str) -> str:
    """Namespaced cache key. Long/unsafe parts are hashed, not embedded raw."""
    return "biogpt:" + ":".join(parts)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


async def get_client():
    """
    Return a connected Redis client, or ``None`` if unavailable.

    A successful connection is reused. A failure disables Redis for
    ``REDIS_RETRY_COOLDOWN_SECONDS`` rather than for the life of the process:
    retrying a dead endpoint on every request would add a timeout to every
    request, but latching the failure permanently would mean a brief Redis blip
    silently disables caching until the next deploy. The cooldown is the middle
    ground — one probe per interval, and self-healing.
    """
    global _client, _state

    if _state == "ready":
        return _client
    if _state == "unavailable":
        if time.monotonic() < _retry_after:
            return None
        logger.info("Cache cooldown elapsed — re-probing Redis.")
        _state = "unprobed"

    if not settings.REDIS_ENABLED:
        logger.info("Cache disabled (REDIS_ENABLED=false).")
        _mark_unavailable(cooldown=False)
        return None

    try:
        from redis import asyncio as aioredis
    except ImportError:
        logger.info("redis package not installed — running without cache.")
        _mark_unavailable(cooldown=False)
        return None

    try:
        client = aioredis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=settings.REDIS_TIMEOUT_SECONDS,
            socket_timeout=settings.REDIS_TIMEOUT_SECONDS,
        )
        await client.ping()
    except Exception as exc:
        logger.warning(
            "Redis unavailable at %s (%s) — continuing without cache.",
            settings.REDIS_URL, type(exc).__name__,
        )
        _mark_unavailable()
        return None

    logger.info("Cache ready: %s", settings.REDIS_URL)
    _client, _state = client, "ready"
    return _client


def _mark_unavailable(cooldown: bool = True) -> None:
    """
    Mark the cache unavailable, optionally scheduling a re-probe.

    ``cooldown=False`` is for permanent conditions — the package is not
    installed, or caching is switched off — where re-probing can never help.
    """
    global _state, _retry_after
    _state = "unavailable"
    _retry_after = (
        time.monotonic() + settings.REDIS_RETRY_COOLDOWN_SECONDS
        if cooldown
        else float("inf")
    )


def _disable(exc: Exception) -> None:
    """Trip the breaker after a runtime failure on a previously-good client."""
    logger.warning(
        "Redis error (%s) — disabling cache for %ss.",
        type(exc).__name__, settings.REDIS_RETRY_COOLDOWN_SECONDS,
    )
    _mark_unavailable()


async def close() -> None:
    """Release the connection pool (called from the app's lifespan shutdown)."""
    global _client, _state, _retry_after
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:  # pragma: no cover - shutdown best effort
            pass
    _client, _state, _retry_after = None, "unprobed", 0.0


def is_active() -> bool:
    """Whether the cache is currently serving (for /health)."""
    return _state == "ready"


# ---------------------------------------------------------------------------
# Embedding vectors
# ---------------------------------------------------------------------------
#: Vectors are stored as raw little-endian float32 rather than JSON: a 2048-dim
#: vector is 8 KB packed against ~40 KB as text, and np.frombuffer decodes it
#: without a parse step.
async def get_embedding(model: str, text: str) -> np.ndarray | None:
    client = await get_client()
    if client is None:
        return None
    try:
        raw = await client.get(_key("emb", model, digest(text)))
    except Exception as exc:
        _disable(exc)
        return None
    if raw is None:
        return None
    return np.frombuffer(raw, dtype="<f4")


async def set_embedding(model: str, text: str, vector: np.ndarray) -> None:
    client = await get_client()
    if client is None:
        return
    try:
        await client.set(
            _key("emb", model, digest(text)),
            vector.astype("<f4").tobytes(),
            ex=settings.CACHE_TTL_EMBEDDINGS,
        )
    except Exception as exc:
        _disable(exc)


# ---------------------------------------------------------------------------
# JSON payloads (PubMed responses)
# ---------------------------------------------------------------------------
async def get_json(namespace: str, key: str) -> Any | None:
    client = await get_client()
    if client is None:
        return None
    try:
        raw = await client.get(_key(namespace, digest(key)))
    except Exception as exc:
        _disable(exc)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        # A poisoned or format-changed entry must not break the caller.
        logger.warning("Discarding undecodable cache entry in %s.", namespace)
        return None


async def set_json(namespace: str, key: str, value: Any, ttl: int | None = None) -> None:
    client = await get_client()
    if client is None:
        return
    try:
        await client.set(
            _key(namespace, digest(key)),
            json.dumps(value),
            ex=ttl or settings.CACHE_TTL_PUBMED,
        )
    except Exception as exc:
        _disable(exc)
