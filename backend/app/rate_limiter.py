"""
Per-IP rate limiting, Redis-backed with an in-process fallback.

The original limiter kept counters in a per-process ``defaultdict``. That has two
defects that only show up in the configuration this project actually ships:

* **It resets on restart.** Any redeploy cleared every counter.
* **It does not coordinate across workers.** ``uvicorn --workers 4`` gave each
  worker its own dict, so the effective limit was 4× the configured value —
  silently, and precisely under the load where the limit matters.

Redis fixes both, since the counter lives outside the process. It stays optional:
without Redis the in-process limiter still applies, which is the correct
degradation — losing coordination is bad, losing rate limiting entirely is worse.
"""

import asyncio
import logging
import time
from collections import defaultdict

from fastapi import HTTPException, Request, status

from app import cache
from app.config import settings

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0


class InProcessLimiter:
    """Sliding-window limiter using local memory. Fallback when Redis is absent."""

    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.window = WINDOW_SECONDS
        self.history: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def hit(self, client_ip: str) -> int | None:
        """Record a request. Returns retry-after seconds if the limit is exceeded."""
        now = time.monotonic()
        async with self._lock:
            recent = [t for t in self.history[client_ip] if now - t < self.window]
            self.history[client_ip] = recent

            if len(recent) >= self.requests_per_minute:
                return max(1, int(self.window - (now - recent[0])))

            recent.append(now)
            return None


class RedisLimiter:
    """
    Sliding-window limiter backed by a Redis sorted set.

    One key per client, scored by timestamp: drop entries older than the window,
    count what remains, add the current request. Executed as a pipeline so the
    sequence costs a single round trip.
    """

    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.window = WINDOW_SECONDS

    async def hit(self, client, client_ip: str) -> int | None:
        now = time.time()
        key = f"biogpt:ratelimit:{client_ip}"

        pipe = client.pipeline()
        pipe.zremrangebyscore(key, 0, now - self.window)
        pipe.zcard(key)
        pipe.zadd(key, {f"{now}:{id(pipe)}": now})
        # Expire the key so idle clients do not accumulate forever.
        pipe.expire(key, int(self.window) + 1)
        _, count, _, _ = await pipe.execute()

        if count >= self.requests_per_minute:
            # Roll back this request's own entry: a rejected request should not
            # extend the window it was rejected by.
            try:
                await client.zremrangebyscore(key, now, now)
            except Exception:  # pragma: no cover - best effort
                pass
            oldest = await client.zrange(key, 0, 0, withscores=True)
            if oldest:
                return max(1, int(self.window - (now - oldest[0][1])))
            return int(self.window)
        return None


_in_process = InProcessLimiter(settings.RATE_LIMIT_REQUESTS_PER_MINUTE)
_redis_limiter = RedisLimiter(settings.RATE_LIMIT_REQUESTS_PER_MINUTE)


async def check(request: Request) -> None:
    if settings.RATE_LIMIT_REQUESTS_PER_MINUTE <= 0:
        return

    client_ip = request.client.host if request.client else "unknown"

    retry_after: int | None
    redis_client = await cache.get_client()
    if redis_client is not None:
        try:
            retry_after = await _redis_limiter.hit(redis_client, client_ip)
        except Exception as exc:
            logger.warning(
                "Redis rate-limit check failed (%s) — falling back to in-process.",
                type(exc).__name__,
            )
            retry_after = await _in_process.hit(client_ip)
    else:
        retry_after = await _in_process.hit(client_ip)

    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too Many Requests",
            headers={"Retry-After": str(retry_after)},
        )


async def rate_limit_dependency(request: Request):
    await check(request)
