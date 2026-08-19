"""
Shared retry policy for outbound LLM / embedding calls.

The default tenacity policy retried *every* exception, which meant a
permanently-invalid request (bad model id, malformed payload) was still
attempted three times with exponential backoff before failing. Those are
deterministic 4xx errors — retrying them only adds latency.

Not all 429s are alike, either. A per-minute rate limit clears in seconds and is
worth waiting out; a *daily* quota does not clear until midnight UTC, so
retrying is guaranteed to fail — and on OpenRouter's free tier every retry is
itself a counted request. Retrying a daily-quota 429 therefore spends three
requests from an already-exhausted budget to learn what the first response
already said.
"""

import logging

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry_if_exception

logger = logging.getLogger(__name__)

# Errors worth retrying: transport hiccups, rate limits, and upstream 5xx.
TRANSIENT_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

#: Substrings that mark a 429 as a long-window quota rather than a short burst
#: limit. OpenRouter reports these as "free-models-per-day" with
#: ``limit_source: openrouter_free_tier_daily``.
_QUOTA_MARKERS = ("per-day", "per day", "daily", "quota")


def is_daily_quota_error(exc: BaseException) -> bool:
    """True for a 429 that will not clear until the quota window resets."""
    if not isinstance(exc, RateLimitError):
        return False
    return any(marker in str(exc).lower() for marker in _QUOTA_MARKERS)


def _is_transient(exc: BaseException) -> bool:
    if is_daily_quota_error(exc):
        logger.warning(
            "Daily quota exhausted — not retrying (each retry would consume "
            "another request from a spent budget)."
        )
        return False
    if isinstance(exc, TRANSIENT_ERRORS):
        return True
    logger.debug("Not retrying non-transient error: %s", type(exc).__name__)
    return False


#: Use as ``@retry(retry=retry_if_transient, ...)``.
retry_if_transient = retry_if_exception(_is_transient)
