"""
Retry-policy tests.

The distinction under test is not cosmetic: on a metered free tier every retry
is itself a counted request, so retrying a 429 that cannot clear until midnight
spends three requests from a spent budget to learn what the first reply said.
"""

import httpx
import pytest
from openai import APIConnectionError, BadRequestError, RateLimitError

from app.retry import _is_transient, is_daily_quota_error


def _rate_limit_error(message: str) -> RateLimitError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return RateLimitError(
        message, response=httpx.Response(429, request=request), body=None
    )


DAILY = (
    "Error code: 429 - {'error': {'message': 'Rate limit exceeded: "
    "free-models-per-day. Add 10 credits to unlock 1000 free model requests "
    "per day', 'metadata': {'limit_source': 'openrouter_free_tier_daily'}}}"
)
BURST = "Error code: 429 - {'error': {'message': 'Rate limit exceeded: 20 per minute'}}"


def test_daily_quota_429_is_not_retried():
    assert is_daily_quota_error(_rate_limit_error(DAILY)) is True
    assert _is_transient(_rate_limit_error(DAILY)) is False


def test_short_window_429_is_still_retried():
    assert is_daily_quota_error(_rate_limit_error(BURST)) is False
    assert _is_transient(_rate_limit_error(BURST)) is True


def test_connection_errors_are_retried():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    assert _is_transient(APIConnectionError(request=request)) is True


def test_deterministic_4xx_is_not_retried():
    """A bad model id fails identically every time — retrying only adds latency."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    exc = BadRequestError(
        "invalid model id", response=httpx.Response(400, request=request), body=None
    )
    assert _is_transient(exc) is False


@pytest.mark.parametrize("message", [DAILY, BURST])
def test_non_rate_limit_types_never_classified_as_quota(message):
    assert is_daily_quota_error(ValueError(message)) is False
