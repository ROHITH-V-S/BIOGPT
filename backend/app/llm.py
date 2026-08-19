"""
LLM client with pluggable backend, model fallback and retry logic.

Two backends, selected by ``LLM_BACKEND``:

* ``openrouter`` — hosted models. Capable, but the free tier allows 50
  requests/day across all ``:free`` models, which is not enough to run the
  evaluation harness even once.
* ``ollama`` — models served locally. Unmetered and offline, at the cost of
  running a much smaller model.

Ollama implements an OpenAI-compatible API, so both share this module's client
code; only the base URL, credential and model list differ. Models within a
backend are tried in order — if one fails (429 / error), the next is attempted.
"""

import logging
import time
from typing import AsyncGenerator

from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt

from app.config import settings
from app.retry import retry_if_transient

logger = logging.getLogger(__name__)


class AllModelsFailedError(RuntimeError):
    """
    Every model in the fallback list failed.

    ``rate_limited`` distinguishes "the free-tier quota is spent" — a
    temporary, user-actionable condition that deserves a 503 and a clear
    message — from a genuine internal fault.
    """

    def __init__(self, message: str, rate_limited: bool = False) -> None:
        super().__init__(message)
        self.rate_limited = rate_limited

# ---------------------------------------------------------------------------
# Backend selection (OpenAI SDK pointed at OpenRouter or a local Ollama)
# ---------------------------------------------------------------------------
#: Clients are cached per backend rather than built at import time: the eval
#: harness switches ``LLM_BACKEND`` after import, and building at import would
#: pin whichever backend happened to be configured first.
_clients: dict[str, AsyncOpenAI] = {}


def active_backend() -> str:
    backend = settings.LLM_BACKEND.lower()
    if backend not in ("openrouter", "ollama"):
        raise ValueError(
            f"Unknown LLM_BACKEND {settings.LLM_BACKEND!r} — "
            "expected 'openrouter' or 'ollama'."
        )
    return backend


def active_models() -> list[str]:
    """The fallback chain for the active backend."""
    if active_backend() == "ollama":
        return settings.OLLAMA_MODEL_FALLBACK_LIST
    return settings.LLM_MODEL_FALLBACK_LIST


def _get_client() -> AsyncOpenAI:
    backend = active_backend()
    if backend not in _clients:
        if backend == "ollama":
            _clients[backend] = AsyncOpenAI(
                api_key=settings.OLLAMA_API_KEY,
                base_url=settings.OLLAMA_BASE_URL,
                # A cold Ollama start loads several GB into VRAM; the SDK's
                # default 10-minute timeout is generous enough, but the default
                # 2 retries would triple an already slow first call.
                max_retries=0,
            )
        else:
            _clients[backend] = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
            )
    return _clients[backend]


# Track which model was last used successfully (for SSE metadata)
_last_successful_model: str | None = None

SYSTEM_PROMPT = """\
You are BioGPT Explorer, an expert biomedical AI research assistant.

Context blocks come from two kinds of source, and each block is tagged:
• [Local corpus] — passages retrieved from the user's ingested documents.
• [PubMed <PMID>] — the abstract of a published paper.

Guidelines:
• Answer the user's question **accurately and concisely** using ONLY the
  provided context.
• If the context does not contain enough information, clearly state that.
• Use scientific terminology appropriately.
• Structure your answer with clear paragraphs.
• Cite the source of each claim by its tag — e.g. "[Chunk 2]" for local
  passages, "[PubMed 34567890]" for abstracts. Never attribute a statement
  to a source that does not support it.
• Do NOT fabricate information beyond what is in the context.
"""


def _build_messages(question: str, context_chunks: list[str]) -> list[dict]:
    """Build the chat messages array for the LLM call."""
    numbered_chunks = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    user_content = (
        f"=== Retrieved Context ===\n{numbered_chunks}\n\n"
        f"=== Question ===\n{question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


@retry(
    retry=retry_if_transient,
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _call_llm(model: str, messages: list[dict], stream: bool = False):
    """Call a single model with retry-on-failure."""
    logger.info("Calling model: %s (stream=%s)", model, stream)
    return await _get_client().chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
    )


def _get_successful_model() -> str | None:
    """Return the model that was last used successfully."""
    return _last_successful_model


async def generate_answer(
    question: str, context_chunks: list[str], models: list[str] | None = None
) -> str:
    """
    Generate a non-streaming answer, trying each fallback model in order.

    ``models`` overrides the backend's fallback chain. The eval harness uses it
    to judge with a different model than the one that produced the answer.
    """
    global _last_successful_model
    messages = _build_messages(question, context_chunks)
    chain = models or active_models()
    last_exc: Exception | None = None
    all_rate_limited = True

    for model in chain:
        try:
            t0 = time.monotonic()
            response = await _call_llm(model, messages, stream=False)
            elapsed = time.monotonic() - t0
            _last_successful_model = model
            logger.info("Model %s responded in %.2fs", model, elapsed)
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Model %s failed: %s", model, exc)
            last_exc = exc
            all_rate_limited &= isinstance(exc, RateLimitError)

    raise AllModelsFailedError(
        f"All {len(chain)} {active_backend()} models failed. "
        f"Last error: {last_exc}",
        rate_limited=all_rate_limited,
    )


async def stream_answer(
    question: str, context_chunks: list[str]
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from the LLM, trying each fallback model in order.
    """
    global _last_successful_model
    messages = _build_messages(question, context_chunks)
    chain = active_models()
    last_exc: Exception | None = None
    all_rate_limited = True

    for model in chain:
        try:
            t0 = time.monotonic()
            response_stream = await _call_llm(model, messages, stream=True)
            _last_successful_model = model
            logger.info("Streaming from model: %s", model)

            async for chunk in response_stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

            elapsed = time.monotonic() - t0
            logger.info("Stream from %s completed in %.2fs", model, elapsed)
            return  # success — exit the generator
        except Exception as exc:
            logger.warning("Model %s failed: %s", model, exc)
            last_exc = exc
            all_rate_limited &= isinstance(exc, RateLimitError)

    raise AllModelsFailedError(
        f"All {len(chain)} {active_backend()} models failed. "
        f"Last error: {last_exc}",
        rate_limited=all_rate_limited,
    )
