"""
OpenRouter LLM client with model fallback and retry logic.

All LLM calls go through OpenRouter's OpenAI-compatible endpoint.
Free-tier models are tried in order; if one fails (429 / error),
the next model in the fallback list is attempted.
"""

import logging
import time
from typing import AsyncGenerator

from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenRouter client (OpenAI SDK pointed at OpenRouter)
# ---------------------------------------------------------------------------
client = AsyncOpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
)

# Track which model was last used successfully (for SSE metadata)
_last_successful_model: str | None = None

SYSTEM_PROMPT = """\
You are BioGPT Explorer, an expert biomedical AI research assistant.

Guidelines:
• Answer the user's question **accurately and concisely** using ONLY the
  provided context chunks.
• If the context does not contain enough information, clearly state that.
• Use scientific terminology appropriately.
• Structure your answer with clear paragraphs.
• When relevant, reference which chunk(s) support your statements.
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
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
)
async def _call_llm(model: str, messages: list[dict], stream: bool = False):
    """Call a single model with retry-on-failure."""
    logger.info("Calling model: %s (stream=%s)", model, stream)
    return await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
    )


def _get_successful_model() -> str | None:
    """Return the model that was last used successfully."""
    return _last_successful_model


async def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Generate a non-streaming answer, trying each fallback model in order.
    """
    global _last_successful_model
    messages = _build_messages(question, context_chunks)
    last_exc: Exception | None = None

    for model in settings.LLM_MODEL_FALLBACK_LIST:
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

    raise RuntimeError(
        f"All {len(settings.LLM_MODEL_FALLBACK_LIST)} fallback models "
        f"failed. Last error: {last_exc}"
    )


async def stream_answer(
    question: str, context_chunks: list[str]
) -> AsyncGenerator[str, None]:
    """
    Stream tokens from the LLM, trying each fallback model in order.
    """
    global _last_successful_model
    messages = _build_messages(question, context_chunks)
    last_exc: Exception | None = None

    for model in settings.LLM_MODEL_FALLBACK_LIST:
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

    raise RuntimeError(
        f"All {len(settings.LLM_MODEL_FALLBACK_LIST)} fallback models "
        f"failed. Last error: {last_exc}"
    )
