from __future__ import annotations

import re
from functools import lru_cache

from fastapi import status

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from pydantic import SecretStr
from groq import RateLimitError

from app.config import get_settings


class LLMConfigError(RuntimeError):
  pass


@lru_cache
def get_llm() -> Runnable:
  settings = get_settings()
  if not settings.groq_api_key:
    raise LLMConfigError("GROQ_API_KEY is not set. Add it to .env and restart the server.")
  groq_llm = ChatGroq(
    model = settings.groq_model,
    api_key = SecretStr(settings.groq_api_key),
    temperature = 0.2,
    max_tokens=1024,
    max_retries = 0,
    timeout = 60,
    rate_limiter=InMemoryRateLimiter(
      requests_per_second=0.4,
      check_every_n_seconds=0.1,
      max_bucket_size=1,
    ),
    reasoning_format = "hidden",
    reasoning_effort = settings.groq_reasoning_effort,
  )

  if not settings.gemini_api_key:
    return groq_llm

  gemini_llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=SecretStr(settings.gemini_api_key),
    temperature=0.2,
    max_output_tokens=1024,
    max_retries=0,
  )

  return groq_llm.with_fallbacks([gemini_llm], exceptions_to_handle=(RateLimitError,))


def strip_think(text: str) -> str:
  """Safety net in case a model still emits <think>...</think>."""
  return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def describe__llm_error(exc: Exception) -> tuple[int, str]:
  """Map provider exceptions to (http_status, user-facing message)."""
  if isinstance(exc, LLMConfigError):
    return status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)
  name = type(exc).__name__
  if "RateLimit" in name:
    return status.HTTP_429_TOO_MANY_REQUESTS, "Groq's free-tier rate limit was hit. Wait a few seconds and try again."
  if "Authentication" in name or "PermissionDenied" in name:
    return status.HTTP_502_BAD_GATEWAY, "Groq rejected the API key. Check GROQ_API_KEY in .env."
  if "Timeout" in name or "Connection" in name:
    return status.HTTP_504_GATEWAY_TIMEOUT, "Couldn't reach Groq. Check your connection and try again."
  return status.HTTP_500_INTERNAL_SERVER_ERROR, "The language model request failed. Please try again."