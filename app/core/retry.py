"""Retry policy for OpenAI calls, shared by embeddings and chat.

Only transient faults are retried. Retrying a bad request or a bad key burns
the attempt budget and delays the error the caller needs to see.
"""

from __future__ import annotations

import openai
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.core.config import Settings

# Jittered backoff: synchronised retries from concurrent questions would
# otherwise arrive together and re-trigger the same rate limit.
_RETRYABLE: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_FATAL: tuple[type[Exception], ...] = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
    openai.NotFoundError,
)


def retry_policy(settings: Settings) -> AsyncRetrying:
    return AsyncRetrying(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_random_exponential(
            multiplier=settings.llm_backoff_base_seconds,
            max=settings.llm_backoff_max_seconds,
        ),
        stop=stop_after_attempt(settings.llm_max_attempts),
        reraise=True,
    )


def is_fatal(exc: BaseException) -> bool:
    return isinstance(exc, _FATAL)
