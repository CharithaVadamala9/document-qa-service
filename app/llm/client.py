"""Answer generation against retrieved context.

LangChain's ChatOpenAI is used here (unlike embeddings) because it exposes
``usage_metadata`` and handles the structured-output plumbing. Its own retry
loop is disabled so that our policy in ``app.core.retry`` is the only one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import openai
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.errors import ConfigurationError, UpstreamError, UpstreamTimeout
from app.core.logging import Timer, get_logger
from app.core.models import NOT_FOUND_TEXT, AnswerStatus, Citation, TokenUsage
from app.core.retry import is_fatal, retry_policy
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.vector_store import ScoredChunk

logger = get_logger(__name__)


class AnswerSchema(BaseModel):
    """Structured output contract enforced by the provider."""

    status: Literal["answered", "not_found"] = Field(
        description="answered only when the extracts fully support the answer"
    )
    answer: str = Field(description="The answer, or an empty string when not_found")
    sources: list[int] = Field(
        default_factory=list, description="Numbers of the extracts actually used"
    )


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    status: AnswerStatus
    answer: str
    citations: list[Citation]
    usage: TokenUsage
    latency_ms: float


class AnswerGenerator(Protocol):
    async def generate(self, question: str, chunks: Sequence[ScoredChunk]) -> GeneratedAnswer: ...


def _not_found(latency_ms: float = 0.0, usage: TokenUsage | None = None) -> GeneratedAnswer:
    return GeneratedAnswer(
        status=AnswerStatus.NOT_FOUND,
        answer=NOT_FOUND_TEXT,
        citations=[],
        usage=usage or TokenUsage(),
        latency_ms=latency_ms,
    )


def resolve_citations(sources: Sequence[int], chunks: Sequence[ScoredChunk]) -> list[Citation]:
    """Map extract numbers to citations built from chunk metadata.

    Out-of-range numbers are dropped rather than trusted: the list is model
    output, and a hallucinated index must not become a citation.
    """
    citations: list[Citation] = []
    seen: set[str] = set()
    for number in sources:
        if not 1 <= number <= len(chunks):
            logger.warning("answer.invalid_source_index", index=number, available=len(chunks))
            continue
        chunk = chunks[number - 1].chunk
        if chunk.id not in seen:
            seen.add(chunk.id)
            citations.append(chunk.citation())
    return citations


class OpenAIAnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_openai_key:
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
        self._model = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            max_completion_tokens=settings.max_answer_tokens,
            timeout=settings.llm_timeout_seconds,
            api_key=settings.openai_api_key,
            max_retries=0,  # our retry policy owns this
        ).with_structured_output(AnswerSchema, include_raw=True, strict=True)

    async def generate(self, question: str, chunks: Sequence[ScoredChunk]) -> GeneratedAnswer:
        # No retrieved context means no possible grounded answer, so skip the
        # call entirely rather than pay for a guaranteed "not found".
        if not chunks:
            logger.info("answer.skipped_no_context")
            return _not_found()

        timer = Timer()
        messages = [("system", SYSTEM_PROMPT), ("human", build_user_prompt(question, chunks))]
        result = await self._invoke(messages)

        parsed: AnswerSchema | None = result.get("parsed")
        if parsed is None:
            raise UpstreamError("The model returned a response that could not be parsed.")

        usage = self._usage(result.get("raw"))
        citations = resolve_citations(parsed.sources, chunks)

        # A claimed answer with no valid citation is not grounded, so it is
        # reported as not found rather than passed through uncited.
        if parsed.status == "answered" and not citations:
            logger.warning("answer.downgraded_uncited")
            return _not_found(timer.ms, usage)
        if parsed.status == "not_found" or not parsed.answer.strip():
            return _not_found(timer.ms, usage)

        return GeneratedAnswer(
            status=AnswerStatus.ANSWERED,
            answer=parsed.answer.strip(),
            citations=citations,
            usage=usage,
            latency_ms=timer.ms,
        )

    async def _invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        response: dict[str, Any] | None = None
        async with self._semaphore:
            try:
                async for attempt in retry_policy(self._settings):
                    with attempt:
                        response = await self._model.ainvoke(messages)
            except openai.APITimeoutError as exc:
                raise UpstreamTimeout("The model did not respond in time.") from exc
            except Exception as exc:
                if is_fatal(exc):
                    raise ConfigurationError(
                        "The model request was rejected. Check the API key and "
                        "the configured model name."
                    ) from exc
                raise UpstreamError("The model service is unavailable.") from exc

        if response is None:  # pragma: no cover - retry always yields an attempt
            raise UpstreamError("The model returned no response.")
        return response

    @staticmethod
    def _usage(raw: Any) -> TokenUsage:
        metadata = getattr(raw, "usage_metadata", None) or {}
        return TokenUsage(
            prompt_tokens=int(metadata.get("input_tokens", 0)),
            completion_tokens=int(metadata.get("output_tokens", 0)),
        )
