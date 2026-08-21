"""Orchestrates ingest, indexing and concurrent answering."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.errors import DocQAError, UpstreamError
from app.core.logging import Timer, get_logger
from app.core.models import (
    NOT_FOUND_TEXT,
    AnswerStatus,
    Citation,
    SourceType,
    TokenUsage,
)
from app.core.tokens import estimate_cost_usd
from app.ingestion.loader import load_document
from app.llm.client import AnswerGenerator
from app.retrieval.embedder import Embedder
from app.retrieval.retriever import DocumentIndex
from app.services.cache import CachedDocument, DocumentCache, cache_key

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QuestionAnswer:
    question: str
    answer: str
    status: AnswerStatus
    citations: list[Citation] = field(default_factory=list)
    latency_ms: float = 0.0
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class QAResult:
    document: str
    source_type: SourceType
    results: list[QuestionAnswer]
    usage: TokenUsage
    cost_usd: float
    latency_ms: float
    chunk_count: int
    cache_hit: bool

    @property
    def answered(self) -> int:
        return sum(1 for r in self.results if r.status is AnswerStatus.ANSWERED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status is AnswerStatus.ERROR)


class QAService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        generator: AnswerGenerator,
        settings: Settings,
        cache: DocumentCache,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._settings = settings
        self._cache = cache

    async def answer(self, *, data: bytes, filename: str, questions: Sequence[str]) -> QAResult:
        if not questions:
            raise ValueError("answer requires at least one question")

        timer = Timer()
        key = cache_key(data, self._settings)
        cached, was_cached = await self._cache.get_or_create(
            key, lambda: self._build(data, filename)
        )

        semaphore = asyncio.Semaphore(self._settings.max_concurrent_questions)
        # gather preserves input order, so results align with the questions as
        # submitted regardless of the order they finish in.
        results = await asyncio.gather(
            *(self._answer_one(cached.index, q, semaphore) for q in questions)
        )

        usage = sum((r.usage for r in results), start=TokenUsage()) + TokenUsage(
            # Index-building tokens are only charged to the request that paid
            # for them; a cache hit spent nothing on embedding the document.
            embedding_tokens=0 if was_cached else cached.index.embedding_tokens
        )
        answers = [r.answer for r in results]

        if all(a.status is AnswerStatus.ERROR for a in answers):
            raise UpstreamError(
                "No question could be answered. The model service is failing or "
                "every question exceeded its time limit."
            )

        cost = estimate_cost_usd(
            llm_model=self._settings.llm_model,
            embedding_model=self._settings.embedding_model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            embedding_tokens=usage.embedding_tokens,
        )
        result = QAResult(
            document=cached.document.source,
            source_type=cached.document.source_type,
            results=answers,
            usage=usage,
            cost_usd=cost,
            latency_ms=timer.ms,
            chunk_count=len(cached.document.chunks),
            cache_hit=was_cached,
        )
        logger.info(
            "qa.completed",
            document_type=result.source_type,
            question_count=len(questions),
            answered=result.answered,
            not_found=len(questions) - result.answered - result.failed,
            failed=result.failed,
            chunk_count=result.chunk_count,
            cache_hit=was_cached,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            embedding_tokens=usage.embedding_tokens,
            cost_usd=cost,
            total_latency_ms=result.latency_ms,
        )
        return result

    async def _build(self, data: bytes, filename: str) -> CachedDocument:
        # PyMuPDF and JSON parsing are blocking C/CPU work; keeping them off the
        # event loop is what allows other requests to progress during an ingest.
        document = await asyncio.to_thread(
            load_document, data, filename=filename, settings=self._settings
        )
        index = await DocumentIndex.build(
            document.chunks, embedder=self._embedder, settings=self._settings
        )
        return CachedDocument(document=document, index=index)

    async def _answer_one(
        self, index: DocumentIndex, question: str, semaphore: asyncio.Semaphore
    ) -> _AnswerOutcome:
        timer = Timer()
        async with semaphore:
            try:
                async with asyncio.timeout(self._settings.question_timeout_seconds):
                    retrieved = await index.retrieve(question)
                    generated = await self._generator.generate(question, retrieved.chunks)
            except TimeoutError:
                logger.warning("question.timeout", latency_ms=timer.ms)
                return _AnswerOutcome(
                    _error(
                        question,
                        "question_timeout",
                        "This question timed out before an answer was produced.",
                        timer.ms,
                    ),
                    TokenUsage(),
                )
            except DocQAError as exc:
                # One question failing must not discard the others; the failure
                # is reported in its own result entry.
                logger.warning("question.failed", error_code=exc.code, latency_ms=timer.ms)
                return _AnswerOutcome(
                    _error(question, exc.code, exc.message, timer.ms), TokenUsage()
                )

        usage = generated.usage + TokenUsage(embedding_tokens=retrieved.query_tokens)
        return _AnswerOutcome(
            QuestionAnswer(
                question=question,
                answer=generated.answer or NOT_FOUND_TEXT,
                status=generated.status,
                citations=generated.citations,
                latency_ms=timer.ms,
            ),
            usage,
        )


@dataclass(frozen=True, slots=True)
class _AnswerOutcome:
    answer: QuestionAnswer
    usage: TokenUsage


def _error(question: str, code: str, message: str, latency_ms: float) -> QuestionAnswer:
    return QuestionAnswer(
        question=question,
        answer=message,
        status=AnswerStatus.ERROR,
        latency_ms=latency_ms,
        error_code=code,
    )
