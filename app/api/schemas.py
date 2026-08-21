"""HTTP request and response contracts.

Pydantic is used at this boundary because it is the only place input is
untrusted. Domain code below uses dataclasses.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.models import AnswerStatus, Citation, SourceType
from app.services.qa_service import QAResult, QuestionAnswer


class CitationOut(BaseModel):
    chunk_id: str = Field(description="Stable id of the chunk the answer came from")
    source: str
    snippet: str
    pages: list[int] = Field(default_factory=list, description="1-based PDF pages")
    json_paths: list[str] = Field(default_factory=list, description="e.g. $.controls[3]")
    section: str | None = None

    @classmethod
    def from_domain(cls, citation: Citation) -> CitationOut:
        return cls(
            chunk_id=citation.chunk_id,
            source=citation.source,
            snippet=citation.snippet,
            pages=citation.pages,
            json_paths=citation.json_paths,
            section=citation.section,
        )


class AnswerOut(BaseModel):
    question: str
    answer: str
    status: AnswerStatus
    citations: list[CitationOut] = Field(default_factory=list)
    latency_ms: float
    error_code: str | None = Field(default=None, description="Set only when status is 'error'")

    @classmethod
    def from_domain(cls, answer: QuestionAnswer) -> AnswerOut:
        return cls(
            question=answer.question,
            answer=answer.answer,
            status=answer.status,
            citations=[CitationOut.from_domain(c) for c in answer.citations],
            latency_ms=round(answer.latency_ms, 2),
            error_code=answer.error_code,
        )


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class MetadataOut(BaseModel):
    questions_processed: int
    answered: int
    not_found: int
    failed: int
    document_type: SourceType
    chunk_count: int
    document_cache_hit: bool
    processing_time_ms: float
    usage: UsageOut


class QAResponse(BaseModel):
    document: str
    results: list[AnswerOut]
    metadata: MetadataOut

    @classmethod
    def from_domain(cls, result: QAResult) -> QAResponse:
        not_found = sum(1 for r in result.results if r.status is AnswerStatus.NOT_FOUND)
        return cls(
            document=result.document,
            results=[AnswerOut.from_domain(r) for r in result.results],
            metadata=MetadataOut(
                questions_processed=len(result.results),
                answered=result.answered,
                not_found=not_found,
                failed=result.failed,
                document_type=result.source_type,
                chunk_count=result.chunk_count,
                document_cache_hit=result.cache_hit,
                processing_time_ms=round(result.latency_ms, 2),
                usage=UsageOut(
                    input_tokens=result.usage.prompt_tokens,
                    output_tokens=result.usage.completion_tokens,
                    embedding_tokens=result.usage.embedding_tokens,
                    total_tokens=result.usage.total_tokens,
                    estimated_cost_usd=result.cost_usd,
                ),
            ),
        )


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Uniform error envelope. Never carries stack traces or document text."""

    error: ErrorBody
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    environment: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    detail: str
