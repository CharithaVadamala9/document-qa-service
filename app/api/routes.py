"""HTTP endpoints.

Handlers validate, delegate to the service layer, and shape the response.
No parsing, chunking, retrieval or model orchestration happens here.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, File, Request, UploadFile

from app.api.dependencies import MetricsDep, QAServiceDep, SettingsDep
from app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    QAResponse,
    ReadinessResponse,
)
from app.core.errors import PayloadTooLarge, RequestTimeout, ValidationError
from app.core.logging import get_logger
from app.services.questions import parse_questions

logger = get_logger(__name__)

router = APIRouter()

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    502: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


async def _read_upload(upload: UploadFile, field: str, limit: int) -> bytes:
    """Per-field size check.

    The middleware can only police the combined body length, so a single
    oversized part still has to be caught here, and with the same 413 that
    the middleware would have returned.
    """
    data = await upload.read()
    if not data:
        raise ValidationError(f"{field} is empty.", field=field)
    if len(data) > limit:
        raise PayloadTooLarge(
            f"{field} is {len(data) // 1024} KB, which exceeds its limit of {limit // 1024} KB.",
            field=field,
            limit_bytes=limit,
            size_bytes=len(data),
        )
    return data


@router.post(
    "/api/v1/qa",
    response_model=QAResponse,
    responses=_ERROR_RESPONSES,
    summary="Answer questions from a document",
)
async def answer_questions(
    service: QAServiceDep,
    settings: SettingsDep,
    metrics: MetricsDep,
    document_file: Annotated[UploadFile, File(description="Source document: PDF or JSON")],
    questions_file: Annotated[UploadFile, File(description="JSON array of questions")],
) -> QAResponse:
    document = await _read_upload(document_file, "document_file", settings.max_file_size_bytes)
    raw_questions = await _read_upload(
        questions_file, "questions_file", settings.max_questions_file_bytes
    )
    questions = parse_questions(raw_questions, settings)

    try:
        async with asyncio.timeout(settings.request_timeout_seconds):
            result = await service.answer(
                data=document,
                filename=document_file.filename or "document",
                questions=questions,
            )
    except TimeoutError as exc:
        raise RequestTimeout(
            f"The request exceeded the {settings.request_timeout_seconds:.0f}s "
            "limit. Try fewer questions or a smaller document."
        ) from exc

    metrics.record_answers(
        questions=len(result.results),
        cache_hit=result.cache_hit,
        input_tokens=result.usage.prompt_tokens,
        output_tokens=result.usage.completion_tokens,
        embedding_tokens=result.usage.embedding_tokens,
        cost_usd=result.cost_usd,
    )
    return QAResponse.from_domain(result)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    """Process is up. Deliberately does not touch OpenAI: a provider outage
    must not cause an orchestrator to restart healthy containers."""
    return HealthResponse(status="ok", environment=settings.environment)


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def ready(request: Request) -> ReadinessResponse:
    if getattr(request.app.state, "qa_service", None) is None:
        return ReadinessResponse(status="not_ready", detail="OPENAI_API_KEY is not configured.")
    return ReadinessResponse(status="ready", detail="Accepting requests.")


@router.get("/metrics", summary="Process metrics")
async def get_metrics_snapshot(metrics: MetricsDep) -> dict[str, Any]:
    return metrics.snapshot()
