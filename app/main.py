"""Application factory, lifespan and exception handling."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import openai
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.middleware import REQUEST_ID_HEADER, AdmissionMiddleware, RequestContextMiddleware
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.errors import DocQAError
from app.core.logging import configure_logging, get_logger
from app.core.metrics import Metrics
from app.llm.client import OpenAIAnswerGenerator
from app.retrieval.embedder import OpenAIEmbedder
from app.services.cache import DocumentCache
from app.services.qa_service import QAService

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _build_service(settings: Settings) -> QAService | None:
    """Wire the service, or return None when the process has no API key.

    Starting without a key is allowed so that health checks and the UI still
    respond; /ready reports not_ready and /api/v1/qa returns a configuration
    error. Crashing on boot would make a missing secret look like a bad image.
    """
    if not settings.has_openai_key:
        logger.warning("startup.no_api_key", detail="OPENAI_API_KEY is not set")
        return None

    client = openai.AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
        max_retries=0,  # our retry policy owns this
    )
    return QAService(
        embedder=OpenAIEmbedder(client, settings),
        generator=OpenAIAnswerGenerator(settings),
        settings=settings,
        cache=DocumentCache(settings.document_cache_size),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # Metrics are created in create_app so the endpoint works under test
    # clients that do not run lifespan; re-creating here would discard them.
    app.state.qa_service = _build_service(settings)
    logger.info(
        "startup.complete",
        environment=settings.environment,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        ready=app.state.qa_service is not None,
    )
    yield
    logger.info("shutdown.complete")


def _error_payload(
    request: Request, code: str, message: str, detail: dict[str, Any]
) -> dict[str, Any]:
    return {
        "error": {"code": code, "message": message, "detail": detail},
        "request_id": getattr(request.state, "request_id", None),
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="Document QA Service",
        version="0.1.0",
        summary=(
            "Answer questions from a PDF or JSON document using retrieval-augmented generation."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics = Metrics()

    # Added last runs first: correlation must wrap admission so that a shed
    # request still carries a request id in its response and its log line.
    app.add_middleware(AdmissionMiddleware, settings=settings)
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(DocQAError)
    async def handle_domain_error(request: Request, exc: DocQAError) -> JSONResponse:
        log = logger.warning if exc.http_status < 500 else logger.error
        log("request.failed", error_code=exc.code, status=exc.http_status)
        return JSONResponse(
            status_code=exc.http_status,
            content=_error_payload(request, exc.code, exc.message, exc.detail),
            headers={REQUEST_ID_HEADER: getattr(request.state, "request_id", "-")},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        missing = sorted({str(err["loc"][-1]) for err in exc.errors() if err["type"] == "missing"})
        message = (
            f"Missing required form field(s): {', '.join(missing)}."
            if missing
            else "The request could not be validated."
        )
        return JSONResponse(
            status_code=422,
            content=_error_payload(request, "validation_error", message, {"fields": missing}),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Log the type, never echo it: an unexpected exception's message may
        # contain internals or document content.
        logger.exception("request.unhandled_error", error=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                request,
                "internal_error",
                "An unexpected error occurred. The request id can be used to "
                "locate the failure in the logs.",
                {},
            ),
        )

    app.include_router(router)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
