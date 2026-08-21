"""Request correlation, access logging and admission control."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import Settings
from app.core.errors import PayloadTooLarge, ServiceOverloaded
from app.core.logging import Timer, get_logger, new_request_id, request_context

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

Next = Callable[[Request], Awaitable[Response]]


def _error_response(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {"code": code, "message": message, "detail": {}},
            "request_id": request_id,
        },
        headers={REQUEST_ID_HEADER: request_id},
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds a request id to every log line and records access latency."""

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        timer = Timer()

        with request_context(request_id=request_id):
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id

            metrics = getattr(request.app.state, "metrics", None)
            if metrics is not None:
                metrics.record_request(status_code=response.status_code, latency_ms=timer.ms)

            log = logger.info if response.status_code < 500 else logger.error
            log(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                total_latency_ms=timer.ms,
            )
            return response


class AdmissionMiddleware(BaseHTTPMiddleware):
    """Sheds load and oversized uploads before any work is done.

    Rejecting at the door keeps latency bounded for requests already in
    flight, which is preferable to degrading all of them equally.
    """

    def __init__(self, app: object, settings: Settings) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._settings = settings
        self._in_flight = 0

    def _limit_bytes(self) -> int:
        return self._settings.max_file_size_bytes + self._settings.max_questions_file_bytes

    async def dispatch(self, request: Request, call_next: Next) -> Response:
        request_id = getattr(request.state, "request_id", "-")

        # Content-Length lets us reject an oversized body before reading it
        # into memory. Absent on chunked uploads, where the loader's own size
        # check is the backstop.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._limit_bytes():
            too_large = PayloadTooLarge(
                f"The upload exceeds the combined size limit of "
                f"{self._limit_bytes() // (1024 * 1024)} MB."
            )
            logger.warning("http.rejected_oversized", declared_bytes=int(declared))
            return _error_response(
                too_large.http_status, too_large.code, too_large.message, request_id
            )

        if self._in_flight >= self._settings.max_concurrent_requests:
            overloaded = ServiceOverloaded("The service is at capacity. Retry shortly.")
            logger.warning("http.shed_load", in_flight=self._in_flight)
            response = _error_response(
                overloaded.http_status, overloaded.code, overloaded.message, request_id
            )
            response.headers["Retry-After"] = "5"
            return response

        self._in_flight += 1
        try:
            return await call_next(request)
        finally:
            self._in_flight -= 1
