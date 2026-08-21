"""Structured JSON logging.

``request_id`` is bound to a context variable so it propagates into every log
line of a request, including across asyncio fan-out and worker threads.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.core.config import Settings
from app.core.errors import ConfigurationError

_LEVELS = logging.getLevelNamesMapping()


def configure_logging(settings: Settings) -> None:
    """Route structlog and stdlib records through one JSON stdout handler, so
    uvicorn does not emit unstructured text into the same stream."""
    level = _LEVELS.get(settings.log_level.upper())
    if level is None:
        raise ConfigurationError(f"Unknown log level {settings.log_level!r}.")
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    # Applied to records that did not originate from structlog.
    foreign_pre_chain: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.stdlib.ExtraAdder(),
    ]

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    render: list[structlog.typing.Processor]
    if settings.log_format == "json":
        # JSONRenderer cannot serialise a raw exc_info tuple, so it must be
        # formatted into a string first. ConsoleRenderer does this itself.
        render = [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    else:
        render = [structlog.dev.ConsoleRenderer(colors=True)]

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=foreign_pre_chain,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, *render],
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "httpx", "httpcore"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    # Superseded by our own middleware, which records the fields we query on.
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def new_request_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def request_context(**values: Any) -> Iterator[None]:
    """Bind values to every log line emitted inside the block."""
    bind_contextvars(**values)
    try:
        yield
    finally:
        clear_contextvars()


class Timer:
    """Latency stopwatch. Uses perf_counter so NTP steps cannot skew it."""

    __slots__ = ("_elapsed", "_start")

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._elapsed: float | None = None

    def stop(self) -> float:
        self._elapsed = time.perf_counter() - self._start
        return self._elapsed

    @property
    def ms(self) -> float:
        elapsed = self._elapsed if self._elapsed is not None else time.perf_counter() - self._start
        return round(elapsed * 1000, 2)


@contextmanager
def timed(logger: structlog.stdlib.BoundLogger, event: str, **fields: Any) -> Iterator[Timer]:
    """Log ``event`` with latency, or ``event.failed`` and re-raise."""
    timer = Timer()
    try:
        yield timer
    except Exception as exc:
        logger.warning(f"{event}.failed", latency_ms=timer.ms, error=type(exc).__name__, **fields)
        raise
    else:
        logger.info(event, latency_ms=timer.ms, **fields)
