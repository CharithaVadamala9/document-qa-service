"""Dependency wiring.

Providers read from ``app.state``, which is populated once during lifespan.
Tests override these with fakes via ``dependency_overrides`` rather than
patching module globals.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.metrics import Metrics
from app.services.qa_service import QAService


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_metrics(request: Request) -> Metrics:
    metrics: Metrics = request.app.state.metrics
    return metrics


def get_qa_service(request: Request) -> QAService:
    service: QAService | None = getattr(request.app.state, "qa_service", None)
    if service is None:
        # Reached when the process started without a usable API key. The app
        # still serves /health so an orchestrator can distinguish "running but
        # misconfigured" from "crashed".
        raise ConfigurationError(
            "The question-answering service is not configured. Set OPENAI_API_KEY "
            "and restart the service."
        )
    return service


SettingsDep = Annotated[Settings, Depends(get_settings)]
MetricsDep = Annotated[Metrics, Depends(get_metrics)]
QAServiceDep = Annotated[QAService, Depends(get_qa_service)]
