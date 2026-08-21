"""Shared fixtures.

No test requires an API key or network access: provider-backed dependencies
are supplied as fakes through FastAPI's dependency overrides.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from app.api.dependencies import get_qa_service
from app.core.config import Settings
from app.core.models import SourceType
from app.ingestion.chunking import chunk_segments
from app.ingestion.pdf import extract_pdf
from app.main import create_app
from app.services.cache import DocumentCache
from app.services.qa_service import QAService
from tests.fixtures import factory
from tests.fixtures.fakes import FakeAnswerGenerator, FakeEmbedder


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Hermetic settings.

    ``_env_file=None`` and an explicitly blank key stop the suite from picking
    up a developer's real .env, which would make tests pass locally and fail in
    CI (or vice versa) depending on whose machine they run on.
    """
    return Settings(
        _env_file=None,
        environment="test",
        log_level="WARNING",
        log_format="console",
        openai_api_key=SecretStr(""),
    )


# Generating the PDF costs ~100ms, so build it once for the whole session.
@pytest.fixture(scope="session")
def soc2_pdf() -> bytes:
    return factory.build_soc2_pdf()


@pytest.fixture(scope="session")
def security_json() -> bytes:
    return factory.build_security_json()


@pytest.fixture(scope="session")
def sample_questions() -> list[str]:
    return list(factory.SAMPLE_QUESTIONS)


@pytest.fixture
def pdf_chunks(soc2_pdf: bytes, settings: Settings):
    segments, _ = extract_pdf(soc2_pdf, settings)
    return chunk_segments(
        segments, source="soc2.pdf", source_type=SourceType.PDF, settings=settings
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def generator() -> FakeAnswerGenerator:
    return FakeAnswerGenerator()


@pytest.fixture
def qa_service(
    embedder: FakeEmbedder, generator: FakeAnswerGenerator, settings: Settings
) -> QAService:
    return QAService(
        embedder=embedder,
        generator=generator,
        settings=settings,
        cache=DocumentCache(settings.document_cache_size),
    )


@pytest.fixture
def app(settings: Settings, qa_service: QAService) -> Iterator[FastAPI]:
    application = create_app(settings)
    application.dependency_overrides[get_qa_service] = lambda: qa_service
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
