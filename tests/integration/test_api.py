"""End-to-end tests through the ASGI application.

Real routing, middleware, validation, serialisation and exception handling.
The only substitutions are the embedder and the model, injected as fakes, so
no test needs an API key or a network connection.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from app.api.dependencies import get_qa_service
from app.core.config import Settings
from app.core.models import NOT_FOUND_TEXT
from app.services.cache import DocumentCache
from app.services.qa_service import QAService
from tests.fixtures import factory
from tests.fixtures.fakes import FakeAnswerGenerator, FakeEmbedder

ENDPOINT = "/api/v1/qa"


def _files(
    document: bytes, questions: list[str], name: str = "soc2.pdf", mime: str = "application/pdf"
) -> dict[str, tuple[str, bytes, str]]:
    return {
        "document_file": (name, document, mime),
        "questions_file": (
            "questions.json",
            factory.build_questions_json(questions),
            "application/json",
        ),
    }


class TestHappyPath:
    async def test_pdf_returns_an_answer_per_question(
        self, client: httpx.AsyncClient, soc2_pdf: bytes, sample_questions: list[str]
    ) -> None:
        response = await client.post(ENDPOINT, files=_files(soc2_pdf, sample_questions))
        assert response.status_code == 200

        body = response.json()
        assert body["document"] == "soc2.pdf"
        assert [r["question"] for r in body["results"]] == sample_questions
        assert body["metadata"]["questions_processed"] == len(sample_questions)
        assert body["metadata"]["document_type"] == "pdf"

    async def test_response_matches_the_documented_schema(
        self, client: httpx.AsyncClient, soc2_pdf: bytes
    ) -> None:
        response = await client.post(
            ENDPOINT, files=_files(soc2_pdf, ["What monitoring is performed?"])
        )
        body = response.json()

        assert set(body) == {"document", "results", "metadata"}
        assert set(body["results"][0]) == {
            "question",
            "answer",
            "status",
            "citations",
            "latency_ms",
            "error_code",
        }
        assert set(body["metadata"]) == {
            "questions_processed",
            "answered",
            "not_found",
            "failed",
            "document_type",
            "chunk_count",
            "document_cache_hit",
            "processing_time_ms",
            "usage",
        }
        assert set(body["metadata"]["usage"]) == {
            "input_tokens",
            "output_tokens",
            "embedding_tokens",
            "total_tokens",
            "estimated_cost_usd",
        }

    async def test_citations_carry_page_numbers_from_metadata(
        self, client: httpx.AsyncClient, soc2_pdf: bytes
    ) -> None:
        response = await client.post(
            ENDPOINT, files=_files(soc2_pdf, ["What are the incident notification SLAs?"])
        )
        answered = [r for r in response.json()["results"] if r["status"] == "answered"]
        assert answered, "expected at least one answered question"

        citation = answered[0]["citations"][0]
        assert citation["source"] == "soc2.pdf"
        assert citation["pages"] and all(isinstance(p, int) for p in citation["pages"])
        assert citation["snippet"]
        assert citation["chunk_id"]

    async def test_json_document_is_supported(
        self, client: httpx.AsyncClient, security_json: bytes
    ) -> None:
        response = await client.post(
            ENDPOINT,
            files=_files(
                security_json,
                ["Which cloud providers and regions are used?"],
                name="controls.json",
                mime="application/json",
            ),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["metadata"]["document_type"] == "json"

        answered = [r for r in body["results"] if r["status"] == "answered"]
        assert answered[0]["citations"][0]["json_paths"]

    async def test_unsupported_question_returns_not_found_not_a_guess(
        self, client: httpx.AsyncClient, soc2_pdf: bytes
    ) -> None:
        response = await client.post(
            ENDPOINT, files=_files(soc2_pdf, [factory.UNANSWERABLE_QUESTION])
        )
        result = response.json()["results"][0]
        assert result["status"] == "not_found"
        assert result["answer"] == NOT_FOUND_TEXT
        assert result["citations"] == []

    async def test_many_questions_in_one_request(
        self, client: httpx.AsyncClient, soc2_pdf: bytes, settings: Settings
    ) -> None:
        questions = [f"Question {i} about access control and monitoring?" for i in range(25)]
        response = await client.post(ENDPOINT, files=_files(soc2_pdf, questions))
        assert response.status_code == 200
        assert response.json()["metadata"]["questions_processed"] == 25

    async def test_repeat_request_reuses_the_cached_index(
        self, client: httpx.AsyncClient, soc2_pdf: bytes
    ) -> None:
        first = await client.post(ENDPOINT, files=_files(soc2_pdf, ["monitoring?"]))
        second = await client.post(ENDPOINT, files=_files(soc2_pdf, ["access?"]))

        assert first.json()["metadata"]["document_cache_hit"] is False
        assert second.json()["metadata"]["document_cache_hit"] is True
        assert (
            second.json()["metadata"]["usage"]["embedding_tokens"]
            < first.json()["metadata"]["usage"]["embedding_tokens"]
        )


class TestErrorHandling:
    async def test_missing_both_files(self, client: httpx.AsyncClient) -> None:
        response = await client.post(ENDPOINT)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert set(error["detail"]["fields"]) == {"document_file", "questions_file"}

    async def test_missing_questions_file(self, client: httpx.AsyncClient, soc2_pdf: bytes) -> None:
        response = await client.post(
            ENDPOINT, files={"document_file": ("soc2.pdf", soc2_pdf, "application/pdf")}
        )
        assert response.status_code == 422
        assert "questions_file" in response.json()["error"]["message"]

    async def test_unsupported_document_type(
        self, client: httpx.AsyncClient, soc2_pdf: bytes
    ) -> None:
        response = await client.post(
            ENDPOINT,
            files=_files(b"just some plain text", ["q?"], name="notes.txt", mime="text/plain"),
        )
        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_file_type"

    async def test_html_claiming_to_be_a_pdf_is_rejected(self, client: httpx.AsyncClient) -> None:
        # A PDF URL that redirects to an error page must not ingest silently.
        html = b"<!doctype html><html><body>Not found</body></html>" * 40
        response = await client.post(ENDPOINT, files=_files(html, ["q?"]))
        assert response.status_code in (415, 422)

    async def test_encrypted_pdf(self, client: httpx.AsyncClient) -> None:
        response = await client.post(ENDPOINT, files=_files(factory.build_encrypted_pdf(), ["q?"]))
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "encrypted_document"

    async def test_scanned_pdf_says_so(self, client: httpx.AsyncClient) -> None:
        response = await client.post(ENDPOINT, files=_files(factory.build_scanned_pdf(), ["q?"]))
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "no_extractable_text"
        assert "scanned" in body["error"]["message"]

    @pytest.mark.parametrize(
        ("questions_payload", "expected_code"),
        [
            (b'{"questions": ', "malformed_document"),
            (b"[]", "validation_error"),
            (b"null", "validation_error"),
            (b'[""]', "validation_error"),
        ],
    )
    async def test_bad_questions_file(
        self,
        client: httpx.AsyncClient,
        soc2_pdf: bytes,
        questions_payload: bytes,
        expected_code: str,
    ) -> None:
        response = await client.post(
            ENDPOINT,
            files={
                "document_file": ("soc2.pdf", soc2_pdf, "application/pdf"),
                "questions_file": ("q.json", questions_payload, "application/json"),
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == expected_code

    async def test_too_many_questions(
        self, client: httpx.AsyncClient, soc2_pdf: bytes, settings: Settings
    ) -> None:
        questions = ["q?"] * (settings.max_questions + 1)
        response = await client.post(ENDPOINT, files=_files(soc2_pdf, questions))
        assert response.status_code == 422
        assert str(settings.max_questions) in response.json()["error"]["message"]

    async def test_empty_document(self, client: httpx.AsyncClient) -> None:
        response = await client.post(ENDPOINT, files=_files(b"", ["q?"]))
        assert response.status_code == 422

    async def test_oversized_upload_rejected_by_content_length(
        self, client: httpx.AsyncClient, settings: Settings
    ) -> None:
        oversized = b"%PDF-1.7\n" + b"0" * (settings.max_file_size_bytes + 1024)
        response = await client.post(ENDPOINT, files=_files(oversized, ["q?"]))
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    async def test_errors_never_leak_internals(
        self, client: httpx.AsyncClient, soc2_pdf: bytes
    ) -> None:
        response = await client.post(ENDPOINT, files=_files(factory.build_encrypted_pdf(), ["q?"]))
        body = json.dumps(response.json()).lower()
        for leak in ("traceback", 'file "/', '.py", line', "openai_api_key", "sk-"):
            assert leak not in body

    async def test_service_unconfigured_returns_a_clear_error(
        self, app: FastAPI, soc2_pdf: bytes
    ) -> None:
        # No override and no API key: the endpoint must explain the cause
        # rather than fail obscurely.
        app.dependency_overrides.pop(get_qa_service, None)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.post(ENDPOINT, files=_files(soc2_pdf, ["q?"]))
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "configuration_error"


class TestOperationalEndpoints:
    async def test_health_does_not_depend_on_the_provider(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_readiness_reports_missing_configuration(self, client: httpx.AsyncClient) -> None:
        # The test app never runs lifespan, so no service is wired.
        assert (await client.get("/ready")).json()["status"] == "not_ready"

    async def test_request_id_is_returned_and_echoed(self, client: httpx.AsyncClient) -> None:
        generated = await client.get("/health")
        assert generated.headers["X-Request-ID"]

        supplied = await client.get("/health", headers={"X-Request-ID": "trace-me-123"})
        assert supplied.headers["X-Request-ID"] == "trace-me-123"

    async def test_metrics_reflect_traffic(
        self, client: httpx.AsyncClient, soc2_pdf: bytes
    ) -> None:
        await client.post(ENDPOINT, files=_files(soc2_pdf, ["monitoring?", "access?"]))
        snapshot = (await client.get("/metrics")).json()

        assert snapshot["requests_total"] >= 1
        assert snapshot["questions_total"] >= 2
        assert snapshot["tokens"]["total"] > 0
        assert snapshot["latency_ms"]["p95"] >= 0

    async def test_ui_is_served(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Document QA Service" in response.text

    async def test_openapi_schema_is_valid(self, client: httpx.AsyncClient) -> None:
        schema = (await client.get("/openapi.json")).json()
        assert ENDPOINT in schema["paths"]


class TestConcurrency:
    async def test_parallel_requests_are_served(
        self, client: httpx.AsyncClient, soc2_pdf: bytes
    ) -> None:
        responses = await asyncio.gather(
            *(
                client.post(ENDPOINT, files=_files(soc2_pdf, [f"question {i} on access?"]))
                for i in range(8)
            )
        )
        assert all(r.status_code == 200 for r in responses)

    async def test_load_is_shed_when_saturated(self, settings: Settings, soc2_pdf: bytes) -> None:
        # A single in-flight slot makes the second concurrent request shed.
        from app.main import create_app

        tight = settings.model_copy(update={"max_concurrent_requests": 1})
        app = create_app(tight)
        slow = QAService(
            embedder=FakeEmbedder(),
            generator=FakeAnswerGenerator(delay=0.2),
            settings=tight,
            cache=DocumentCache(4),
        )
        app.dependency_overrides[get_qa_service] = lambda: slow

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            responses = await asyncio.gather(
                *(c.post(ENDPOINT, files=_files(soc2_pdf, ["monitoring?"])) for _ in range(4))
            )

        statuses = [r.status_code for r in responses]
        assert 503 in statuses, "expected admission control to shed load"
        shed = next(r for r in responses if r.status_code == 503)
        assert shed.json()["error"]["code"] == "service_overloaded"
        assert shed.headers["Retry-After"] == "5"
