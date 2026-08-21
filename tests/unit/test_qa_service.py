from __future__ import annotations

import asyncio

import pytest

from app.core.config import Settings
from app.core.errors import UnsupportedFileType, UpstreamError
from app.core.models import AnswerStatus, SourceType
from app.ingestion.loader import detect_source_type, sanitise_filename
from app.services.cache import DocumentCache, cache_key
from app.services.qa_service import QAService
from tests.fixtures.fakes import FakeAnswerGenerator, FakeEmbedder


def _service(
    settings: Settings, generator: FakeAnswerGenerator, embedder: FakeEmbedder, cache_size: int = 8
) -> QAService:
    return QAService(
        embedder=embedder,
        generator=generator,
        settings=settings,
        cache=DocumentCache(cache_size),
    )


class TestSourceDetection:
    def test_detects_by_content_not_extension(self) -> None:
        assert detect_source_type(b"%PDF-1.7\n...") is SourceType.PDF
        assert detect_source_type(b'{"a": 1}') is SourceType.JSON
        assert detect_source_type(b"  [1, 2]") is SourceType.JSON
        assert detect_source_type(b"\xef\xbb\xbf{}") is SourceType.JSON

    def test_rejects_anything_else(self) -> None:
        with pytest.raises(UnsupportedFileType):
            detect_source_type(b"plain text, not a document")

    @pytest.mark.parametrize(
        ("supplied", "expected"),
        [
            ("../../etc/passwd", "etc_passwd".replace("etc_passwd", "passwd")),
            ("/absolute/path/report.pdf", "report.pdf"),
            ("..\\..\\windows\\system32\\a.json", "a.json"),
            ("", "document"),
            ("...", "document"),
        ],
    )
    def test_filename_is_reduced_to_a_safe_label(self, supplied: str, expected: str) -> None:
        assert sanitise_filename(supplied) == expected


class TestCacheKey:
    def test_same_bytes_same_key(self, settings: Settings) -> None:
        assert cache_key(b"abc", settings) == cache_key(b"abc", settings)

    def test_chunking_settings_are_part_of_the_key(self, settings: Settings) -> None:
        # Otherwise a tuning change would silently serve an index built under
        # the previous values.
        retuned = settings.model_copy(update={"chunk_size_tokens": 400})
        assert cache_key(b"abc", settings) != cache_key(b"abc", retuned)


class TestDocumentCache:
    async def test_builds_once_and_reuses(self) -> None:
        cache = DocumentCache(4)
        builds = 0

        async def factory() -> object:
            nonlocal builds
            builds += 1
            return object()

        first, hit_a = await cache.get_or_create("k", factory)  # type: ignore[arg-type]
        second, hit_b = await cache.get_or_create("k", factory)  # type: ignore[arg-type]

        assert builds == 1
        assert first is second
        assert (hit_a, hit_b) == (False, True)

    async def test_concurrent_callers_share_one_build(self) -> None:
        cache = DocumentCache(4)
        builds = 0

        async def factory() -> object:
            nonlocal builds
            builds += 1
            await asyncio.sleep(0.05)
            return object()

        results = await asyncio.gather(
            *(cache.get_or_create("k", factory) for _ in range(5))  # type: ignore[arg-type]
        )
        assert builds == 1, "a stampede must not embed the document five times"
        assert len({id(entry) for entry, _ in results}) == 1

    async def test_evicts_least_recently_used(self) -> None:
        cache = DocumentCache(2)

        async def factory() -> object:
            return object()

        await cache.get_or_create("a", factory)  # type: ignore[arg-type]
        await cache.get_or_create("b", factory)  # type: ignore[arg-type]
        await cache.get_or_create("a", factory)  # type: ignore[arg-type]  # refresh 'a'
        await cache.get_or_create("c", factory)  # type: ignore[arg-type]  # evicts 'b'

        assert cache.size == 2
        _, hit_b = await cache.get_or_create("b", factory)  # type: ignore[arg-type]
        assert hit_b is False

    async def test_zero_size_disables_caching(self) -> None:
        cache = DocumentCache(0)
        builds = 0

        async def factory() -> object:
            nonlocal builds
            builds += 1
            return object()

        await cache.get_or_create("k", factory)  # type: ignore[arg-type]
        _, hit = await cache.get_or_create("k", factory)  # type: ignore[arg-type]
        assert builds == 2
        assert hit is False

    async def test_failed_build_is_not_cached(self) -> None:
        cache = DocumentCache(4)

        async def failing() -> object:
            raise UpstreamError("boom")

        with pytest.raises(UpstreamError):
            await cache.get_or_create("k", failing)  # type: ignore[arg-type]
        assert cache.size == 0


class TestQAService:
    async def test_answers_every_question_in_order(
        self, soc2_pdf: bytes, settings: Settings, sample_questions: list[str]
    ) -> None:
        service = _service(settings, FakeAnswerGenerator(), FakeEmbedder())
        result = await service.answer(
            data=soc2_pdf, filename="soc2.pdf", questions=sample_questions
        )
        assert [r.question for r in result.results] == sample_questions

    async def test_order_preserved_when_questions_finish_out_of_order(
        self, soc2_pdf: bytes, settings: Settings
    ) -> None:
        questions = [f"Question {i} about monitoring and access?" for i in range(6)]
        service = _service(settings, FakeAnswerGenerator(delay=0.01), FakeEmbedder())
        result = await service.answer(data=soc2_pdf, filename="soc2.pdf", questions=questions)
        assert [r.question for r in result.results] == questions

    async def test_bounded_concurrency(self, soc2_pdf: bytes, settings: Settings) -> None:
        generator = FakeAnswerGenerator(delay=0.02)
        service = _service(settings, generator, FakeEmbedder())
        await service.answer(
            data=soc2_pdf,
            filename="soc2.pdf",
            questions=[f"Question {i} about access control?" for i in range(20)],
        )
        assert generator.concurrency_peak <= settings.max_concurrent_questions
        assert generator.concurrency_peak > 1, "questions should run concurrently"

    async def test_unanswerable_question_returns_not_found(
        self, soc2_pdf: bytes, settings: Settings
    ) -> None:
        service = _service(settings, FakeAnswerGenerator(), FakeEmbedder())
        result = await service.answer(
            data=soc2_pdf,
            filename="soc2.pdf",
            questions=["What was the chief executive's total compensation package?"],
        )
        assert result.results[0].status is AnswerStatus.NOT_FOUND
        assert result.results[0].citations == []

    async def test_one_failing_question_does_not_sink_the_others(
        self, soc2_pdf: bytes, settings: Settings
    ) -> None:
        questions = ["First about monitoring?", "Second about access?", "Third about backups?"]
        generator = FakeAnswerGenerator(fail_questions=frozenset({questions[1]}))
        service = _service(settings, generator, FakeEmbedder())

        result = await service.answer(data=soc2_pdf, filename="soc2.pdf", questions=questions)
        assert result.failed == 1
        assert result.results[1].status is AnswerStatus.ERROR
        assert result.results[1].error_code == "upstream_error"
        survivors = (result.results[0], result.results[2])
        assert all(r.status is not AnswerStatus.ERROR for r in survivors)

    async def test_total_failure_raises_rather_than_reporting_success(
        self, soc2_pdf: bytes, settings: Settings
    ) -> None:
        questions = ["a?", "b?"]
        generator = FakeAnswerGenerator(fail_questions=frozenset(questions))
        service = _service(settings, generator, FakeEmbedder())

        # Returning 200 with every entry failed would misreport a total outage.
        with pytest.raises(UpstreamError):
            await service.answer(data=soc2_pdf, filename="soc2.pdf", questions=questions)

    async def test_a_timed_out_question_does_not_delay_the_others(
        self, soc2_pdf: bytes, settings: Settings
    ) -> None:
        questions = ["slow monitoring question?", "fast access question?"]
        impatient = settings.model_copy(
            update={"question_timeout_seconds": 0.1, "llm_timeout_seconds": 0.1}
        )
        generator = FakeAnswerGenerator(slow_questions=frozenset({questions[0]}))
        service = _service(impatient, generator, FakeEmbedder())

        result = await service.answer(data=soc2_pdf, filename="soc2.pdf", questions=questions)
        assert result.results[0].status is AnswerStatus.ERROR
        assert result.results[0].error_code == "question_timeout"
        assert result.results[1].status is not AnswerStatus.ERROR

    async def test_every_question_timing_out_raises(
        self, soc2_pdf: bytes, settings: Settings
    ) -> None:
        questions = ["slow a?", "slow b?"]
        impatient = settings.model_copy(
            update={"question_timeout_seconds": 0.1, "llm_timeout_seconds": 0.1}
        )
        service = _service(
            impatient, FakeAnswerGenerator(slow_questions=frozenset(questions)), FakeEmbedder()
        )
        with pytest.raises(UpstreamError):
            await service.answer(data=soc2_pdf, filename="soc2.pdf", questions=questions)

    async def test_second_request_reuses_the_index(
        self, soc2_pdf: bytes, settings: Settings
    ) -> None:
        embedder = FakeEmbedder()
        service = _service(settings, FakeAnswerGenerator(), embedder)

        first = await service.answer(data=soc2_pdf, filename="a.pdf", questions=["monitoring?"])
        second = await service.answer(data=soc2_pdf, filename="a.pdf", questions=["access?"])

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert embedder.document_calls == 1
        # A cache hit must not be billed for embedding the document again.
        assert second.usage.embedding_tokens < first.usage.embedding_tokens

    async def test_json_documents_are_answered_with_path_citations(
        self, security_json: bytes, settings: Settings
    ) -> None:
        service = _service(settings, FakeAnswerGenerator(), FakeEmbedder())
        result = await service.answer(
            data=security_json,
            filename="controls.json",
            questions=["Which cloud providers and regions are used for hosting?"],
        )
        assert result.source_type is SourceType.JSON
        answered = result.results[0]
        assert answered.status is AnswerStatus.ANSWERED
        assert answered.citations[0].json_paths

    async def test_usage_and_cost_are_reported(self, soc2_pdf: bytes, settings: Settings) -> None:
        service = _service(settings, FakeAnswerGenerator(), FakeEmbedder())
        result = await service.answer(
            data=soc2_pdf, filename="a.pdf", questions=["monitoring?", "access?"]
        )
        assert result.usage.prompt_tokens > 0
        assert result.usage.embedding_tokens > 0
        assert result.cost_usd > 0

    async def test_rejects_empty_question_list(self, soc2_pdf: bytes, settings: Settings) -> None:
        service = _service(settings, FakeAnswerGenerator(), FakeEmbedder())
        with pytest.raises(ValueError):
            await service.answer(data=soc2_pdf, filename="a.pdf", questions=[])
