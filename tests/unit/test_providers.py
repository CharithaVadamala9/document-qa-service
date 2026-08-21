"""Provider adapters: batching, retries and error translation.

The OpenAI client is stubbed rather than called. These tests cover the code
that only runs when the provider misbehaves, which is precisely the code that
cannot be exercised by a happy-path live run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.errors import ConfigurationError, UpstreamError, UpstreamTimeout
from app.core.models import AnswerStatus, Chunk, SourceType
from app.llm.client import AnswerSchema, OpenAIAnswerGenerator
from app.retrieval.embedder import OpenAIEmbedder
from app.retrieval.vector_store import ScoredChunk

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def _rate_limit() -> openai.RateLimitError:
    return openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=_REQUEST), body=None
    )


def _auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError(
        "bad key", response=httpx.Response(401, request=_REQUEST), body=None
    )


class StubEmbeddings:
    """Records calls and replays scripted outcomes."""

    def __init__(
        self, dimensions: int = 8, *, raises: list[Exception] | None = None, shuffle: bool = False
    ) -> None:
        self.dimensions = dimensions
        self.batch_sizes: list[int] = []
        self._raises = raises or []
        self._shuffle = shuffle

    # Signature mirrors openai.resources.Embeddings.create, timeout included.
    async def create(
        self,
        *,
        model: str,
        input: list[str],
        dimensions: int,
        timeout: float,  # noqa: ASYNC109 - mirrors the SDK, not our own API
    ) -> Any:
        self.batch_sizes.append(len(input))
        if self._raises:
            raise self._raises.pop(0)

        # Deliberately ignores the requested `dimensions` so a mismatch
        # between what we ask for and what we get can be simulated.
        data = [
            SimpleNamespace(index=i, embedding=[float(i + 1)] * self.dimensions)
            for i in range(len(input))
        ]
        if self._shuffle:
            data.reverse()  # the API does not promise ordering
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=7 * len(input)))


def _client(embeddings: StubEmbeddings) -> Any:
    return SimpleNamespace(embeddings=embeddings)


@pytest.fixture
def embed_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "embedding_dimensions": 8,
            "embedding_batch_size": 10,
            "llm_backoff_base_seconds": 0.001,
            "llm_backoff_max_seconds": 0.002,
        }
    )


class TestOpenAIEmbedder:
    async def test_batches_documents(self, embed_settings: Settings) -> None:
        stub = StubEmbeddings()
        embedder = OpenAIEmbedder(_client(stub), embed_settings)

        result = await embedder.embed_documents([f"text {i}" for i in range(25)])

        # 25 texts at a batch size of 10 is three calls, not twenty-five.
        assert stub.batch_sizes == [10, 10, 5]
        assert result.vectors.shape == (25, 8)
        assert result.tokens == 7 * 25

    async def test_restores_order_when_the_api_returns_it_shuffled(
        self, embed_settings: Settings
    ) -> None:
        stub = StubEmbeddings(shuffle=True)
        embedder = OpenAIEmbedder(_client(stub), embed_settings)

        result = await embedder.embed_documents(["a", "b", "c"])

        # Each stub vector encodes its own index, so misordering is visible.
        assert [row[0] for row in result.vectors] == [1.0, 2.0, 3.0]

    async def test_rejects_wrong_dimensions(self, embed_settings: Settings) -> None:
        embedder = OpenAIEmbedder(_client(StubEmbeddings(dimensions=4)), embed_settings)
        with pytest.raises(ConfigurationError, match="dimensional"):
            await embedder.embed_documents(["a"])

    async def test_retries_transient_failures_then_succeeds(self, embed_settings: Settings) -> None:
        stub = StubEmbeddings(raises=[_rate_limit(), _rate_limit()])
        embedder = OpenAIEmbedder(_client(stub), embed_settings)

        result = await embedder.embed_documents(["a"])

        assert len(stub.batch_sizes) == 3, "expected two retries before success"
        assert result.vectors.shape == (1, 8)

    async def test_gives_up_after_the_attempt_budget(self, embed_settings: Settings) -> None:
        attempts = embed_settings.llm_max_attempts
        stub = StubEmbeddings(raises=[_rate_limit() for _ in range(attempts + 2)])
        embedder = OpenAIEmbedder(_client(stub), embed_settings)

        with pytest.raises(UpstreamError, match="unavailable"):
            await embedder.embed_documents(["a"])
        assert len(stub.batch_sizes) == attempts

    async def test_authentication_failure_is_not_retried(self, embed_settings: Settings) -> None:
        # Retrying a bad key wastes the budget and delays the real diagnosis.
        stub = StubEmbeddings(raises=[_auth_error()])
        embedder = OpenAIEmbedder(_client(stub), embed_settings)

        with pytest.raises(ConfigurationError, match="API key"):
            await embedder.embed_documents(["a"])
        assert len(stub.batch_sizes) == 1

    async def test_timeout_is_reported_as_such(self, embed_settings: Settings) -> None:
        stub = StubEmbeddings(raises=[openai.APITimeoutError(request=_REQUEST)] * 8)
        embedder = OpenAIEmbedder(_client(stub), embed_settings)

        with pytest.raises(UpstreamTimeout, match="did not respond"):
            await embedder.embed_documents(["a"])

    async def test_rejects_empty_input(self, embed_settings: Settings) -> None:
        embedder = OpenAIEmbedder(_client(StubEmbeddings()), embed_settings)
        with pytest.raises(ValueError):
            await embedder.embed_documents([])
        with pytest.raises(ValueError):
            await embedder.embed_query("   ")


def _scored(index: int) -> ScoredChunk:
    chunk = Chunk(
        id=f"c{index}",
        text=f"Extract {index}: notification occurs within seventy-two (72) hours.",
        index=index,
        token_count=5,
        source="soc2.pdf",
        source_type=SourceType.PDF,
        pages=(index + 1,),
    )
    return ScoredChunk(chunk=chunk, score=0.9)


def _raw(input_tokens: int = 120, output_tokens: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens}
    )


@pytest.fixture
def generator(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> OpenAIAnswerGenerator:
    # model_copy skips validation, so the SecretStr must be constructed here.
    configured = settings.model_copy(update={"openai_api_key": SecretStr("sk-test-not-real")})
    return OpenAIAnswerGenerator(configured)


class TestOpenAIAnswerGenerator:
    async def test_requires_an_api_key(self, settings: Settings) -> None:
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            OpenAIAnswerGenerator(settings)

    async def test_no_context_skips_the_model_call(
        self, generator: OpenAIAnswerGenerator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = False

        async def _fail(_messages: object) -> dict[str, Any]:
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(generator, "_invoke", _fail)
        answer = await generator.generate("anything?", [])

        assert answer.status is AnswerStatus.NOT_FOUND
        assert called is False, "an empty context cannot yield a grounded answer"
        assert answer.usage.total_tokens == 0

    async def test_answered_response_is_cited_from_metadata(
        self, generator: OpenAIAnswerGenerator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parsed = AnswerSchema(status="answered", answer="Within 72 hours.", sources=[2])

        async def _reply(_messages: object) -> dict[str, Any]:
            return {"parsed": parsed, "raw": _raw()}

        monkeypatch.setattr(generator, "_invoke", _reply)
        answer = await generator.generate("SLA?", [_scored(0), _scored(1)])

        assert answer.status is AnswerStatus.ANSWERED
        assert answer.answer == "Within 72 hours."
        assert [c.chunk_id for c in answer.citations] == ["c1"]
        assert answer.citations[0].pages == [2]
        assert answer.usage.prompt_tokens == 120

    async def test_answered_without_a_valid_source_is_downgraded(
        self, generator: OpenAIAnswerGenerator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This is the guarantee behind "do not fabricate": an uncited claim is
        # reported as not found rather than passed through.
        parsed = AnswerSchema(status="answered", answer="Made up.", sources=[99])

        async def _reply(_messages: object) -> dict[str, Any]:
            return {"parsed": parsed, "raw": _raw()}

        monkeypatch.setattr(generator, "_invoke", _reply)
        answer = await generator.generate("q?", [_scored(0)])

        assert answer.status is AnswerStatus.NOT_FOUND
        assert answer.citations == []
        # Usage is still charged: the call really was made.
        assert answer.usage.prompt_tokens == 120

    async def test_empty_answer_text_is_treated_as_not_found(
        self, generator: OpenAIAnswerGenerator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parsed = AnswerSchema(status="answered", answer="   ", sources=[1])

        async def _reply(_messages: object) -> dict[str, Any]:
            return {"parsed": parsed, "raw": _raw()}

        monkeypatch.setattr(generator, "_invoke", _reply)
        answer = await generator.generate("q?", [_scored(0)])
        assert answer.status is AnswerStatus.NOT_FOUND

    async def test_unparseable_response_raises(
        self, generator: OpenAIAnswerGenerator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _reply(_messages: object) -> dict[str, Any]:
            return {"parsed": None, "raw": _raw(), "parsing_error": "bad json"}

        monkeypatch.setattr(generator, "_invoke", _reply)
        with pytest.raises(UpstreamError, match="could not be parsed"):
            await generator.generate("q?", [_scored(0)])

    async def test_missing_usage_metadata_defaults_to_zero(
        self, generator: OpenAIAnswerGenerator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parsed = AnswerSchema(status="answered", answer="Yes.", sources=[1])

        async def _reply(_messages: object) -> dict[str, Any]:
            return {"parsed": parsed, "raw": SimpleNamespace()}

        monkeypatch.setattr(generator, "_invoke", _reply)
        answer = await generator.generate("q?", [_scored(0)])
        assert answer.usage.total_tokens == 0
