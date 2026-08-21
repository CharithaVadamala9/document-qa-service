"""Embedding generation.

Uses the OpenAI client directly rather than LangChain's wrapper because the
wrapper discards the ``usage`` field, and per-request embedding token counts
are needed for cost reporting.

Documents are embedded in batches and the batches run concurrently: one API
call per chunk is the single biggest avoidable source of ingest latency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import openai
from numpy.typing import NDArray

from app.core.config import Settings
from app.core.errors import ConfigurationError, UpstreamError, UpstreamTimeout
from app.core.logging import get_logger
from app.core.retry import is_fatal, retry_policy

logger = get_logger(__name__)

Vectors = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: Vectors  # shape (n, dimensions)
    tokens: int


@runtime_checkable
class Embedder(Protocol):
    """Implemented by the OpenAI embedder and by the test fake."""

    @property
    def dimensions(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult: ...

    async def embed_query(self, text: str) -> EmbeddingResult: ...


class OpenAIEmbedder:
    def __init__(self, client: openai.AsyncOpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)

    @property
    def dimensions(self) -> int:
        return self._settings.embedding_dimensions

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            raise ValueError("embed_documents requires at least one text")

        size = self._settings.embedding_batch_size
        batches = [texts[i : i + size] for i in range(0, len(texts), size)]
        results = await asyncio.gather(*(self._embed_batch(b) for b in batches))

        # gather preserves batch order, so concatenation matches input order.
        vectors = np.vstack([r.vectors for r in results])
        tokens = sum(r.tokens for r in results)
        logger.info(
            "embeddings.generated",
            texts=len(texts),
            batches=len(batches),
            tokens=tokens,
        )
        return EmbeddingResult(vectors=vectors, tokens=tokens)

    async def embed_query(self, text: str) -> EmbeddingResult:
        if not text.strip():
            raise ValueError("embed_query requires non-empty text")
        return await self._embed_batch([text])

    async def _embed_batch(self, texts: Sequence[str]) -> EmbeddingResult:
        response: openai.types.CreateEmbeddingResponse | None = None
        async with self._semaphore:
            try:
                async for attempt in retry_policy(self._settings):
                    with attempt:
                        response = await self._client.embeddings.create(
                            model=self._settings.embedding_model,
                            input=list(texts),
                            dimensions=self._settings.embedding_dimensions,
                            timeout=self._settings.llm_timeout_seconds,
                        )
            except openai.APITimeoutError as exc:
                raise UpstreamTimeout("The embedding service did not respond in time.") from exc
            except Exception as exc:
                if is_fatal(exc):
                    raise ConfigurationError(
                        "The embedding request was rejected. Check the API key "
                        "and the configured embedding model."
                    ) from exc
                raise UpstreamError("The embedding service is unavailable.") from exc

        if response is None:  # pragma: no cover - retry always yields an attempt
            raise UpstreamError("The embedding service returned no response.")

        # Order within a batch is not guaranteed by the API contract; each item
        # carries its own index, so sort rather than trust arrival order.
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(texts):
            raise UpstreamError(
                "The embedding service returned a different number of vectors than were requested."
            )

        vectors = np.asarray([item.embedding for item in items], dtype=np.float32)
        if vectors.shape[1] != self.dimensions:
            raise ConfigurationError(
                f"Expected {self.dimensions}-dimensional embeddings but received "
                f"{vectors.shape[1]}."
            )
        return EmbeddingResult(vectors=vectors, tokens=response.usage.prompt_tokens)
