"""Test doubles for the provider-backed dependencies.

The embedder is a hashing vectoriser rather than a random or constant stub, so
lexical overlap still produces vector similarity. That makes retrieval ordering
meaningful in tests: a question about monitoring genuinely ranks the monitoring
chunk first, without any network call.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Sequence

import numpy as np

from app.core.errors import UpstreamError
from app.core.models import NOT_FOUND_TEXT, AnswerStatus, TokenUsage
from app.llm.client import GeneratedAnswer, resolve_citations
from app.retrieval.embedder import EmbeddingResult
from app.retrieval.vector_store import ScoredChunk

_WORD = re.compile(r"[a-z0-9]+")
# Kept as prose and split at import: a 49-element list literal is far less
# readable, and this runs once.
_STOPWORD_TEXT = (
    "the a an of to for and or in on is are was were do does you your we our "
    "what which any if as be been that this these those with by from at it its "
    "have has had not can could will would should there their they them"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def _hash_bucket(word: str, dimensions: int) -> int:
    # blake2b, not hash(): PYTHONHASHSEED randomisation would make vectors
    # differ between runs and turn ordering assertions flaky.
    digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


class FakeEmbedder:
    def __init__(self, dimensions: int = 128) -> None:
        self._dimensions = dimensions
        self.document_calls = 0
        self.query_calls = 0

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimensions, dtype=np.float32)
        words = _WORD.findall(text.lower())
        for word in words:
            vector[_hash_bucket(word, self._dimensions)] += 1.0
        if not words:
            raise ValueError("cannot embed text with no words")
        return vector

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            raise ValueError("embed_documents requires at least one text")
        self.document_calls += 1
        vectors = np.vstack([self._vector(t) for t in texts])
        return EmbeddingResult(vectors=vectors, tokens=sum(len(t.split()) for t in texts))

    async def embed_query(self, text: str) -> EmbeddingResult:
        self.query_calls += 1
        return EmbeddingResult(vectors=self._vector(text).reshape(1, -1), tokens=len(text.split()))


class FakeAnswerGenerator:
    """Answers from lexical overlap with the retrieved chunks.

    Deterministic and offline, but not a constant stub: it genuinely returns
    not_found when the retrieved context does not contain the question's terms,
    so grounding behaviour is exercised rather than assumed.

    ``concurrency_peak`` records the highest number of simultaneous calls,
    which is how the service's semaphore is verified.
    """

    def __init__(
        self,
        *,
        delay: float = 0.0,
        fail_questions: frozenset[str] = frozenset(),
        slow_questions: frozenset[str] = frozenset(),
        min_overlap: int = 2,
    ) -> None:
        self._delay = delay
        self._fail_questions = fail_questions
        # When set, only these questions are delayed, so a test can time one
        # question out while the rest of the request succeeds.
        self._slow_questions = slow_questions
        self._min_overlap = min_overlap
        self.calls = 0
        self.concurrency_peak = 0
        self._active = 0

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {w for w in _WORD.findall(text.lower()) if len(w) > 3 and w not in _STOPWORDS}

    async def generate(self, question: str, chunks: Sequence[ScoredChunk]) -> GeneratedAnswer:
        self.calls += 1
        self._active += 1
        self.concurrency_peak = max(self.concurrency_peak, self._active)
        try:
            if self._slow_questions:
                if question in self._slow_questions:
                    await asyncio.sleep(max(self._delay, 0.5))
            elif self._delay:
                await asyncio.sleep(self._delay)
            if question in self._fail_questions:
                raise UpstreamError("Simulated model failure.")

            wanted = self._terms(question)
            scored = [
                (len(wanted & self._terms(c.chunk.text)), i) for i, c in enumerate(chunks, start=1)
            ]
            best_overlap, best_index = max(scored, default=(0, 0))

            if not chunks or best_overlap < self._min_overlap:
                return GeneratedAnswer(
                    status=AnswerStatus.NOT_FOUND,
                    answer=NOT_FOUND_TEXT,
                    citations=[],
                    usage=TokenUsage(prompt_tokens=40, completion_tokens=8),
                    latency_ms=0.0,
                )

            return GeneratedAnswer(
                status=AnswerStatus.ANSWERED,
                answer=f"Answer drawn from extract {best_index}.",
                citations=resolve_citations([best_index], chunks),
                usage=TokenUsage(prompt_tokens=120, completion_tokens=24),
                latency_ms=0.0,
            )
        finally:
            self._active -= 1
