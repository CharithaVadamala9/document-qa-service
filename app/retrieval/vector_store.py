"""Vector index and MMR selection.

FAISS is used in-process: it is an in-memory index with no server to run, which
suits a per-request document index. Everything behind the ``VectorStore``
protocol, so swapping in a hosted vector database is a single-file change.

Vectors are L2-normalised and searched with inner product, which makes the
score exact cosine similarity.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import faiss
import numpy as np

from app.core.errors import ConfigurationError, UpstreamError
from app.core.logging import get_logger
from app.core.models import Chunk
from app.retrieval.embedder import Vectors

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    @property
    def size(self) -> int: ...

    async def add(self, chunks: Sequence[Chunk], vectors: Vectors) -> None: ...

    async def search(
        self, query: Vectors, *, top_k: int, fetch_k: int, mmr_lambda: float
    ) -> list[ScoredChunk]: ...


def _normalise(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise UpstreamError("The embedding service returned a zero vector.")
    return (vectors / norms).astype(np.float32)


def maximal_marginal_relevance(
    query: Vectors, candidates: Vectors, *, k: int, lambda_mult: float
) -> list[int]:
    """Select k candidates trading relevance against redundancy.

    Plain top-k over overlapping chunks routinely returns several near copies
    of one passage, wasting the context budget. MMR penalises a candidate by
    its similarity to what is already selected.
    """
    if candidates.shape[0] == 0:
        return []

    relevance = candidates @ query
    selected: list[int] = [int(np.argmax(relevance))]
    remaining = [i for i in range(candidates.shape[0]) if i != selected[0]]

    while remaining and len(selected) < k:
        redundancy = (candidates[remaining] @ candidates[selected].T).max(axis=1)
        scores = lambda_mult * relevance[remaining] - (1.0 - lambda_mult) * redundancy
        best = remaining.pop(int(np.argmax(scores)))
        selected.append(best)

    return selected


class FaissVectorStore:
    """In-memory index for one document.

    Built once then read concurrently. Reads of a flat FAISS index are safe in
    parallel; writes are serialised by a lock, and the index is sealed after
    the first add so a concurrent read can never observe a partial build.
    """

    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self._index = faiss.IndexFlatIP(dimensions)
        self._vectors: Vectors = np.empty((0, dimensions), dtype=np.float32)
        self._chunks: list[Chunk] = []
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._chunks)

    async def add(self, chunks: Sequence[Chunk], vectors: Vectors) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            raise ValueError("add requires at least one chunk")
        if vectors.shape[1] != self._dimensions:
            raise ConfigurationError(
                f"Expected {self._dimensions}-dimensional vectors but received {vectors.shape[1]}."
            )

        normalised = _normalise(vectors)
        async with self._lock:
            # FAISS releases the GIL but the copy does not; keep both off the
            # event loop so concurrent requests are not stalled by an ingest.
            await asyncio.to_thread(self._index.add, normalised)
            self._vectors = np.vstack([self._vectors, normalised])
            self._chunks.extend(chunks)

    async def search(
        self, query: Vectors, *, top_k: int, fetch_k: int, mmr_lambda: float
    ) -> list[ScoredChunk]:
        if self.size == 0:
            raise UpstreamError("The document index is empty.")
        if top_k <= 0 or fetch_k <= 0:
            raise ValueError("top_k and fetch_k must be positive")

        vector = _normalise(query.reshape(1, -1))
        return await asyncio.to_thread(self._search_sync, vector, top_k, fetch_k, mmr_lambda)

    def _search_sync(
        self, vector: Vectors, top_k: int, fetch_k: int, mmr_lambda: float
    ) -> list[ScoredChunk]:
        scores, indices = self._index.search(vector, min(fetch_k, self.size))
        # FAISS pads with -1 when fewer neighbours exist than requested.
        candidates = [
            (int(i), float(s)) for i, s in zip(indices[0], scores[0], strict=True) if i >= 0
        ]
        if not candidates:
            return []

        candidate_ids = [i for i, _ in candidates]
        chosen = maximal_marginal_relevance(
            vector[0], self._vectors[candidate_ids], k=top_k, lambda_mult=mmr_lambda
        )
        return [
            ScoredChunk(chunk=self._chunks[candidates[c][0]], score=candidates[c][1])
            for c in chosen
        ]
