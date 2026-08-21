"""Builds a document index and retrieves context for a question."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import MalformedDocument
from app.core.logging import Timer, get_logger
from app.core.models import Chunk
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import FaissVectorStore, ScoredChunk, VectorStore

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Retrieved:
    chunks: list[ScoredChunk]
    query_tokens: int
    latency_ms: float


def _embedding_text(chunk: Chunk) -> str:
    """Prefix the section heading unless the chunk already opens with it.

    A chunk of body prose does not repeat its own heading, so without this a
    question phrased in the heading's vocabulary has nothing to match against.
    """
    if chunk.section and not chunk.text.startswith(chunk.section):
        return f"{chunk.section}\n\n{chunk.text}"
    return chunk.text


class DocumentIndex:
    """One document's vectors plus the embedder needed to query them."""

    def __init__(
        self, store: VectorStore, embedder: Embedder, settings: Settings, embedding_tokens: int
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._embedding_tokens = embedding_tokens

    @property
    def embedding_tokens(self) -> int:
        """Tokens spent building the index, counted once per document."""
        return self._embedding_tokens

    @property
    def size(self) -> int:
        return self._store.size

    @classmethod
    async def build(
        cls, chunks: Sequence[Chunk], *, embedder: Embedder, settings: Settings
    ) -> DocumentIndex:
        if not chunks:
            raise MalformedDocument("The document produced no indexable content.")

        timer = Timer()
        result = await embedder.embed_documents([_embedding_text(c) for c in chunks])
        store = FaissVectorStore(embedder.dimensions)
        await store.add(chunks, result.vectors)

        logger.info(
            "index.built",
            chunks=len(chunks),
            embedding_tokens=result.tokens,
            latency_ms=timer.ms,
        )
        return cls(store, embedder, settings, result.tokens)

    async def retrieve(self, question: str) -> Retrieved:
        timer = Timer()
        embedded = await self._embedder.embed_query(question)
        chunks = await self._store.search(
            embedded.vectors[0],
            top_k=self._settings.retrieval_top_k,
            fetch_k=self._settings.mmr_fetch_k,
            mmr_lambda=self._settings.mmr_lambda,
        )
        # Chunk ids, never chunk text: logs must not carry document content.
        logger.debug(
            "retrieval.completed",
            retrieved=len(chunks),
            chunk_ids=[c.chunk.id for c in chunks],
            latency_ms=timer.ms,
        )
        return Retrieved(chunks=chunks, query_tokens=embedded.tokens, latency_ms=timer.ms)
