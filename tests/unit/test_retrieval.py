from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.core.config import Settings
from app.core.errors import ConfigurationError, MalformedDocument, UpstreamError
from app.core.models import Chunk, SourceType
from app.retrieval.retriever import DocumentIndex
from app.retrieval.vector_store import FaissVectorStore, maximal_marginal_relevance
from tests.fixtures.fakes import FakeEmbedder


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        id=f"c{index}",
        text=text,
        index=index,
        token_count=len(text.split()),
        source="a.pdf",
        source_type=SourceType.PDF,
        pages=(index + 1,),
    )


def _unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TestMMR:
    def test_picks_most_relevant_first(self) -> None:
        query = _unit(1, 0)
        candidates = np.vstack([_unit(0, 1), _unit(1, 0.05)])
        assert maximal_marginal_relevance(query, candidates, k=1, lambda_mult=0.5) == [1]

    # Three dimensions are required to observe diversity at all: in 2D with a
    # unit query, redundancy against the first pick equals relevance for every
    # candidate, so every MMR score collapses to zero.
    QUERY = _unit(1, 0, 0)
    CANDIDATES = np.vstack(
        [
            _unit(0.90, 0.44, 0.00),  # strong match
            _unit(0.90, 0.43, 0.05),  # near duplicate of the above
            _unit(0.80, 0.00, 0.60),  # slightly weaker, genuinely different
        ]
    )

    def test_prefers_diversity_over_a_near_duplicate(self) -> None:
        chosen = maximal_marginal_relevance(self.QUERY, self.CANDIDATES, k=2, lambda_mult=0.5)
        assert chosen[0] in (0, 1)
        assert chosen[1] == 2, "a near duplicate should not take the second slot"

    def test_lambda_one_is_plain_relevance_ranking(self) -> None:
        # With no diversity term the two near-duplicates win on relevance.
        chosen = maximal_marginal_relevance(self.QUERY, self.CANDIDATES, k=2, lambda_mult=1.0)
        assert set(chosen) == {0, 1}

    def test_empty_candidates(self) -> None:
        assert (
            maximal_marginal_relevance(
                _unit(1, 0), np.empty((0, 2), np.float32), k=3, lambda_mult=0.5
            )
            == []
        )


class TestFaissVectorStore:
    async def test_scores_are_cosine_similarity(self) -> None:
        store = FaissVectorStore(2)
        # Unnormalised input: the store must normalise before scoring.
        await store.add(
            [_chunk(0, "a"), _chunk(1, "b")], np.array([[3.0, 0.0], [0.0, 7.0]], dtype=np.float32)
        )
        results = await store.search(
            np.array([2.0, 0.0], dtype=np.float32), top_k=2, fetch_k=2, mmr_lambda=1.0
        )
        assert results[0].chunk.id == "c0"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_search_on_empty_index_raises(self) -> None:
        store = FaissVectorStore(4)
        with pytest.raises(UpstreamError, match="empty"):
            await store.search(np.ones(4, dtype=np.float32), top_k=1, fetch_k=1, mmr_lambda=0.5)

    async def test_rejects_dimension_mismatch(self) -> None:
        store = FaissVectorStore(4)
        with pytest.raises(ConfigurationError, match="dimensional"):
            await store.add([_chunk(0, "a")], np.ones((1, 8), dtype=np.float32))

    async def test_rejects_zero_vector(self) -> None:
        store = FaissVectorStore(3)
        with pytest.raises(UpstreamError, match="zero vector"):
            await store.add([_chunk(0, "a")], np.zeros((1, 3), dtype=np.float32))

    async def test_rejects_length_mismatch(self) -> None:
        store = FaissVectorStore(3)
        with pytest.raises(ValueError, match="same length"):
            await store.add([_chunk(0, "a")], np.ones((2, 3), dtype=np.float32))

    async def test_top_k_larger_than_index_returns_everything(self) -> None:
        store = FaissVectorStore(2)
        await store.add(
            [_chunk(0, "a"), _chunk(1, "b")], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        )
        results = await store.search(_unit(1, 0), top_k=50, fetch_k=50, mmr_lambda=0.5)
        assert len(results) == 2


class TestDocumentIndex:
    async def test_document_embedded_once_regardless_of_question_count(
        self, pdf_chunks, settings: Settings
    ) -> None:
        embedder = FakeEmbedder()
        index = await DocumentIndex.build(pdf_chunks, embedder=embedder, settings=settings)

        await asyncio.gather(*(index.retrieve(f"question {i}?") for i in range(10)))

        assert embedder.document_calls == 1, "the document must not be re-embedded"
        assert embedder.query_calls == 10
        assert index.embedding_tokens > 0

    async def test_ranks_the_matching_topic_first(self, settings: Settings) -> None:
        # Controlled chunks with disjoint vocabulary. The PDF fixture is too
        # small for this assertion: its chunks each span several sections, so
        # more than one legitimately matches a given question.
        topics = [
            "Backups are replicated nightly to a secondary region and retained "
            "for thirty five days with quarterly restore drills.",
            "Application Performance Monitoring captures distributed traces, "
            "error rates and latency percentiles for production services.",
            "Background screening is completed before personnel are granted "
            "access to customer data, subject to local employment law.",
        ]
        chunks = [_chunk(i, text) for i, text in enumerate(topics)]
        index = await DocumentIndex.build(chunks, embedder=FakeEmbedder(), settings=settings)

        retrieved = await index.retrieve("What latency and error rate monitoring is performed?")
        assert retrieved.chunks[0].chunk.id == "c1"

        retrieved = await index.retrieve("How often are backups replicated and retained?")
        assert retrieved.chunks[0].chunk.id == "c0"

    async def test_chunk_holding_the_answer_is_retrieved(
        self, pdf_chunks, settings: Settings
    ) -> None:
        index = await DocumentIndex.build(pdf_chunks, embedder=FakeEmbedder(), settings=settings)
        retrieved = await index.retrieve("Which cloud providers and hosting regions are used?")
        texts = " ".join(c.chunk.text for c in retrieved.chunks)
        assert "Amazon Web Services" in texts

    async def test_returns_at_most_top_k(self, pdf_chunks, settings: Settings) -> None:
        index = await DocumentIndex.build(pdf_chunks, embedder=FakeEmbedder(), settings=settings)
        retrieved = await index.retrieve("monitoring")
        assert 0 < len(retrieved.chunks) <= settings.retrieval_top_k

    async def test_build_requires_chunks(self, settings: Settings) -> None:
        with pytest.raises(MalformedDocument):
            await DocumentIndex.build([], embedder=FakeEmbedder(), settings=settings)
