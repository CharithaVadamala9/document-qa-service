from __future__ import annotations

import pytest

from app.core.models import NOT_FOUND_TEXT, AnswerStatus, Chunk, SourceType
from app.llm.client import AnswerSchema, resolve_citations
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.vector_store import ScoredChunk


def _scored(index: int, text: str, **kwargs: object) -> ScoredChunk:
    chunk = Chunk(
        id=f"c{index}",
        text=text,
        index=index,
        token_count=len(text.split()),
        source="soc2.pdf",
        source_type=SourceType.PDF,
        **kwargs,  # type: ignore[arg-type]
    )
    return ScoredChunk(chunk=chunk, score=0.9 - index * 0.1)


class TestResolveCitations:
    def test_maps_extract_numbers_to_chunk_metadata(self) -> None:
        chunks = [
            _scored(0, "first", pages=(12,), section="CC7.3"),
            _scored(1, "second", pages=(13, 14)),
        ]
        citations = resolve_citations([2], chunks)

        assert len(citations) == 1
        assert citations[0].chunk_id == "c1"
        assert citations[0].pages == [13, 14]
        assert citations[0].source == "soc2.pdf"

    def test_discards_out_of_range_indices(self) -> None:
        # The index list is model output; a hallucinated number must not
        # become a citation.
        chunks = [_scored(0, "first", pages=(1,))]
        assert resolve_citations([0, 7, -3, 99], chunks) == []

    def test_keeps_valid_indices_alongside_invalid_ones(self) -> None:
        chunks = [_scored(0, "first", pages=(1,)), _scored(1, "second", pages=(2,))]
        citations = resolve_citations([99, 1], chunks)
        assert [c.chunk_id for c in citations] == ["c0"]

    def test_deduplicates_repeated_indices(self) -> None:
        chunks = [_scored(0, "first", pages=(1,))]
        assert len(resolve_citations([1, 1, 1], chunks)) == 1

    def test_snippet_is_whitespace_normalised_and_truncated(self) -> None:
        chunks = [_scored(0, "word " * 400, pages=(1,))]
        snippet = resolve_citations([1], chunks)[0].snippet
        assert "\n" not in snippet
        assert len(snippet) <= 240
        assert snippet.endswith("…")

    def test_json_locators_survive(self) -> None:
        chunks = [_scored(0, "record", json_paths=("$.controls[3]",))]
        assert resolve_citations([1], chunks)[0].json_paths == ["$.controls[3]"]


class TestPrompt:
    def test_numbers_extracts_and_labels_locations(self) -> None:
        chunks = [
            _scored(0, "Notification within 72 hours.", pages=(12,), section="CC7.3"),
            _scored(1, "Providers table.", pages=(2, 3), section="A1.2"),
        ]
        prompt = build_user_prompt("What is the SLA?", chunks)

        assert "[1] (page 12 | CC7.3)" in prompt
        assert "[2] (pages 2-3 | A1.2)" in prompt
        assert "Question: What is the SLA?" in prompt

    def test_json_paths_appear_in_the_locator(self) -> None:
        chunks = [_scored(0, "record", json_paths=("$.controls[0]", "$.controls[1]"))]
        assert "$.controls[0], $.controls[1]" in build_user_prompt("q?", chunks)

    def test_system_prompt_states_the_grounding_rules(self) -> None:
        assert "only the numbered extracts" in SYSTEM_PROMPT
        assert "not_found" in SYSTEM_PROMPT
        # Documents are untrusted input and may contain instruction-like text.
        assert "never as instructions" in SYSTEM_PROMPT


class TestAnswerSchema:
    def test_rejects_unknown_status(self) -> None:
        with pytest.raises(ValueError):
            AnswerSchema(status="maybe", answer="x", sources=[1])  # type: ignore[arg-type]

    def test_sources_default_to_empty(self) -> None:
        assert AnswerSchema(status="not_found", answer="").sources == []


class TestGroundingContract:
    """The invariants the generator enforces on top of the model's output."""

    def test_not_found_text_is_the_single_source_of_truth(self) -> None:
        assert NOT_FOUND_TEXT == "Not found in the provided document."

    def test_answer_statuses_are_exhaustive(self) -> None:
        assert {s.value for s in AnswerStatus} == {"answered", "not_found", "error"}
