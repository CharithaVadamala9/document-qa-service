from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import MalformedDocument
from app.core.models import SourceType
from app.core.tokens import count_tokens
from app.ingestion.chunking import chunk_segments
from app.ingestion.json_document import extract_json
from app.ingestion.pdf import extract_pdf
from app.ingestion.segments import Segment, SegmentKind


def _prose(text: str, page: int = 1, section: str | None = None) -> Segment:
    return Segment(text=text, kind=SegmentKind.PROSE, page=page, section=section)


def test_respects_token_budget(soc2_pdf: bytes, settings: Settings) -> None:
    segments, _ = extract_pdf(soc2_pdf, settings)
    chunks = chunk_segments(
        segments, source="soc2.pdf", source_type=SourceType.PDF, settings=settings
    )
    assert chunks
    for chunk in chunks:
        assert chunk.token_count <= settings.chunk_size_tokens


def test_metadata_is_preserved(soc2_pdf: bytes, settings: Settings) -> None:
    segments, _ = extract_pdf(soc2_pdf, settings)
    chunks = chunk_segments(
        segments, source="soc2.pdf", source_type=SourceType.PDF, settings=settings
    )
    for index, chunk in enumerate(chunks):
        assert chunk.index == index
        assert chunk.source == "soc2.pdf"
        assert chunk.source_type is SourceType.PDF
        assert chunk.pages
        assert chunk.token_count == count_tokens(chunk.text, settings.llm_model)


def test_page_spans_recorded_when_a_chunk_crosses_a_boundary(
    soc2_pdf: bytes, settings: Settings
) -> None:
    segments, _ = extract_pdf(soc2_pdf, settings)
    chunks = chunk_segments(
        segments, source="soc2.pdf", source_type=SourceType.PDF, settings=settings
    )
    assert any(len(c.pages) > 1 for c in chunks), "expected a chunk spanning pages"
    for chunk in chunks:
        assert list(chunk.pages) == sorted(chunk.pages)


def test_chunk_ids_are_stable_and_content_derived(settings: Settings) -> None:
    segments = [_prose("Retention is ninety days after termination.")]
    kwargs = {"source": "a.pdf", "source_type": SourceType.PDF, "settings": settings}

    first = chunk_segments(segments, **kwargs)
    again = chunk_segments(segments, **kwargs)
    assert first[0].id == again[0].id

    changed = chunk_segments([_prose("Retention is sixty days.")], **kwargs)
    assert changed[0].id != first[0].id

    other_source = chunk_segments(
        segments, source="b.pdf", source_type=SourceType.PDF, settings=settings
    )
    assert other_source[0].id != first[0].id


def test_prose_chunks_overlap(settings: Settings) -> None:
    tight = settings.model_copy(update={"chunk_size_tokens": 120, "chunk_overlap_tokens": 30})
    sentences = [
        f"Sentence number {i} describes control activity {i} in detail and at length."
        for i in range(40)
    ]
    chunks = chunk_segments(
        [_prose(s) for s in sentences],
        source="a.pdf",
        source_type=SourceType.PDF,
        settings=tight,
    )
    assert len(chunks) > 1
    # The tail of one chunk must reappear at the head of the next.
    tail = chunks[0].text.rsplit(".", 2)[-2].strip()
    assert tail in chunks[1].text


def test_records_do_not_overlap(security_json: bytes, settings: Settings) -> None:
    # Repeating a self-contained record would give it two near-identical
    # vectors and let it win two retrieval slots.
    tight = settings.model_copy(update={"chunk_size_tokens": 200, "chunk_overlap_tokens": 40})
    segments, _ = extract_json(security_json, tight)
    chunks = chunk_segments(segments, source="c.json", source_type=SourceType.JSON, settings=tight)
    assert len(chunks) > 1
    first_titles = [line for line in chunks[0].text.splitlines() if line.startswith("controls >")]
    for later in chunks[1:]:
        for title in first_titles:
            assert title not in later.text


def test_oversized_table_splits_by_row_and_repeats_header(settings: Settings) -> None:
    header = "| Provider | Region |"
    rule = "| --- | --- |"
    rows = [f"| Provider number {i} with a long name | region-{i}-somewhere |" for i in range(60)]
    table = Segment(
        text="\n".join([header, rule, *rows]),
        kind=SegmentKind.TABLE,
        page=2,
        header=header,
    )
    tight = settings.model_copy(update={"chunk_size_tokens": 200, "chunk_overlap_tokens": 20})
    chunks = chunk_segments([table], source="a.pdf", source_type=SourceType.PDF, settings=tight)

    assert len(chunks) > 1
    for chunk in chunks:
        assert header in chunk.text
        assert rule in chunk.text
        # Rows must never be cut in half.
        for line in chunk.text.splitlines():
            if line.startswith("| Provider number"):
                assert line.endswith("|")


def test_packed_records_keep_every_locator(security_json: bytes, settings: Settings) -> None:
    segments, _ = extract_json(security_json, settings)
    chunks = chunk_segments(
        segments, source="c.json", source_type=SourceType.JSON, settings=settings
    )
    # When records are packed together, citing only the first would
    # misattribute the rest.
    packed = [c for c in chunks if len(c.json_paths) > 1]
    assert packed, "expected small records to be packed into one chunk"
    assert packed[0].json_paths == ("$.controls[0]", "$.controls[1]", "$.controls[2]")


def test_tiny_trailing_fragment_dropped_but_short_document_survives(settings: Settings) -> None:
    only = chunk_segments(
        [_prose("Short.")], source="a.pdf", source_type=SourceType.PDF, settings=settings
    )
    assert len(only) == 1, "a short document must still produce one chunk"


def test_empty_input_rejected(settings: Settings) -> None:
    with pytest.raises(MalformedDocument):
        chunk_segments([], source="a.pdf", source_type=SourceType.PDF, settings=settings)


def test_header_larger_than_budget_is_rejected(settings: Settings) -> None:
    huge_header = " ".join(f"field{i}=value{i}" for i in range(400))
    segment = Segment(text="body", kind=SegmentKind.RECORD, header=huge_header)
    with pytest.raises(MalformedDocument, match="repeated context"):
        chunk_segments([segment], source="c.json", source_type=SourceType.JSON, settings=settings)
