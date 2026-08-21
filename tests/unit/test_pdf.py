from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import (
    EncryptedDocument,
    MalformedDocument,
    NoExtractableText,
    ValidationError,
)
from app.ingestion.pdf import extract_pdf
from app.ingestion.segments import SegmentKind
from tests.fixtures import factory


def test_extracts_all_pages(soc2_pdf: bytes, settings: Settings) -> None:
    segments, stats = extract_pdf(soc2_pdf, settings)
    assert stats.pages >= 3
    assert stats.pages_with_text == stats.pages
    assert segments


def test_strips_running_header_and_footer(soc2_pdf: bytes, settings: Settings) -> None:
    segments, stats = extract_pdf(soc2_pdf, settings)
    body = "\n".join(s.text for s in segments)

    assert stats.boilerplate_lines >= 2
    assert "Do not distribute" not in body
    assert "SOC 2 Type II Report" not in body
    # Page numbers vary per page but normalise to the same key.
    assert "Page 1 of" not in body


def test_table_rendered_as_markdown_with_introducing_section(
    soc2_pdf: bytes, settings: Settings
) -> None:
    segments, _ = extract_pdf(soc2_pdf, settings)
    tables = [s for s in segments if s.kind is SegmentKind.TABLE]

    assert len(tables) == 1
    table = tables[0]
    assert table.text.startswith("| Provider |")
    assert "| Amazon Web Services |" in table.text
    assert table.header == "| Provider | Primary Region | Backup Region | Workload |"
    # Attribution comes from reading order, not from the page's last heading.
    assert table.section is not None
    assert table.section.startswith("A1.2")


def test_paragraphs_are_reflowed_not_line_per_segment(soc2_pdf: bytes, settings: Settings) -> None:
    segments, _ = extract_pdf(soc2_pdf, settings)
    prose = [s for s in segments if s.kind is SegmentKind.PROSE]

    # A wrapped paragraph must arrive as one segment, not one per rendered line.
    assert any(len(s.text) > 200 for s in prose)
    assert all("\n" not in s.text for s in prose)


def test_section_tracks_position_not_page(soc2_pdf: bytes, settings: Settings) -> None:
    segments, _ = extract_pdf(soc2_pdf, settings)
    first_page = [s for s in segments if s.page == 1]

    # Page 1 opens with CC7.3 and later moves to CC6.7; a per-page label would
    # give every segment the same (wrong) section.
    assert first_page[0].section is not None
    assert first_page[0].section.startswith("CC7.3")
    assert len({s.section for s in first_page}) > 1


def test_headings_detected(soc2_pdf: bytes, settings: Settings) -> None:
    segments, _ = extract_pdf(soc2_pdf, settings)
    headings = [s.text for s in segments if s.kind is SegmentKind.HEADING]
    assert "CC7.3 Incident Notification Commitments" in headings


def test_rejects_html_masquerading_as_pdf(settings: Settings) -> None:
    # MuPDF sniffs content and will parse HTML unless we check the header.
    with pytest.raises(MalformedDocument, match="not a PDF"):
        extract_pdf(b"<!doctype html><html><body>Nope</body></html>" * 40, settings)


def test_rejects_corrupt_pdf(settings: Settings) -> None:
    with pytest.raises(MalformedDocument):
        extract_pdf(b"%PDF-1.7\ngarbage not a real pdf", settings)


def test_rejects_encrypted_pdf(settings: Settings) -> None:
    with pytest.raises(EncryptedDocument, match="password-protected"):
        extract_pdf(factory.build_encrypted_pdf(), settings)


def test_scanned_pdf_reports_no_extractable_text(settings: Settings) -> None:
    # Must be distinguishable from "answer not found", which would blame the
    # question rather than the unreadable document.
    with pytest.raises(NoExtractableText, match="scanned or image-only"):
        extract_pdf(factory.build_scanned_pdf(), settings)


def test_partially_scanned_pdf_is_rejected_by_ratio(settings: Settings) -> None:
    # One readable page among many image-only ones still fails the ratio check.
    with pytest.raises(NoExtractableText):
        extract_pdf(factory.build_mostly_scanned_pdf(), settings)


def test_enforces_page_limit(soc2_pdf: bytes, settings: Settings) -> None:
    tight = settings.model_copy(update={"max_pdf_pages": 1})
    with pytest.raises(ValidationError, match="exceeds"):
        extract_pdf(soc2_pdf, tight)


def test_rejects_pdf_with_no_pages(settings: Settings) -> None:
    # Hand-written: PyMuPDF refuses to save a zero-page document, but such
    # files exist in the wild and must not reach the chunker.
    empty = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )
    with pytest.raises(MalformedDocument):
        extract_pdf(empty, settings)
