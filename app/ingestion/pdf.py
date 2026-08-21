"""Layout-aware PDF extraction.

Strips running headers/footers (which otherwise produce one near-duplicate
chunk per page), renders tables as markdown instead of column soup, and tags
each segment with its section heading.

Synchronous by design: PyMuPDF is a blocking C extension, so callers offload
it to a thread rather than have this module pretend to be async.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field

import pymupdf

from app.core.config import Settings
from app.core.errors import EncryptedDocument, MalformedDocument, NoExtractableText, ValidationError
from app.core.logging import get_logger
from app.ingestion.segments import Segment, SegmentKind

logger = get_logger(__name__)

# PyMuPDF print()s a package recommendation to stdout, which would corrupt the
# JSON log stream.
pymupdf.no_recommend_layout()

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_MAX_HEADING_CHARS = 120
_HEADING_SIZE_RATIO = 1.15
_PARAGRAPH_GAP_RATIO = 0.8
# Headings are set with more leading than body text, so a wrapped heading is
# merged across a wider gap than a wrapped paragraph would be.
_HEADING_MERGE_GAP_RATIO = 1.8


@dataclass(frozen=True, slots=True)
class PdfStats:
    """Extraction provenance, surfaced in logs and the ingest response."""

    pages: int
    pages_with_text: int
    tables: int
    boilerplate_lines: int
    characters: int


@dataclass(slots=True)
class _Line:
    """Layout decisions are made per line, not per block: PyMuPDF often groups
    a heading with the preceding paragraph, so block-level font tests
    misclassify the whole run."""

    text: str
    size: float
    bold: bool
    y0: float
    y1: float

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass(frozen=True, slots=True)
class _Table:
    """Position is kept so tables interleave with prose in reading order;
    appending them per page would attribute each to the page's last heading."""

    markdown: str
    y0: float


@dataclass(slots=True)
class _RawPage:
    number: int
    height: float
    lines: list[_Line] = field(default_factory=list)
    tables: list[_Table] = field(default_factory=list)


def _normalise(text: str) -> str:
    """Collapse a line to a comparison key for boilerplate detection."""
    return _DIGITS.sub("#", _WS.sub(" ", text).strip().lower())


def _render_table(rows: list[list[str | None]]) -> str:
    """Markdown keeps the column-to-value association, which linearised table
    text destroys. Returns "" when the table holds no cells."""
    cleaned = [[_WS.sub(" ", (c or "").strip()) for c in row] for row in rows if any(row)]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]

    header, *body = cleaned
    if not any(header):  # headerless: synthesise positional columns
        header = [f"col{i + 1}" for i in range(width)]
        body = cleaned

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _extract_tables(page: pymupdf.Page, number: int) -> tuple[list[_Table], list[pymupdf.Rect]]:
    """Detect tables and render them. Failures raise: a silently dropped table
    is a silently wrong answer, since tables carry the densest facts."""
    try:
        found = page.find_tables()
    except Exception as exc:
        raise MalformedDocument(
            f"Table detection failed on page {number}; the PDF structure could not be read.",
            page=number,
        ) from exc

    rendered: list[_Table] = []
    boxes: list[pymupdf.Rect] = []
    for table in found.tables:
        try:
            markdown = _render_table(table.extract())
        except Exception as exc:
            raise MalformedDocument(
                f"A table on page {number} could not be read.", page=number
            ) from exc
        if not markdown:
            continue
        rect = pymupdf.Rect(table.bbox)
        rendered.append(_Table(markdown=markdown, y0=rect.y0))
        boxes.append(rect)
    return rendered, boxes


def _covered_by_table(bbox: pymupdf.Rect, table_boxes: list[pymupdf.Rect]) -> bool:
    """True if this geometry sits mostly inside a table already captured."""
    area = bbox.get_area()
    if area <= 0:
        return False
    return any(bbox.intersects(tb) and (bbox & tb).get_area() > 0.5 * area for tb in table_boxes)


def _extract_raw_page(page: pymupdf.Page, number: int) -> _RawPage:
    tables, table_boxes = _extract_tables(page, number)
    height = page.rect.height
    if height <= 0:
        raise MalformedDocument(f"Page {number} has no usable page geometry.", page=number)
    raw = _RawPage(number=number, height=height, tables=tables)

    # sort=True yields reading order rather than content-stream order.
    data = page.get_text("dict", sort=True)
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 0 = text
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            bbox = pymupdf.Rect(line["bbox"])
            if _covered_by_table(bbox, table_boxes):  # already captured as a table
                continue
            raw.lines.append(
                _Line(
                    text="".join(s["text"] for s in spans),
                    size=max(s.get("size", 0.0) for s in spans),
                    # Bit 4 of PyMuPDF's span flags is bold. A line counts as
                    # bold only if every span is: body text with an inline bold
                    # phrase must not read as a heading.
                    bold=all(s.get("flags", 0) & 2**4 for s in spans),
                    y0=bbox.y0,
                    y1=bbox.y1,
                )
            )
    return raw


def _detect_boilerplate(pages: list[_RawPage], settings: Settings) -> set[str]:
    """Find running headers/footers by position plus cross-page frequency.

    Position alone would strip legitimate first-line content; frequency alone
    keeps false positives low."""
    if len(pages) < 2:  # nothing to compare against
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        band = settings.header_footer_band * page.height
        seen = {
            key
            for line in page.lines
            if (line.y1 <= band or line.y0 >= page.height - band) and (key := _normalise(line.text))
        }
        counts.update(seen)

    threshold = settings.boilerplate_page_ratio * len(pages)
    boilerplate = {key for key, n in counts.items() if n >= threshold}
    if boilerplate:
        logger.debug("pdf.boilerplate_detected", count=len(boilerplate), pages=len(pages))
    return boilerplate


def _body_font_size(pages: list[_RawPage]) -> float:
    """Median, not mean: headings are a minority by line count, so the median
    lands on body text instead of being dragged up by title lines."""
    sizes = [line.size for p in pages for line in p.lines if line.size > 0]
    if not sizes:
        raise NoExtractableText(
            "No text could be extracted from this PDF; it appears to be scanned "
            "or image-only. OCR is not supported — please supply a text-based PDF.",
            pages=len(pages),
            pages_with_text=0,
        )
    return statistics.median(sizes)


def _is_heading(line: _Line, heading_floor: float) -> bool:
    """Larger than body text, or set entirely in bold.

    Size alone is not enough: professional reports routinely set subsection
    headings in bold at the body size, and those are exactly the labels worth
    citing. Length still bounds it, so a long bold sentence is not a heading.
    """
    if len(line.text.strip()) > _MAX_HEADING_CHARS:
        return False
    return line.size >= heading_floor or line.bold


def _join(previous: str, nxt: str) -> str:
    """Rejoin wrapped lines, undoing hyphenation so "confiden- tiality" stays
    matchable by retrieval."""
    if previous.endswith("-") and nxt[:1].islower():
        return previous[:-1] + nxt
    return f"{previous} {nxt}"


@dataclass(frozen=True, slots=True)
class _Paragraph:
    text: str
    is_heading: bool
    y0: float


def _reflow(lines: list[_Line], heading_floor: float) -> list[_Paragraph]:
    """Group lines into paragraphs, breaking on a heading, a font-size change,
    or a vertical gap wider than normal line spacing. Without this every
    wrapped line becomes its own paragraph and creates false split points."""
    paragraphs: list[_Paragraph] = []
    buffer: str = ""
    buffer_heading = False
    buffer_y0 = 0.0
    previous: _Line | None = None

    def flush() -> None:
        nonlocal buffer, buffer_heading
        if buffer.strip():
            paragraphs.append(_Paragraph(_WS.sub(" ", buffer).strip(), buffer_heading, buffer_y0))
        buffer, buffer_heading = "", False

    for line in lines:
        heading = _is_heading(line, heading_floor)
        gap = (line.y0 - previous.y1) if previous else 0.0
        size_changed = (
            previous is not None and abs(line.size - previous.size) > 0.15 * previous.size
        )
        # A heading that wraps onto a second line is one heading. Without
        # this, "Section III - Description of the / System" becomes two, and
        # the trailing fragment wins as the section label for everything after.
        continues_heading = (
            buffer_heading
            and heading
            and not size_changed
            and gap <= _HEADING_MERGE_GAP_RATIO * line.height
        )
        starts_paragraph = not continues_heading and (
            previous is None
            or heading
            or buffer_heading
            or size_changed
            or gap > _PARAGRAPH_GAP_RATIO * line.height
        )

        if starts_paragraph:
            flush()
            buffer, buffer_heading, buffer_y0 = line.text.strip(), heading, line.y0
        else:
            buffer = _join(buffer, line.text.strip())
        previous = line

    flush()
    return paragraphs


def _assemble(
    pages: list[_RawPage], boilerplate: set[str], body_font: float
) -> tuple[list[Segment], dict[int, int]]:
    """Emit page-attributed segments plus a per-page character tally.

    ``section`` is a running value stamped per segment, not per page: a page
    usually holds the end of one section and the start of the next, and it is
    the chunk's section that ends up in a citation.
    """
    segments: list[Segment] = []
    per_page_chars: dict[int, int] = {}
    current_section: str | None = None
    heading_floor = body_font * _HEADING_SIZE_RATIO if body_font else float("inf")

    for page in pages:
        kept = [ln for ln in page.lines if _normalise(ln.text) not in boilerplate]
        chars = 0

        # Reading-order merge, so a table inherits the heading that introduces it.
        ordered: list[tuple[float, _Paragraph | _Table]] = [
            *((p.y0, p) for p in _reflow(kept, heading_floor)),
            *((t.y0, t) for t in page.tables),
        ]
        ordered.sort(key=lambda item: item[0])

        for _, item in ordered:
            if isinstance(item, _Table):
                chars += len(item.markdown)
                # Header row kept separately so a split table keeps its columns.
                head, _, rest = item.markdown.partition("\n")
                segments.append(
                    Segment(
                        text=item.markdown,
                        kind=SegmentKind.TABLE,
                        page=page.number,
                        section=current_section,
                        header=head if rest else None,
                    )
                )
                continue

            if item.is_heading:
                current_section = item.text
            chars += len(item.text)
            segments.append(
                Segment(
                    text=item.text,
                    kind=SegmentKind.HEADING if item.is_heading else SegmentKind.PROSE,
                    page=page.number,
                    section=current_section,
                )
            )

        per_page_chars[page.number] = chars

    return segments, per_page_chars


def extract_pdf(data: bytes, settings: Settings) -> tuple[list[Segment], PdfStats]:
    """Extract cleaned, page-attributed segments from PDF bytes."""
    # MuPDF sniffs content and will parse HTML or EPUB when asked for a PDF,
    # so a redirect to an error page would ingest as a valid document.
    if b"%PDF-" not in data[:1024]:
        raise MalformedDocument(
            "The file is not a PDF: the %PDF- header is missing. If this came "
            "from a URL, check that it did not redirect to an HTML page."
        )

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise MalformedDocument(
            "The file could not be opened as a PDF. It may be corrupt or not a PDF."
        ) from exc

    with doc:
        if doc.needs_pass:
            raise EncryptedDocument(
                "The PDF is password-protected. Please supply a decrypted copy."
            )
        if doc.page_count > settings.max_pdf_pages:
            raise ValidationError(
                f"The PDF has {doc.page_count} pages, which exceeds the "
                f"{settings.max_pdf_pages}-page limit. Please split it and retry.",
                pages=doc.page_count,
                limit=settings.max_pdf_pages,
            )
        if doc.page_count == 0:
            raise MalformedDocument("The PDF contains no pages.")

        raw_pages = [_extract_raw_page(doc[i], i + 1) for i in range(doc.page_count)]

    boilerplate = _detect_boilerplate(raw_pages, settings)
    segments, per_page_chars = _assemble(raw_pages, boilerplate, _body_font_size(raw_pages))

    populated = sum(1 for n in per_page_chars.values() if n >= settings.min_chars_per_page)
    if populated / len(raw_pages) < settings.min_extractable_page_ratio:
        raise NoExtractableText(
            "Almost no text could be extracted from this PDF; it appears to be "
            "scanned or image-only. OCR is not supported — please supply a "
            "text-based PDF.",
            pages=len(raw_pages),
            pages_with_text=populated,
        )

    stats = PdfStats(
        pages=len(raw_pages),
        pages_with_text=populated,
        tables=sum(len(p.tables) for p in raw_pages),
        boilerplate_lines=len(boilerplate),
        characters=sum(per_page_chars.values()),
    )
    logger.info("pdf.extracted", **asdict(stats), segments=len(segments))
    return segments, stats
