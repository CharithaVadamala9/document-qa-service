"""Document type detection and the ingest pipeline entry point."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import PurePosixPath

from app.core.config import Settings
from app.core.errors import PayloadTooLarge, UnsupportedFileType
from app.core.logging import get_logger
from app.core.models import ParsedDocument, SourceType
from app.ingestion.chunking import chunk_segments
from app.ingestion.json_document import extract_json
from app.ingestion.pdf import extract_pdf

logger = get_logger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_NAME_CHARS = 120


def sanitise_filename(name: str) -> str:
    """Reduce an uploaded name to a display label.

    The result is never used as a filesystem path -- documents are processed in
    memory -- but it is echoed back in responses and citations, so path
    separators and traversal segments are stripped regardless.
    """
    stem = PurePosixPath(name.replace("\\", "/")).name
    cleaned = _UNSAFE.sub("_", stem).strip("._")
    return cleaned[:_MAX_NAME_CHARS] or "document"


def detect_source_type(data: bytes) -> SourceType:
    """Identify the format from content, never from the extension.

    Filename and Content-Type are supplied by the client and are not evidence.
    """
    if b"%PDF-" in data[:1024]:
        return SourceType.PDF

    head = data[:64].lstrip(b"\xef\xbb\xbf").lstrip()
    if head[:1] in (b"{", b"["):
        return SourceType.JSON

    raise UnsupportedFileType(
        "Unsupported file type. Provide a PDF or a JSON document; the content matched neither."
    )


def load_document(data: bytes, *, filename: str, settings: Settings) -> ParsedDocument:
    if not data:
        raise UnsupportedFileType("The uploaded document is empty.")
    if len(data) > settings.max_file_size_bytes:
        raise PayloadTooLarge(
            f"The document is larger than the {settings.max_file_size_mb} MB limit.",
            limit_bytes=settings.max_file_size_bytes,
            size_bytes=len(data),
        )

    source = sanitise_filename(filename)
    source_type = detect_source_type(data)

    if source_type is SourceType.PDF:
        segments, pdf_stats = extract_pdf(data, settings)
        stats = asdict(pdf_stats)
    else:
        segments, json_stats = extract_json(data, settings)
        stats = asdict(json_stats)

    chunks = chunk_segments(segments, source=source, source_type=source_type, settings=settings)
    logger.info(
        "document.loaded",
        source_type=source_type,
        size_bytes=len(data),
        chunks=len(chunks),
        **stats,
    )
    return ParsedDocument(source=source, source_type=source_type, chunks=chunks, stats=stats)
