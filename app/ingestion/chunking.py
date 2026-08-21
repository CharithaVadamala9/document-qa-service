"""Token-aware chunking over segments from either loader.

Segments are packed to the token budget rather than split from a flat string,
so a chunk never straddles a record boundary or a table row. Only segments that
exceed the budget on their own are split, and always at a structural seam.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.core.errors import MalformedDocument
from app.core.logging import get_logger
from app.core.models import Chunk, SourceType
from app.core.tokens import count_tokens
from app.ingestion.segments import Segment, SegmentKind

logger = get_logger(__name__)

_WS = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _chunk_id(source: str, locator: str, text: str) -> str:
    """Stable across runs so chunk ids can be logged instead of content."""
    normalised = _WS.sub(" ", text).strip().lower()
    digest = hashlib.sha256(f"{source}|{locator}|{normalised}".encode())
    return digest.hexdigest()[:16]


def _locator(segment: Segment) -> str:
    if segment.json_path is not None:
        return segment.json_path
    return f"p{segment.page}" if segment.page is not None else "-"


def _prose_splitter(settings: Settings, budget: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=settings.llm_model,
        chunk_size=budget,
        chunk_overlap=min(settings.chunk_overlap_tokens, budget // 2),
        separators=_SEPARATORS,
    )


def _split_table(segment: Segment, budget: int, settings: Settings) -> list[Segment]:
    """Split by rows, repeating the header so every piece keeps its columns."""
    lines = segment.text.split("\n")
    if len(lines) < 3:  # header, rule, at least one row
        return [segment]
    head, rule, rows = lines[0], lines[1], lines[2:]
    prefix = f"{head}\n{rule}"
    prefix_tokens = count_tokens(prefix, settings.llm_model)

    pieces: list[Segment] = []
    buffer: list[str] = []
    used = prefix_tokens

    def flush() -> None:
        nonlocal buffer, used
        if buffer:
            text = "\n".join([prefix, *buffer])
            pieces.append(replace(segment, text=text))
        buffer, used = [], prefix_tokens

    for row in rows:
        row_tokens = count_tokens(row, settings.llm_model)
        if buffer and used + row_tokens > budget:
            flush()
        buffer.append(row)
        used += row_tokens
    flush()
    return pieces or [segment]


def _split_record(segment: Segment, budget: int, settings: Settings) -> list[Segment]:
    """Split at top-level keys, repeating the title line on every piece."""
    lines = segment.text.split("\n")
    title, body = lines[0], lines[1:]
    title_tokens = count_tokens(title, settings.llm_model)

    pieces: list[Segment] = []
    buffer: list[str] = []
    used = title_tokens

    def flush() -> None:
        nonlocal buffer, used
        if buffer:
            text = "\n".join([title, *buffer])
            pieces.append(replace(segment, text=text))
        buffer, used = [], title_tokens

    for line in body:
        tokens = count_tokens(line, settings.llm_model)
        # Break only before a top-level key, never mid-object.
        if buffer and used + tokens > budget and not line.startswith(" "):
            flush()
        buffer.append(line)
        used += tokens
    flush()
    return pieces or [segment]


def _split_oversized(segment: Segment, budget: int, settings: Settings) -> list[Segment]:
    if count_tokens(segment.text, settings.llm_model) <= budget:
        return [segment]
    if segment.kind is SegmentKind.TABLE:
        return _split_table(segment, budget, settings)
    if segment.kind is SegmentKind.RECORD:
        return _split_record(segment, budget, settings)
    return [
        replace(segment, text=piece)
        for piece in _prose_splitter(settings, budget).split_text(segment.text)
    ]


def _overlap_tail(text: str, overlap_tokens: int, settings: Settings) -> str:
    """Trailing sentences of the previous chunk, for prose continuity."""
    if overlap_tokens <= 0:
        return ""
    sentences = _SENTENCE_END.split(text)
    tail: list[str] = []
    used = 0
    for sentence in reversed(sentences):
        tokens = count_tokens(sentence, settings.llm_model)
        if used + tokens > overlap_tokens:
            break
        tail.insert(0, sentence)
        used += tokens
    return " ".join(tail).strip()


def chunk_segments(
    segments: list[Segment],
    *,
    source: str,
    source_type: SourceType,
    settings: Settings,
) -> tuple[Chunk, ...]:
    if not segments:
        raise MalformedDocument("The document produced no readable content.")

    header = segments[0].header or ""
    header_tokens = count_tokens(header, settings.llm_model) if header else 0
    budget = settings.chunk_size_tokens - header_tokens - settings.chunk_overlap_tokens
    if budget <= settings.min_chunk_tokens:
        raise MalformedDocument(
            "The document's repeated context is too large for the configured "
            "chunk size, leaving no room for content."
        )

    units: list[Segment] = []
    for segment in segments:
        units.extend(_split_oversized(segment, budget, settings))

    chunks: list[Chunk] = []
    buffer: list[Segment] = []
    used = 0
    carry = ""

    def flush() -> None:
        nonlocal buffer, used, carry
        if not buffer:
            return
        first = buffer[0]
        parts = [p for p in (first.header, carry, *(s.text for s in buffer)) if p]
        text = "\n\n".join(parts).strip()
        tokens = count_tokens(text, settings.llm_model)

        # A trailing fragment below the floor is dropped only when other chunks
        # exist; otherwise a short document would index to nothing.
        if tokens >= settings.min_chunk_tokens or not chunks:
            # Every member's locator is kept: a packed chunk that cited only
            # its first record would misattribute the rest.
            pages = tuple(sorted({s.page for s in buffer if s.page is not None}))
            json_paths = tuple(dict.fromkeys(s.json_path for s in buffer if s.json_path))
            chunks.append(
                Chunk(
                    id=_chunk_id(source, _locator(first), text),
                    text=text,
                    index=len(chunks),
                    token_count=tokens,
                    source=source,
                    source_type=source_type,
                    pages=pages,
                    json_paths=json_paths,
                    section=next((s.section for s in buffer if s.section), None),
                )
            )
        # Overlap applies to prose only: repeating a whole record or table row
        # would duplicate a self-contained unit rather than preserve context.
        carry = (
            _overlap_tail(buffer[-1].text, settings.chunk_overlap_tokens, settings)
            if buffer[-1].kind in (SegmentKind.PROSE, SegmentKind.HEADING)
            else ""
        )
        buffer, used = [], 0

    for unit in units:
        tokens = count_tokens(unit.text, settings.llm_model)
        if buffer and used + tokens > budget:
            flush()
        buffer.append(unit)
        used += tokens
    flush()

    if not chunks:
        raise MalformedDocument("The document produced no readable content.")

    logger.info(
        "document.chunked",
        source_type=source_type,
        segments=len(segments),
        units=len(units),
        chunks=len(chunks),
        tokens=sum(c.token_count for c in chunks),
    )
    return tuple(chunks)
