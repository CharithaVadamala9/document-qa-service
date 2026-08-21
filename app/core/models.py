"""Domain types shared by ingestion, retrieval and QA."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SourceType(StrEnum):
    PDF = "pdf"
    JSON = "json"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    NOT_FOUND = "not_found"
    ERROR = "error"


NOT_FOUND_TEXT = "Not found in the provided document."


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit. Locators are type-specific: PDFs cite pages,
    JSON cites a path."""

    id: str
    text: str
    index: int
    token_count: int
    source: str
    source_type: SourceType
    pages: tuple[int, ...] = ()
    json_paths: tuple[str, ...] = ()
    section: str | None = None

    def citation(self, *, snippet_chars: int = 240) -> Citation:
        snippet = " ".join(self.text.split())
        if len(snippet) > snippet_chars:
            snippet = snippet[: snippet_chars - 1].rstrip() + "…"
        return Citation(
            chunk_id=self.id,
            source=self.source,
            pages=list(self.pages),
            json_paths=list(self.json_paths),
            section=self.section,
            snippet=snippet,
        )


@dataclass(frozen=True, slots=True)
class Citation:
    """Built from chunk metadata only — never from model output."""

    chunk_id: str
    source: str
    snippet: str
    pages: list[int] = field(default_factory=list)
    json_paths: list[str] = field(default_factory=list)
    section: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    source: str
    source_type: SourceType
    chunks: tuple[Chunk, ...]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(c.token_count for c in self.chunks)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.embedding_tokens + other.embedding_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.embedding_tokens
