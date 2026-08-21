"""Intermediate representation between parsing and chunking.

Both loaders emit Segments so one chunker serves both formats. ``header`` is
repeated on every piece when a segment must be split: the markdown header row
for a table, the ancestor context for a JSON record.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SegmentKind(StrEnum):
    PROSE = "prose"
    HEADING = "heading"
    TABLE = "table"
    RECORD = "record"


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    kind: SegmentKind = SegmentKind.PROSE
    page: int | None = None
    json_path: str | None = None
    section: str | None = None
    header: str | None = None
