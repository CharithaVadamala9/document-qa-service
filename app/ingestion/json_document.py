"""Structure-preserving JSON ingestion.

Flattening to ``a.b.c = value`` destroys grouping: nothing then ties ``email``
to the ``owner`` it belonged to, so the model misattributes values. A character
splitter is worse, severing records mid-object. Instead we locate the record
collection, carry ancestor context down, render hierarchy as indented text, and
resolve foreign keys to labels so relational JSON is retrievable at all.

Packing small records into one chunk is left to the chunker, which is the same
mechanism that packs PDF paragraphs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.core.config import Settings
from app.core.errors import MalformedDocument, ValidationError
from app.core.logging import get_logger
from app.core.tokens import truncate_to_tokens
from app.ingestion.segments import Segment, SegmentKind

logger = get_logger(__name__)

Json = Any

_INDENT = "  "
# A single scalar can be arbitrarily large (an embedded base64 blob, a full
# document in one string field). Cap it so one pathological value cannot
# monopolise a chunk.
_MAX_VALUE_TOKENS = 400
# Keys whose values are treated as a record's human-readable label, in order of
# preference, when building the cross-reference map.
_LABEL_KEYS = ("name", "title", "label", "summary", "description", "question")
_ID_KEYS = ("id", "uuid", "key", "code", "identifier")
_REF_SUFFIXES = ("_id", "_ids", "_ref", "_refs", "id", "Id", "Ref")


@dataclass(frozen=True, slots=True)
class JsonStats:
    records: int
    nodes: int
    max_depth: int
    root_path: str
    cross_references_resolved: int


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _decode(data: bytes) -> str:
    # utf-8-sig strips a BOM if present; exporters (notably Excel and Google
    # Sheets) emit one routinely and it makes json.loads fail on byte zero.
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MalformedDocument(
            "The file is not valid UTF-8 text and could not be read as JSON."
        ) from exc


def parse_json(data: bytes) -> Json:
    """RecursionError is caught alongside JSONDecodeError: deep nesting raises
    the former, which is not a ValueError and would otherwise become a 500."""
    text = _decode(data).strip()
    if not text:
        raise MalformedDocument("The JSON document is empty.")

    try:
        return json.loads(text)
    except RecursionError as exc:
        raise ValidationError("The JSON document is nested too deeply to parse safely.") from exc
    except json.JSONDecodeError as exc:
        raise MalformedDocument(
            f"The file is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno}).",
            line=exc.lineno,
            column=exc.colno,
        ) from exc


def _inspect(node: Json, settings: Settings) -> tuple[int, int]:
    """Limits are enforced during the walk, so a hostile document is rejected
    early rather than fully traversed first."""
    nodes = 0
    max_depth = 0
    stack: list[tuple[Json, int]] = [(node, 1)]

    while stack:
        current, depth = stack.pop()
        nodes += 1
        max_depth = max(max_depth, depth)

        if depth > settings.max_json_depth:
            raise ValidationError(
                f"The JSON document nests deeper than the supported limit of "
                f"{settings.max_json_depth} levels.",
                limit=settings.max_json_depth,
            )
        if nodes > settings.max_json_nodes:
            raise ValidationError(
                f"The JSON document contains more than the supported limit of "
                f"{settings.max_json_nodes:,} nodes.",
                limit=settings.max_json_nodes,
            )

        if isinstance(current, dict):
            stack.extend((v, depth + 1) for v in current.values())
        elif isinstance(current, list):
            stack.extend((v, depth + 1) for v in current)

    return nodes, max_depth


# --------------------------------------------------------------------------
# Locating records
# --------------------------------------------------------------------------


Token = str | int
Trail = tuple[Token, ...]


def _format_path(trail: Trail) -> str:
    """Render a key trail as a JSONPath expression, e.g. ``$.controls[17]``."""
    out = "$"
    for token in trail:
        if isinstance(token, int):
            out += f"[{token}]"
        elif token.isidentifier():
            out += f".{token}"
        else:
            out += f"[{token!r}]"
    return out


def _resolve(root: Json, trail: Trail) -> Json:
    node = root
    for token in trail:
        node = node[token]
    return node


def _key_signature_score(items: list[Json]) -> float:
    """Mean key overlap against the first item. Heterogeneous lists score low
    and are not treated as a record collection."""
    dicts = [i for i in items if isinstance(i, dict)]
    if len(dicts) < 2:
        return 0.0
    reference = set(dicts[0])
    if not reference:
        return 0.0
    overlaps = [len(reference & set(d)) / len(reference) for d in dicts[1:]]
    return sum(overlaps) / len(overlaps)


_MIN_SIGNATURE_SCORE = 0.5


def _find_record_root(root: Json) -> tuple[Trail, list[tuple[Trail, Json]]]:
    """Return the record container's trail and the (trail, value) pairs in it.

    A document with no collection is treated as a single record, which is the
    correct reading for configuration-shaped input.
    """
    best: tuple[int, float, Trail, list[tuple[Trail, Json]]] | None = None
    stack: list[tuple[Trail, Json]] = [((), root)]

    while stack:
        trail, node = stack.pop()

        if isinstance(node, list):
            children: list[tuple[Trail, Json]] = [((*trail, i), v) for i, v in enumerate(node)]
            candidate = _key_signature_score(node)
        elif isinstance(node, dict):
            children = [((*trail, k), v) for k, v in node.items()]
            values = list(node.values())
            candidate = (
                _key_signature_score(values)
                if len(values) >= 2 and all(isinstance(v, dict) for v in values)
                else 0.0
            )
        else:
            continue

        # Rank by item count; signature score only breaks ties. The largest
        # coherent collection is what the document is about.
        if candidate >= _MIN_SIGNATURE_SCORE:
            rank = (len(children), candidate)
            if best is None or rank > (best[0], best[1]):
                best = (len(children), candidate, trail, children)
        stack.extend(children)

    if best is None:
        return (), [((), root)]
    return best[2], best[3]


def _ancestor_context(root: Json, container: Trail) -> str:
    """Scalar fields of every ancestor, which qualify all descendants. A chunk
    about control CC6.1 with no indication of whose report it is is useless."""
    parts: list[str] = []
    for depth in range(len(container)):
        node = _resolve(root, container[:depth])
        if isinstance(node, dict):
            parts.extend(
                f"{k}={_scalar(v)}"
                for k, v in node.items()
                if not isinstance(v, dict | list) and v is not None
            )
    return " | ".join(parts)


# --------------------------------------------------------------------------
# Cross-references
# --------------------------------------------------------------------------


def _scalar(value: Json) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return truncate_to_tokens(value.strip(), _MAX_VALUE_TOKENS)
    return str(value)


def _build_reference_map(root: Json) -> dict[str, str]:
    """Built from every object in the tree, not just the chosen collection: a
    foreign key points into a *different* collection, so ``owner_id`` resolves
    against ``teams`` rather than against other controls."""
    mapping: dict[str, str] = {}
    stack: list[Json] = [root]

    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
            continue
        if not isinstance(node, dict):
            continue
        stack.extend(v for v in node.values() if isinstance(v, dict | list))

        lowered = {k.lower(): k for k in node}
        id_key = next((lowered[k] for k in _ID_KEYS if k in lowered), None)
        label_key = next((lowered[k] for k in _LABEL_KEYS if k in lowered), None)
        if id_key is None or label_key is None:
            continue
        identifier, label = node[id_key], node[label_key]
        if isinstance(identifier, str | int) and isinstance(label, str) and label.strip():
            mapping.setdefault(str(identifier), label.strip())

    return mapping


def _looks_like_reference(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in _REF_SUFFIXES) and key.lower() not in _ID_KEYS


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


class _Renderer:
    """Renders a JSON value as indented text, resolving references as it goes."""

    def __init__(self, references: dict[str, str]) -> None:
        self._references = references
        self.resolved = 0

    def _annotate(self, key: str, value: Json) -> str:
        rendered = _scalar(value)
        if not self._references or not _looks_like_reference(key):
            return rendered
        label = self._references.get(str(value))
        if label and label != rendered:
            self.resolved += 1
            return f"{rendered} ({label})"
        return rendered

    def render(self, node: Json, depth: int = 0, key: str | None = None) -> list[str]:
        pad = _INDENT * depth

        if isinstance(node, dict):
            if not node:
                return [f"{pad}(empty)"]
            lines: list[str] = []
            for k, v in node.items():
                if isinstance(v, dict | list):
                    if not v:
                        lines.append(f"{pad}{k}: (empty)")
                    else:
                        lines.append(f"{pad}{k}:")
                        lines.extend(self.render(v, depth + 1, k))
                else:
                    lines.append(f"{pad}{k}: {self._annotate(k, v)}")
            return lines

        if isinstance(node, list):
            lines = []
            for item in node:
                if isinstance(item, dict | list):
                    nested = self.render(item, depth + 1, key)
                    if nested:
                        # Hoist the first line onto the bullet so list items
                        # read as units rather than as an orphaned dash.
                        lines.append(f"{pad}- {nested[0].lstrip()}")
                        lines.extend(nested[1:])
                else:
                    lines.append(f"{pad}- {self._annotate(key or '', item)}")
            return lines

        return [f"{pad}{_scalar(node)}"]


def _record_title(trail: Trail, record: Json) -> str:
    """A short identity line, so a split record still announces what it is."""
    label = " > ".join(str(t) for t in trail) or "document"
    if isinstance(record, dict):
        lowered = {k.lower(): k for k in record}
        for candidate in (*_ID_KEYS, *_LABEL_KEYS):
            if candidate in lowered:
                value = record[lowered[candidate]]
                if isinstance(value, str | int) and str(value).strip():
                    return f"{label} ({_scalar(value)})"
    return label


def extract_json(data: bytes, settings: Settings) -> tuple[list[Segment], JsonStats]:
    """Turn JSON bytes into retrievable, structure-preserving segments."""
    root = parse_json(data)
    nodes, max_depth = _inspect(root, settings)

    container, records = _find_record_root(root)
    header = _ancestor_context(root, container)
    renderer = _Renderer(_build_reference_map(root))

    segments: list[Segment] = []
    for trail, record in records:
        body = "\n".join(renderer.render(record))
        if not body.strip():
            continue
        title = _record_title(trail, record)
        segments.append(
            Segment(
                text=f"{title}\n{body}",
                kind=SegmentKind.RECORD,
                json_path=_format_path(trail),
                section=title,
                header=header or None,  # repeated on every piece if split
            )
        )

    if not segments:
        raise MalformedDocument("The JSON document contains no readable content.")

    stats = JsonStats(
        records=len(segments),
        nodes=nodes,
        max_depth=max_depth,
        root_path=_format_path(container),
        cross_references_resolved=renderer.resolved,
    )
    logger.info("json.extracted", **asdict(stats))
    return segments, stats
