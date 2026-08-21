from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.core.errors import MalformedDocument, ValidationError
from app.ingestion.json_document import extract_json
from app.ingestion.segments import SegmentKind


def test_finds_largest_coherent_collection(security_json: bytes, settings: Settings) -> None:
    # Both `teams` and `controls` are record collections; the larger one is
    # what the document is about.
    _, stats = extract_json(security_json, settings)
    assert stats.root_path == "$.controls"
    assert stats.records == 3


def test_preserves_nesting_rather_than_flattening(security_json: bytes, settings: Settings) -> None:
    segments, _ = extract_json(security_json, settings)
    first = segments[0].text

    # Flattening to owner.team would sever the grouping that tells the model
    # which owner the contact belongs to.
    assert "owner:\n  team: Security Engineering" in first
    assert "  contact: sec@acme.com" in first
    assert "owner.team" not in first


def test_keeps_list_items_together(security_json: bytes, settings: Settings) -> None:
    segments, _ = extract_json(security_json, settings)
    first = segments[0].text
    assert "- procedure: Inspected incident tickets" in first
    assert "  result: No exceptions noted" in first


def test_resolves_cross_collection_references(security_json: bytes, settings: Settings) -> None:
    # owner_id points into `teams`, not into `controls`. Unresolved, the chunk
    # is unreachable by a question naming the team.
    segments, stats = extract_json(security_json, settings)
    assert stats.cross_references_resolved == 3
    assert "owner_id: 42 (Security Engineering)" in segments[0].text


def test_ancestor_context_carried_as_header(security_json: bytes, settings: Settings) -> None:
    segments, _ = extract_json(security_json, settings)
    for segment in segments:
        assert segment.header is not None
        assert "vendor=Acme Corp" in segment.header
        assert "report_year=2024" in segment.header


def test_records_are_addressable(security_json: bytes, settings: Settings) -> None:
    segments, _ = extract_json(security_json, settings)
    assert [s.json_path for s in segments] == [
        "$.controls[0]",
        "$.controls[1]",
        "$.controls[2]",
    ]
    assert all(s.kind is SegmentKind.RECORD for s in segments)


def test_document_without_collection_is_one_record(settings: Settings) -> None:
    payload = json.dumps({"vendor": "Acme", "policy": {"retention_days": 90}}).encode()
    segments, stats = extract_json(payload, settings)
    assert stats.records == 1
    assert "retention_days: 90" in segments[0].text


def test_empty_containers_are_marked(security_json: bytes, settings: Settings) -> None:
    segments, _ = extract_json(security_json, settings)
    assert "tests: (empty)" in segments[2].text


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"", "empty"),
        (b"   ", "empty"),
        (b'{"a": ', "not valid JSON"),
        (b"\x80\x81\x82", "not valid UTF-8"),
    ],
)
def test_rejects_unparseable_input(payload: bytes, match: str, settings: Settings) -> None:
    with pytest.raises(MalformedDocument, match=match):
        extract_json(payload, settings)


def test_rejects_excessive_depth(settings: Settings) -> None:
    nested: object = "leaf"
    for _ in range(80):
        nested = {"child": nested}
    with pytest.raises(ValidationError, match="deeper"):
        extract_json(json.dumps(nested).encode(), settings)


def test_rejects_excessive_node_count(settings: Settings) -> None:
    tight = settings.model_copy(update={"max_json_nodes": 10})
    payload = json.dumps({"items": [{"id": i, "name": f"n{i}"} for i in range(50)]}).encode()
    with pytest.raises(ValidationError, match="nodes"):
        extract_json(payload, tight)


def test_bom_is_tolerated(settings: Settings) -> None:
    # Spreadsheet exporters emit a BOM routinely; json.loads fails on byte zero.
    payload = b"\xef\xbb\xbf" + json.dumps({"vendor": "Acme", "n": 1}).encode()
    segments, _ = extract_json(payload, settings)
    assert "vendor: Acme" in segments[0].text
