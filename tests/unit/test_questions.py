from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.core.errors import MalformedDocument, PayloadTooLarge, ValidationError
from app.services.questions import parse_questions


@pytest.mark.parametrize(
    "payload",
    [
        b'["First question?", "Second question?"]',
        b'{"questions": ["First question?", "Second question?"]}',
        b'[{"question": "First question?"}, {"question": "Second question?"}]',
        b'{"items": [{"text": "First question?"}, {"text": "Second question?"}]}',
    ],
)
def test_accepts_common_shapes(payload: bytes, settings: Settings) -> None:
    assert parse_questions(payload, settings) == ["First question?", "Second question?"]


def test_preserves_order_and_strips_whitespace(settings: Settings) -> None:
    payload = json.dumps(["  b?  ", "a?", "c?"]).encode()
    assert parse_questions(payload, settings) == ["b?", "a?", "c?"]


def test_bom_is_tolerated(settings: Settings) -> None:
    assert parse_questions(b'\xef\xbb\xbf["Q?"]', settings) == ["Q?"]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"", "empty"),
        (b"   ", "empty"),
        (b'{"questions": ', "not valid JSON"),
        (b"null", "must be a JSON array"),
        (b'{"other": [1]}', "must be a JSON array"),
        (b"[]", "no questions"),
        (b'[""]', "is empty"),
        (b'["   "]', "is empty"),
        (b"[123]", "must be a string"),
        (b"[null]", "must be a string"),
    ],
)
def test_rejects_bad_input(payload: bytes, match: str, settings: Settings) -> None:
    with pytest.raises((ValidationError, MalformedDocument), match=match):
        parse_questions(payload, settings)


def test_enforces_question_count_limit(settings: Settings) -> None:
    payload = json.dumps(["q?"] * (settings.max_questions + 1)).encode()
    with pytest.raises(ValidationError, match="exceeds the limit"):
        parse_questions(payload, settings)


def test_allows_exactly_the_limit(settings: Settings) -> None:
    payload = json.dumps(["q?"] * settings.max_questions).encode()
    assert len(parse_questions(payload, settings)) == settings.max_questions


def test_enforces_question_length_limit(settings: Settings) -> None:
    payload = json.dumps(["x" * (settings.max_question_chars + 1)]).encode()
    with pytest.raises(ValidationError, match="exceeds"):
        parse_questions(payload, settings)


def test_enforces_file_size_limit(settings: Settings) -> None:
    tight = settings.model_copy(update={"max_questions_file_kb": 1})
    payload = json.dumps(["a question that is fairly long?" * 4] * 40).encode()
    with pytest.raises(PayloadTooLarge, match="larger than"):
        parse_questions(payload, tight)


def test_error_identifies_the_offending_question(settings: Settings) -> None:
    with pytest.raises(ValidationError) as info:
        parse_questions(b'["fine?", "", "also fine?"]', settings)
    assert info.value.detail["position"] == 2
