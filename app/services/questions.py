"""Parsing and validation of the uploaded questions file."""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.core.errors import MalformedDocument, PayloadTooLarge, ValidationError

# Accepted shapes, in the order they are tried.
_LIST_KEYS = ("questions", "items", "data")
_TEXT_KEYS = ("question", "text", "prompt", "query")


def _extract_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValidationError(
        "The questions file must be a JSON array of questions, or an object "
        'with a "questions" array.'
    )


def _extract_text(item: Any, position: int) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in _TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str):
                return value
    raise ValidationError(
        f'Question {position} must be a string or an object with a "question" field.',
        position=position,
    )


def parse_questions(data: bytes, settings: Settings) -> list[str]:
    if len(data) > settings.max_questions_file_bytes:
        raise PayloadTooLarge(
            f"The questions file is larger than the {settings.max_questions_file_kb} KB limit.",
            limit_bytes=settings.max_questions_file_bytes,
        )
    if not data.strip():
        raise MalformedDocument("The questions file is empty.")

    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise MalformedDocument("The questions file is not valid UTF-8 text.") from exc
    except RecursionError as exc:
        raise ValidationError("The questions file is nested too deeply.") from exc
    except json.JSONDecodeError as exc:
        raise MalformedDocument(
            f"The questions file is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno}).",
            line=exc.lineno,
            column=exc.colno,
        ) from exc

    raw_items = _extract_list(payload)
    if not raw_items:
        raise ValidationError("The questions file contains no questions.")
    if len(raw_items) > settings.max_questions:
        raise ValidationError(
            f"{len(raw_items)} questions were supplied, which exceeds the limit "
            f"of {settings.max_questions}. Split them across several requests.",
            supplied=len(raw_items),
            limit=settings.max_questions,
        )

    questions: list[str] = []
    for position, item in enumerate(raw_items, start=1):
        text = _extract_text(item, position).strip()
        if not text:
            raise ValidationError(f"Question {position} is empty.", position=position)
        if len(text) > settings.max_question_chars:
            raise ValidationError(
                f"Question {position} is {len(text)} characters, which exceeds "
                f"the limit of {settings.max_question_chars}.",
                position=position,
                limit=settings.max_question_chars,
            )
        questions.append(text)

    return questions
