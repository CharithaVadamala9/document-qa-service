"""Deterministic post-checks on a generated answer.

The prompt asks the model to stay within the extracts; this verifies it did,
without a second model call. It only checks what can be checked mechanically.

Figures are the target because they are where an ungrounded answer does real
damage: a notification SLA of 24 hours instead of 72, or 99.99% instead of
99.9%, reads as authoritative and is wrong. Prose drift is subtler and is left
to an LLM judge (see evals/), which is not deterministic and costs money.

Two things are deliberately *not* treated as numeric claims, because flagging
them discards correct answers:

  identifiers  "CC6.6.1", "ISO27001", "TLS1.2" are labels, not quantities. A
               model naming a control it read is not asserting a number.
  notation     "35 days" against a source reading "thirty five days" is the
               same claim written differently.
"""

from __future__ import annotations

import re

# A figure is a digit run, optionally with thousands separators, a decimal
# part, or a trailing percent sign.
_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
# "1." or "2)" at the start of a line is the model's own enumeration, not a
# claim about the document.
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)](?=\s)")
# A letter touching a digit with no separator marks an identifier or an
# ordinal: CC6.6.1, ISO27001, 7th. A hyphen does not ("35-day" stays a claim).
_IDENTIFIER = re.compile(r"[A-Za-z]\d|\d[A-Za-z]")
_DASHES = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"))

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WORD = re.compile(r"[a-z]+")


def _spelled_numbers(text: str) -> set[str]:
    """Digit forms of numbers written as words, e.g. "thirty five" -> "35".

    Compliance prose routinely spells figures out. Without this, a model that
    correctly writes "35 days" is judged against a source saying "thirty five
    days" and its answer is thrown away.
    """
    words = _WORD.findall(text)
    found: set[str] = set()
    index = 0
    while index < len(words):
        word = words[index]
        if word in _TENS:
            value = _TENS[word]
            nxt = words[index + 1] if index + 1 < len(words) else ""
            if nxt in _UNITS and _UNITS[nxt] < 10:
                value += _UNITS[nxt]
                index += 1
            found.add(str(value))
        elif word in _UNITS:
            found.add(str(_UNITS[word]))
        index += 1
    return found


def _normalise(text: str) -> str:
    return text.translate(_DASHES).replace(",", "").lower()


def _figures(text: str) -> list[str]:
    """Digit runs asserted as quantities, excluding identifiers and ordinals."""
    figures: list[str] = []
    for token in text.split():
        if _IDENTIFIER.search(token):
            continue
        figures.extend(m.group(0).rstrip("%").rstrip(".") for m in _FIGURE.finditer(token))
    return figures


def unsupported_figures(answer: str, *, cited_text: str, question: str) -> list[str]:
    """Figures asserted in the answer that appear in neither the cited extracts
    nor the question itself.

    The question is allowed as a source because a model legitimately echoes
    figures the caller supplied ("within the 72 hours you asked about").
    """
    body = _LIST_MARKER.sub("", _normalise(answer))
    allowed = _normalise(cited_text) + " " + _normalise(question)
    spelled = _spelled_numbers(allowed)

    unsupported: list[str] = []
    for figure in _figures(body):
        if figure in allowed or figure in spelled or figure in unsupported:
            continue
        unsupported.append(figure)
    return unsupported
