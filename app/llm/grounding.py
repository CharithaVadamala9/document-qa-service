"""Deterministic post-checks on a generated answer.

The prompt asks the model to stay within the extracts; this verifies it did,
without a second model call. It only checks what can be checked mechanically.

Figures are the target because they are where an ungrounded answer does real
damage: a notification SLA of 24 hours instead of 72, or 99.99% instead of
99.9%, reads as authoritative and is wrong. Prose drift is subtler and is left
to an LLM judge (see evals/), which is not deterministic and costs money.
"""

from __future__ import annotations

import re

# A figure is a digit run, optionally with thousands separators, a decimal
# part, or a trailing percent sign.
_FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
# "1." or "2)" at the start of a line is the model's own enumeration, not a
# claim about the document.
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)](?=\s)")
# Typeset documents use non-breaking hyphens where a keyboard types "-".
_DASHES = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"))


def _normalise(text: str) -> str:
    return text.translate(_DASHES).replace(",", "").lower()


def _figures(text: str) -> list[str]:
    return [m.group(0).rstrip("%").rstrip(".") for m in _FIGURE.finditer(text)]


def unsupported_figures(answer: str, *, cited_text: str, question: str) -> list[str]:
    """Figures asserted in the answer that appear in neither the cited extracts
    nor the question itself.

    The question is allowed as a source because a model legitimately echoes
    figures the caller supplied ("within the 72 hours you asked about").
    """
    body = _LIST_MARKER.sub("", _normalise(answer))
    allowed = _normalise(cited_text) + " " + _normalise(question)

    unsupported: list[str] = []
    for figure in _figures(body):
        if figure not in allowed and figure not in unsupported:
            unsupported.append(figure)
    return unsupported
