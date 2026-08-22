"""Tidying applied to answer prose before it reaches the caller.

Extract numbers are an internal addressing scheme: they exist so the model can
point at a chunk and so we can resolve that pointer to a citation. They mean
nothing to a reader, who sees source names and page numbers instead. The model
nonetheless cites them inline ("...secured areas of the facility (Extract 1,
CC6.4.1)"), so they are removed here.

Markdown emphasis is removed for the same reason: ``answer`` is specified as
prose, and a client that renders it as text shows the delimiters literally.

Only the prose is edited. The ``sources`` list the model returned is untouched,
so citation resolution still works off the original indices.
"""

from __future__ import annotations

import re

# "(Extract 1)", "(Extracts 1 and 3)", "(Extract 1, CC6.4.1)", "[Extract 2]",
# "(see Extract 4)". The trailing group is permissive because the model often
# appends a control id after the number.
_PARENTHESISED = re.compile(
    r"[(\[]\s*(?:see\s+|per\s+|from\s+|in\s+)?extracts?\s*\d+[^)\]]*[)\]]",
    re.IGNORECASE,
)
# "as stated in Extract 2", "according to Extracts 1 and 4".
_INLINE = re.compile(
    r",?\s*(?:as\s+)?(?:stated|noted|shown|described|mentioned|seen|per|according\s+to)?"
    r"\s*(?:in|per)?\s*extracts?\s*\d+(?:\s*(?:,|and)\s*\d+)*",
    re.IGNORECASE,
)

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_REPEATED_SPACE = re.compile(r"[ \t]{2,}")
_EMPTY_PARENS = re.compile(r"[(\[]\s*[)\]]")

# Emphasis only counts when a delimiter hugs the text it wraps, which is what
# separates "**bold**" from "2 * 3". Underscores additionally require a word
# boundary, or identifiers like max_retry_count lose their middle.
_WORD = r"[^\W_]"
_MARKDOWN = (
    re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.DOTALL),
    re.compile(rf"(?<!{_WORD})__(?=\S)(.+?)(?<=\S)__(?!{_WORD})", re.DOTALL),
    re.compile(r"\*(?=\S)([^*\n]+?)(?<=\S)\*"),
    re.compile(rf"(?<!{_WORD})_(?=\S)([^_\n]+?)(?<=\S)_(?!{_WORD})"),
    re.compile(r"`(?=\S)([^`\n]+?)(?<=\S)`"),
)
_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}[ \t]+")


def strip_markdown_emphasis(answer: str) -> str:
    """Remove markdown emphasis, inline code and heading markers from prose."""
    cleaned = _HEADING.sub("", answer)
    for pattern in _MARKDOWN:
        cleaned = pattern.sub(r"\1", cleaned)
    return cleaned


def strip_extract_references(answer: str) -> str:
    """Remove internal extract references from user-facing prose."""
    cleaned = _PARENTHESISED.sub("", answer)
    cleaned = _INLINE.sub("", cleaned)
    cleaned = _EMPTY_PARENS.sub("", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)

    # Collapse runs of spaces without touching the line structure of lists.
    lines = [_REPEATED_SPACE.sub(" ", line).rstrip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip()
