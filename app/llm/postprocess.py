"""Tidying applied to answer prose before it reaches the caller.

Extract numbers are an internal addressing scheme: they exist so the model can
point at a chunk and so we can resolve that pointer to a citation. They mean
nothing to a reader, who sees source names and page numbers instead. The model
nonetheless cites them inline ("...secured areas of the facility (Extract 1,
CC6.4.1)"), so they are removed here.

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


def strip_extract_references(answer: str) -> str:
    """Remove internal extract references from user-facing prose."""
    cleaned = _PARENTHESISED.sub("", answer)
    cleaned = _INLINE.sub("", cleaned)
    cleaned = _EMPTY_PARENS.sub("", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)

    # Collapse runs of spaces without touching the line structure of lists.
    lines = [_REPEATED_SPACE.sub(" ", line).rstrip() for line in cleaned.splitlines()]
    return "\n".join(lines).strip()
