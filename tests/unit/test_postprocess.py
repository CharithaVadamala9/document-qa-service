from __future__ import annotations

import pytest

from app.llm.postprocess import strip_extract_references


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            "Visitor badges do not permit access (Extract 1, CC6.4.4).",
            "Visitor badges do not permit access.",
        ),
        (
            "Entry is restricted (Extract 1). CCTV is installed (Extract 2).",
            "Entry is restricted. CCTV is installed.",
        ),
        (
            "Access is reviewed quarterly [Extract 2] and revoked in a day.",
            "Access is reviewed quarterly and revoked in a day.",
        ),
        ("Backups are daily, as stated in Extract 4.", "Backups are daily."),
        ("Encryption is enforced (see Extract 3).", "Encryption is enforced."),
        ("Both apply (Extracts 1 and 3).", "Both apply."),
    ],
)
def test_extract_references_are_removed(answer: str, expected: str) -> None:
    assert strip_extract_references(answer) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "Uptime is 99.9% and retention is 35 days.",
        "Notification occurs within seventy-two (72) hours.",
        "The extract of the policy is retained for 90 days.",
        "Controls CC6.4.1 through CC6.4.4 apply.",
    ],
)
def test_legitimate_prose_is_untouched(answer: str) -> None:
    # Figures and control ids must survive: they are the answer's substance,
    # and the numeric grounding check runs against them afterwards.
    assert strip_extract_references(answer) == answer


def test_list_structure_survives() -> None:
    answer = (
        "The controls are:\n1. Entry is restricted (Extract 1).\n2. CCTV is installed (Extract 1)."
    )
    assert strip_extract_references(answer) == (
        "The controls are:\n1. Entry is restricted.\n2. CCTV is installed."
    )


def test_answer_consisting_only_of_a_reference_becomes_empty() -> None:
    # The generator treats this as not_found rather than returning a blank.
    assert strip_extract_references("(Extract 1)") == ""


def test_whitespace_is_tidied() -> None:
    assert strip_extract_references("Yes  (Extract 2)  , it applies.") == "Yes, it applies."
