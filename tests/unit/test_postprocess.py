from __future__ import annotations

import pytest

from app.llm.postprocess import strip_extract_references, strip_markdown_emphasis


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**Alpha**: first item", "Alpha: first item"),
        ("*Beta* and _Gamma_", "Beta and Gamma"),
        ("__Delta__ applies", "Delta applies"),
        ("`code` stays readable", "code stays readable"),
        ("### Heading\nbody text", "Heading\nbody text"),
        ("A **multi word phrase** here", "A multi word phrase here"),
    ],
)
def test_markdown_emphasis_is_removed(raw: str, expected: str) -> None:
    assert strip_markdown_emphasis(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        # Spaced asterisks are arithmetic, not emphasis.
        "2 * 3 * 4 is arithmetic",
        # Underscores inside a word belong to the identifier.
        "max_retry_count is a setting",
        "values a_1 and b_2 differ",
        # A lone delimiter wraps nothing.
        "The * character is literal",
        "A snake_case name survives",
        # Bullets and enumeration are structure, not emphasis.
        "1. First\n2. Second",
    ],
)
def test_non_emphasis_punctuation_survives(raw: str) -> None:
    assert strip_markdown_emphasis(raw) == raw


def test_emphasis_around_a_figure_leaves_the_figure_checkable() -> None:
    # The numeric guardrail runs on the stripped text, so a figure wrapped in
    # emphasis has to come out as a bare figure or it cannot be verified.
    assert strip_markdown_emphasis("Retention is **35** days.") == "Retention is 35 days."


def test_emphasis_is_removed_before_extract_references() -> None:
    # Ordering matters: stripping the reference first would strand "****".
    cleaned = strip_extract_references(strip_markdown_emphasis("**Yes** (Extract 1)."))
    assert cleaned == "Yes."
