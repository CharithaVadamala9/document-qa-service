from __future__ import annotations

import pytest

from app.llm.grounding import unsupported_figures

CITED = (
    "The service availability commitment is 99.9% measured monthly. "
    "Notification is delivered within seventy-two (72) hours of confirmation. "
    "Backups are retained for 35 days and reviewed quarterly."
)


@pytest.mark.parametrize(
    "answer",
    [
        "Availability is committed at 99.9%.",
        "Notification occurs within 72 hours.",
        "Backups are kept for 35 days.",
        "Uptime is 99.9% and backups last 35 days.",
        "No figures are stated here.",
    ],
)
def test_supported_figures_pass(answer: str) -> None:
    assert unsupported_figures(answer, cited_text=CITED, question="q?") == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Availability is 99.99%.", ["99.99"]),
        ("Notification within 24 hours.", ["24"]),
        ("Backups are retained for 90 days.", ["90"]),
        ("Uptime 99.95% and retention 60 days.", ["99.95", "60"]),
    ],
)
def test_invented_figures_are_reported(answer: str, expected: list[str]) -> None:
    assert unsupported_figures(answer, cited_text=CITED, question="q?") == expected


def test_list_markers_are_not_claims() -> None:
    # The model's own enumeration must not be mistaken for a figure it asserts.
    answer = "1. Availability is 99.9%.\n2. Backups last 35 days.\n3) Reviewed quarterly."
    assert unsupported_figures(answer, cited_text=CITED, question="q?") == []


def test_figures_echoed_from_the_question_are_allowed() -> None:
    # A caller may name a figure; repeating it is not an invention.
    answer = "The document does not state whether the 48 hour target is met."
    assert (
        unsupported_figures(answer, cited_text=CITED, question="Is the 48 hour target met?") == []
    )


def test_thousands_separators_normalise() -> None:
    cited = "The register holds 12000 entries."
    assert unsupported_figures("There are 12,000 entries.", cited_text=cited, question="q") == []


def test_non_breaking_hyphens_normalise() -> None:
    # Typeset PDFs use U+2011; the answer will use a plain hyphen.
    cited = "Testing is performed semi\u2011annually since 2024."
    assert (
        unsupported_figures("Tested semi-annually since 2024.", cited_text=cited, question="q")
        == []
    )


def test_duplicates_are_reported_once() -> None:
    answer = "It is 90 days, confirmed as 90 days."
    assert unsupported_figures(answer, cited_text=CITED, question="q") == ["90"]


def test_empty_citation_text_flags_every_figure() -> None:
    # An answer citing nothing has nothing supporting its numbers.
    assert unsupported_figures("Retained 35 days.", cited_text="", question="q") == ["35"]
