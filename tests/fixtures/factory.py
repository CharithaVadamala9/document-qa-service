"""Deterministic document fixtures.

Test documents are generated rather than committed as binaries. That keeps the
repository free of third-party artefacts, makes every fixture diffable, and
lets a test state exactly which structural feature it depends on.

The generated report is engineered to exercise the paths that actually break:
a running header and footer on every page, a ruled table, headings set in a
larger font, prose that spans a page boundary, and -- importantly -- content
that answers only *some* of the sample questions, so that the "not found"
path can be tested against a document that is otherwise perfectly readable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf

PAGE_W, PAGE_H = 595.0, 842.0
MARGIN = 56.0
BODY_TOP, BODY_BOTTOM = 104.0, 770.0
BODY_SIZE, HEADING_SIZE = 10.0, 15.0
LEADING = 1.45

RUNNING_HEADER = "Acme Corp — SOC 2 Type II Report — Confidential"
RUNNING_FOOTER = "Page {n} of {total} — © 2024 Acme Corp — Do not distribute"


@dataclass(frozen=True)
class Section:
    heading: str
    paragraphs: tuple[str, ...]


# Answers to sample questions 1-4 are present; question 5 is deliberately only
# partly covered (APM and EUM, no DEM) and nothing here mentions headcount or
# pricing, giving tests a genuine unanswerable question.
SECTIONS: tuple[Section, ...] = (
    Section(
        "CC7.3 Incident Notification Commitments",
        (
            "Acme Corp maintains formally defined criteria for determining when a "
            "security incident requires customer notification. An incident is "
            "classified as notifiable when it involves confirmed or reasonably "
            "suspected unauthorised access to, disclosure of, or destruction of "
            "customer data, or a material degradation of the security posture of "
            "systems processing customer data.",
            "Where an incident is classified as notifiable, Acme Corp commits to "
            "notifying affected customers without undue delay and in any event "
            "within seventy-two (72) hours of incident confirmation. For incidents "
            "rated Severity 1, the notification service level agreement is reduced "
            "to twenty-four (24) hours. Notification is delivered to the customer's "
            "designated security contact by email, with telephone escalation where "
            "the customer has registered an out-of-hours contact.",
            "Initial notification includes the nature of the incident, the "
            "categories of data involved, the measures taken to contain it, and a "
            "named point of contact. A full post-incident report is provided within "
            "fifteen (15) business days of resolution.",
        ),
    ),
    Section(
        "CC6.7 Third-Party Handling of Personal Information",
        (
            "Personal information is transmitted to, processed by, and retained by a "
            "limited set of third-party subprocessors, each of which is subject to a "
            "written data processing agreement incorporating confidentiality, "
            "security and audit obligations.",
            "Subprocessors comprise the infrastructure providers listed in section "
            "A1.2 below, a transactional email provider that processes recipient "
            "email addresses and delivery metadata, an error-monitoring provider "
            "that may incidentally retain user identifiers present in stack traces, "
            "and a payment processor that independently controls cardholder data. "
            "No personal information is sold, and none is disclosed to third parties "
            "for their own marketing purposes.",
            "Retention by subprocessors is contractually bounded to the period "
            "necessary to deliver the contracted service, and in no case beyond "
            "ninety (90) days following termination of the underlying agreement.",
        ),
    ),
    Section(
        "A1.2 Infrastructure Providers and Hosting Regions",
        (
            "Production services are hosted on the cloud infrastructure providers "
            "listed below. The table identifies, for each provider, the primary "
            "region in which production workloads execute and the corresponding "
            "backup region to which encrypted backups are replicated.",
        ),
    ),
    Section(
        "A1.3 Monitoring of the Service",
        (
            "Acme Corp performs Application Performance Monitoring (APM) across all "
            "production services, capturing distributed traces, error rates and "
            "latency percentiles at the p50, p95 and p99 levels. Alert thresholds "
            "are reviewed quarterly by the Site Reliability Engineering team.",
            "End User Monitoring (EUM) is performed for the customer-facing web "
            "application, collecting page load timings, client-side error events "
            "and Core Web Vitals. EUM data is aggregated and retained for thirteen "
            "(13) months.",
            "Monitoring coverage is assessed annually as part of the internal "
            "control review, and gaps are tracked to closure in the corrective "
            "action register maintained by the Security Engineering team.",
        ),
    ),
    # --- distractors -----------------------------------------------------
    # Plausible, on-topic control sections that answer none of the sample
    # questions. Without these the document is so small that retrieval cannot
    # fail, and a precision regression would pass unnoticed.
    Section(
        "CC6.1 Logical Access Provisioning",
        (
            "Access to production systems is granted on the principle of least "
            "privilege and requires documented approval from the system owner. "
            "Requests are raised through the internal service desk and recorded "
            "with the business justification, the entitlement granted and the "
            "approver identity.",
            "Access is reviewed on a quarterly basis. Reviewers confirm that each "
            "entitlement remains necessary for the individual's current role; "
            "entitlements not affirmatively confirmed are revoked at the close of "
            "the review window.",
            "Multi-factor authentication is enforced for all administrative access "
            "to production environments, and direct interactive access to database "
            "hosts is disabled in favour of audited, time-bound session brokering.",
        ),
    ),
    Section(
        "CC8.1 Change Management",
        (
            "Changes to production systems follow a documented change management "
            "process requiring peer review, automated test execution and approval "
            "prior to deployment. Emergency changes may be deployed ahead of "
            "approval but must be retrospectively reviewed within one business day.",
            "All changes are deployed through an automated pipeline that records "
            "the commit identifier, the approving reviewer and the deployment "
            "timestamp. Direct modification of production configuration outside "
            "the pipeline is technically prevented and alerts on attempt.",
        ),
    ),
    Section(
        "A1.1 Capacity Planning and Availability",
        (
            "Capacity is reviewed monthly against observed utilisation trends and "
            "committed customer growth. Autoscaling policies are configured for the "
            "application tier, with headroom maintained to absorb the loss of a "
            "single availability zone without service degradation.",
            "The service availability commitment is 99.9% measured monthly, "
            "excluding scheduled maintenance announced at least seventy-two hours "
            "in advance. Availability is measured from the external synthetic "
            "monitoring endpoints rather than from internal instrumentation.",
        ),
    ),
    Section(
        "CC1.4 Personnel Security",
        (
            "All personnel complete background screening prior to being granted "
            "access to customer data, subject to applicable local law. Screening is "
            "repeated for personnel who transfer into roles with elevated access.",
            "Security awareness training is completed on hire and annually "
            "thereafter, with completion tracked centrally. Personnel with access to "
            "production systems complete additional secure development training "
            "covering the organisation's threat model and secure coding standards.",
            "Upon termination, access is revoked through an automated workflow "
            "triggered by the human resources system, with revocation of all "
            "production entitlements targeted within four hours of the effective "
            "termination time.",
        ),
    ),
)

TABLE_HEADER = ("Provider", "Primary Region", "Backup Region", "Workload")
TABLE_ROWS = (
    (
        "Amazon Web Services",
        "us-east-1 (N. Virginia)",
        "us-west-2 (Oregon)",
        "Primary application tier",
    ),
    (
        "Google Cloud Platform",
        "europe-west1 (Belgium)",
        "europe-north1 (Finland)",
        "Analytics and BigQuery",
    ),
    ("Cloudflare", "Global anycast edge", "Global anycast edge", "CDN and WAF"),
)


def _draw_running_marks(page: pymupdf.Page, number: int, total: int) -> None:
    """Header and footer, positioned inside the bands the extractor scans."""
    page.insert_text((MARGIN, 34.0), RUNNING_HEADER, fontsize=8.0, fontname="helv")
    page.insert_text(
        (MARGIN, PAGE_H - 30.0),
        RUNNING_FOOTER.format(n=number, total=total),
        fontsize=8.0,
        fontname="helv",
    )


def _draw_table(page: pymupdf.Page, top: float) -> float:
    """Draw a ruled table so PyMuPDF's line-based detection can find it."""
    rows = (TABLE_HEADER, *TABLE_ROWS)
    col_w = [118.0, 122.0, 122.0, 121.0]
    row_h = 30.0
    left = MARGIN
    width = sum(col_w)

    for r, row in enumerate(rows):
        y = top + r * row_h
        x = left
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x, y, x + col_w[c], y + row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.6)
            page.insert_textbox(
                # Rect.__add__ insets the rectangle by these margins. This is
                # PyMuPDF geometry, not sequence concatenation, so RUF005's
                # unpacking rewrite would silently change the meaning.
                rect + (3, 4, -3, -3),  # noqa: RUF005
                cell,
                fontsize=7.2,
                fontname="hebo" if r == 0 else "helv",
                align=0,
            )
            x += col_w[c]

    bottom = top + len(rows) * row_h
    # Outer border, drawn last so the table reads as one figure.
    page.draw_rect(pymupdf.Rect(left, top, left + width, bottom), color=(0, 0, 0), width=1.0)
    return bottom


def build_soc2_pdf() -> bytes:
    """Render the multi-page sample report and return its bytes."""
    doc = pymupdf.open()

    def new_page() -> pymupdf.Page:
        # Page handles must not be retained across further mutations of the
        # document -- they detach from their parent. Always re-fetch by index.
        return doc.new_page(width=PAGE_W, height=PAGE_H)

    page = new_page()
    y = BODY_TOP
    text_w = PAGE_W - 2 * MARGIN

    for section in SECTIONS:
        needed = HEADING_SIZE * LEADING + 40.0
        if y + needed > BODY_BOTTOM:
            page = new_page()
            y = BODY_TOP

        page.insert_text((MARGIN, y), section.heading, fontsize=HEADING_SIZE, fontname="hebo")
        y += HEADING_SIZE * LEADING + 8.0

        for para in section.paragraphs:
            # Measure first so a paragraph is never silently clipped: textbox
            # returns a negative value when the content does not fit.
            probe = pymupdf.Rect(MARGIN, y, MARGIN + text_w, BODY_BOTTOM)
            fit = page.insert_textbox(
                probe, para, fontsize=BODY_SIZE, fontname="helv", lineheight=LEADING
            )
            if fit < 0:
                page = new_page()
                y = BODY_TOP
                probe = pymupdf.Rect(MARGIN, y, MARGIN + text_w, BODY_BOTTOM)
                page.insert_textbox(
                    probe, para, fontsize=BODY_SIZE, fontname="helv", lineheight=LEADING
                )
            # Advance by the consumed height: total box height minus the slack
            # the renderer reported as unused.
            used = (BODY_BOTTOM - y) - max(fit, 0.0)
            y += used + 12.0

        if section.heading.startswith("A1.2"):
            if y + 5 * 30.0 > BODY_BOTTOM:
                page = new_page()
                y = BODY_TOP
            y = _draw_table(page, y) + 18.0

    # Drawn last, once the total page count is known.
    total = doc.page_count
    for i in range(total):
        _draw_running_marks(doc[i], i + 1, total)

    data: bytes = doc.tobytes()
    doc.close()
    return data


# A document shaped to defeat naive flattening: two sibling objects each with
# their own nested "owner", a foreign key pointing into a *different*
# collection, and scalar context on the root that qualifies every record.
SECURITY_JSON: dict = {
    "vendor": "Acme Corp",
    "report_year": 2024,
    "report_type": "SOC 2 Type II",
    "teams": [
        {"id": 42, "name": "Security Engineering", "contact": "sec@acme.com"},
        {"id": 43, "name": "Site Reliability Engineering", "contact": "sre@acme.com"},
    ],
    "controls": [
        {
            "id": "CC7.3",
            "name": "Incident Notification Commitments",
            "owner_id": 42,
            "owner": {"team": "Security Engineering", "contact": "sec@acme.com"},
            "notification": {
                "sla_hours": 72,
                "severity_1_sla_hours": 24,
                "channels": ["email", "telephone escalation"],
            },
            "tests": [
                {"procedure": "Inspected incident tickets", "result": "No exceptions noted"},
                {"procedure": "Reviewed notification logs", "result": "No exceptions noted"},
            ],
        },
        {
            "id": "CC6.7",
            "name": "Third-Party Handling of Personal Information",
            "owner_id": 43,
            "owner": {"team": "Site Reliability Engineering", "contact": "sre@acme.com"},
            "subprocessors": [
                {"name": "Transactional email provider", "data": "recipient addresses"},
                {"name": "Payment processor", "data": "cardholder data"},
            ],
            "retention_days_after_termination": 90,
            "tests": [
                {"procedure": "Reviewed subprocessor agreements", "result": "No exceptions noted"}
            ],
        },
        {
            "id": "A1.2",
            "name": "Infrastructure Providers and Hosting Regions",
            "owner_id": 43,
            "owner": {"team": "Site Reliability Engineering", "contact": "sre@acme.com"},
            "providers": [
                {"provider": "Amazon Web Services", "primary": "us-east-1", "backup": "us-west-2"},
                {
                    "provider": "Google Cloud Platform",
                    "primary": "europe-west1",
                    "backup": "europe-north1",
                },
            ],
            "tests": [],
        },
    ],
}


def build_security_json() -> bytes:
    import json

    return json.dumps(SECURITY_JSON, indent=2).encode("utf-8")


def build_questions_json(questions: list[str]) -> bytes:
    import json

    return json.dumps({"questions": questions}).encode("utf-8")


SAMPLE_QUESTIONS: tuple[str, ...] = (
    "Do you have formally defined criteria for notifying a client during an incident "
    "that might impact the security of their data or systems? What are your SLAs for "
    "notification?",
    "Is personal information transmitted, processed, stored, or disclosed to or "
    "retained by third parties? If yes, describe.",
    "Which cloud providers do you rely on?",
    "Please specify the primary data center location/region of the underlying cloud "
    "infrastructure used to host the service(s) as well as the backup location(s).",
    "Which of the following, if any, are performed as part of your monitoring process "
    "for the service: Application Performance Monitoring (APM), End User Monitoring "
    "(EUM), Digital Experience Monitoring (DEM)?",
)

# Answered by neither fixture; used to assert the "not found" path on a
# document that is otherwise perfectly readable.
UNANSWERABLE_QUESTION = "What was the total compensation of the Chief Executive Officer?"


def build_scanned_pdf(pages: int = 3) -> bytes:
    """A PDF with pages but no extractable text, standing in for a scan."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.draw_rect(pymupdf.Rect(80, 80, 500, 600), color=(0.6, 0.6, 0.6), width=1.0)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def build_mostly_scanned_pdf(text_pages: int = 1, image_pages: int = 6) -> bytes:
    """Mostly image-only, with a few readable pages, to exercise the ratio check."""
    doc = pymupdf.open()
    for _ in range(text_pages):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_textbox(
            pymupdf.Rect(MARGIN, BODY_TOP, PAGE_W - MARGIN, BODY_BOTTOM),
            "This page carries readable text. " * 12,
            fontsize=BODY_SIZE,
            fontname="helv",
        )
    for _ in range(image_pages):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.draw_rect(pymupdf.Rect(80, 80, 500, 600), color=(0.6, 0.6, 0.6), width=1.0)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def build_encrypted_pdf() -> bytes:
    """A password-protected PDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((72, 144), "Restricted content.", fontsize=12, fontname="helv")
    data: bytes = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner-secret", user_pw="user-secret"
    )
    doc.close()
    return data
