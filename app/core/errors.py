"""Domain exceptions.

Each carries the HTTP status it surfaces as, so no layer below the API needs
to import FastAPI. Messages are caller-facing: they must never leak stack
traces, provider internals, secrets, or document content.
"""

from __future__ import annotations

from typing import Any


class DocQAError(Exception):
    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail


class ValidationError(DocQAError):
    code = "validation_error"
    http_status = 422


class UnsupportedFileType(ValidationError):
    code = "unsupported_file_type"
    http_status = 415


class PayloadTooLarge(DocQAError):
    code = "payload_too_large"
    http_status = 413


class MalformedDocument(ValidationError):
    code = "malformed_document"


class EncryptedDocument(ValidationError):
    code = "encrypted_document"


class NoExtractableText(ValidationError):
    """Image-only PDF. Distinct from "not found" so the caller knows the
    document is the problem, not the question."""

    code = "no_extractable_text"


class UpstreamError(DocQAError):
    code = "upstream_error"
    http_status = 502


class UpstreamTimeout(DocQAError):
    code = "upstream_timeout"
    http_status = 504


class RequestTimeout(DocQAError):
    code = "request_timeout"
    http_status = 504


class ServiceOverloaded(DocQAError):
    """Admission control refused the request, keeping latency bounded for
    those already in flight."""

    code = "service_overloaded"
    http_status = 503


class ConfigurationError(DocQAError):
    code = "configuration_error"
    http_status = 500
