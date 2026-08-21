"""Environment-driven settings. All operational limits live here."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["local", "test", "production"] = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    openai_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_answer_tokens: int = Field(default=512, gt=0)

    max_file_size_mb: int = Field(default=20, gt=0)
    max_questions_file_kb: int = Field(default=256, gt=0)
    max_pdf_pages: int = Field(default=300, gt=0)
    max_questions: int = Field(default=50, gt=0)
    max_question_chars: int = Field(default=1000, gt=0)
    max_json_depth: int = Field(default=64, gt=0)
    max_json_nodes: int = Field(default=200_000, gt=0)

    # A PDF whose pages yield less than this is image-only; we reject rather
    # than answer "not found" for every question.
    min_chars_per_page: int = Field(default=40, ge=0)
    min_extractable_page_ratio: float = Field(default=0.2, ge=0.0, le=1.0)

    chunk_size_tokens: int = Field(default=600, gt=0)
    chunk_overlap_tokens: int = Field(default=100, ge=0)
    min_chunk_tokens: int = Field(default=20, ge=0)
    boilerplate_page_ratio: float = Field(default=0.7, gt=0.0, le=1.0)
    header_footer_band: float = Field(default=0.08, gt=0.0, lt=0.5)

    retrieval_top_k: int = Field(default=5, gt=0)
    mmr_fetch_k: int = Field(default=20, gt=0)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding_batch_size: int = Field(default=100, gt=0)

    max_concurrent_questions: int = Field(default=5, gt=0)
    max_concurrent_llm_calls: int = Field(default=16, gt=0)
    max_concurrent_requests: int = Field(default=32, gt=0)

    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    question_timeout_seconds: float = Field(default=60.0, gt=0)
    request_timeout_seconds: float = Field(default=300.0, gt=0)

    llm_max_attempts: int = Field(default=4, ge=1)
    llm_backoff_base_seconds: float = Field(default=0.5, gt=0)
    llm_backoff_max_seconds: float = Field(default=8.0, gt=0)

    document_cache_size: int = Field(default=32, ge=0)

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_questions_file_bytes(self) -> int:
        return self.max_questions_file_kb * 1024

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.get_secret_value())

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        if self.mmr_fetch_k < self.retrieval_top_k:
            raise ValueError("mmr_fetch_k must be at least retrieval_top_k")
        if self.question_timeout_seconds < self.llm_timeout_seconds:
            raise ValueError("question_timeout_seconds must be at least llm_timeout_seconds")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
