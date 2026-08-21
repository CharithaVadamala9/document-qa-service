"""Token counting and cost accounting."""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from app.core.errors import ConfigurationError

# USD per 1M tokens, keyed by model so a model swap cannot report a stale price.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.020, 0.0),
    "text-embedding-3-large": (0.130, 0.0),
}

_DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=8)
def _encoder(model: str) -> tiktoken.Encoding:
    """Load the BPE encoder.

    tiktoken downloads its vocabulary on first use. The Docker image warms this
    cache at build time; if it is still unreachable we fail loudly rather than
    approximate, because a wrong token count silently corrupts chunk sizing.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        pass
    except Exception as exc:
        raise ConfigurationError(f"Could not load the tokenizer for {model!r}.") from exc

    try:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception as exc:
        raise ConfigurationError(
            f"Could not load the {_DEFAULT_ENCODING!r} tokenizer. The tiktoken "
            "cache is unavailable and cannot be downloaded."
        ) from exc


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    return len(_encoder(model).encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    encoder = _encoder(model)
    tokens = encoder.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def estimate_cost_usd(
    *,
    llm_model: str,
    embedding_model: str,
    prompt_tokens: int,
    completion_tokens: int,
    embedding_tokens: int,
) -> float:
    """Return 0.0 for unpriced models rather than guessing a price."""
    llm_in, llm_out = _PRICING.get(llm_model, (0.0, 0.0))
    embed_in, _ = _PRICING.get(embedding_model, (0.0, 0.0))
    total = prompt_tokens * llm_in + completion_tokens * llm_out + embedding_tokens * embed_in
    return round(total / 1_000_000, 6)
