# Build stage: resolve dependencies into a self-contained virtualenv.
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN pip install uv==0.5.11

WORKDIR /build

# Dependency metadata is copied before the source so that editing application
# code does not invalidate the (slow) dependency layer.
COPY pyproject.toml README.md ./
RUN mkdir -p app && touch app/__init__.py \
    && uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# Pre-download the tiktoken vocabulary. Without this the first request in a
# network-restricted container would fail, and chunk sizing would silently
# depend on whether the download succeeded.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN mkdir -p /opt/tiktoken \
    && /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"


# Runtime stage: no build tools, no package manager, no source history.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

# Non-root: a container that only reads uploads from memory has no reason to
# run with write access to its own filesystem.
RUN groupadd --system --gid 1001 docqa \
    && useradd --system --uid 1001 --gid docqa --no-create-home docqa

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken /opt/tiktoken

WORKDIR /app
COPY --chown=docqa:docqa app ./app
COPY --chown=docqa:docqa static ./static

USER docqa
EXPOSE 8000

# Talks to the process only; a provider outage must not fail the liveness
# probe and trigger a restart loop.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# Single worker by default: the document cache is per-process, so additional
# workers each keep their own. Scale with replicas, or raise this once a
# shared index store is in place.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
