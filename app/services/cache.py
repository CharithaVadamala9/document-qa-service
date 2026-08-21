"""In-process LRU cache of built document indexes.

Keyed by content hash, so re-asking against the same document skips parsing and
embedding entirely. Concurrent requests for the same document share one build
rather than each paying for it.

Scope: one worker process. A multi-worker deployment gets a cache per worker,
which is correct but not shared; see the README for the shared-store swap.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.models import ParsedDocument
from app.retrieval.retriever import DocumentIndex

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CachedDocument:
    document: ParsedDocument
    index: DocumentIndex


def cache_key(data: bytes, settings: Settings) -> str:
    """Hash the content together with the settings that change the result.

    Chunking and embedding parameters are part of the key: without them, a
    tuning change would silently serve an index built under the old values.
    """
    digest = hashlib.sha256(data)
    fingerprint = (
        f"{settings.embedding_model}|{settings.embedding_dimensions}|"
        f"{settings.chunk_size_tokens}|{settings.chunk_overlap_tokens}"
    )
    digest.update(fingerprint.encode())
    return digest.hexdigest()


class DocumentCache:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._entries: OrderedDict[str, CachedDocument] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[CachedDocument]] = {}
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._entries)

    async def get_or_create(
        self, key: str, factory: Callable[[], Coroutine[Any, Any, CachedDocument]]
    ) -> tuple[CachedDocument, bool]:
        """Return the entry and whether it came from the cache."""
        if self._max_size <= 0:
            return await factory(), False

        async with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                logger.info("document_cache.hit", key=key[:12], size=len(self._entries))
                return cached, True

            task = self._inflight.get(key)
            owns_build = task is None
            if task is None:
                task = asyncio.create_task(factory())
                self._inflight[key] = task

        try:
            # A waiter being cancelled does not cancel the shared build, so one
            # client disconnecting cannot abort the work others are awaiting.
            entry = await task
        finally:
            if owns_build:
                async with self._lock:
                    self._inflight.pop(key, None)

        if owns_build:
            async with self._lock:
                self._entries[key] = entry
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_size:
                    evicted, _ = self._entries.popitem(last=False)
                    logger.info("document_cache.evicted", key=evicted[:12])

        # A caller that joined an in-flight build did not pay for it either.
        return entry, not owns_build
