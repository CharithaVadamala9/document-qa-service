# docqa

Answers a list of questions from a single PDF or JSON document, using retrieval-augmented generation. Retrieved chunks are the only context given to `gpt-4o-mini`, and any question the document does not support comes back as `not_found` rather than a guess.

---

## Overview

Upload two files — a document (PDF or JSON) and a JSON list of questions — and get back one answer per question, each with citations pointing at the pages or JSON paths the answer came from.

- **Inputs:** PDF or JSON document, plus a JSON questions file. Format is detected from file content, not the extension.
- **Retrieval:** the document is chunked, embedded once with `text-embedding-3-small`, and indexed in FAISS. Each question retrieves its own top-5 chunks using MMR.
- **Generation:** `gpt-4o-mini` with structured output, constrained to the retrieved extracts.
- **Output:** one result per question with `status` of `answered`, `not_found`, or `error`, plus citations and token/cost metadata.

## Architecture

```
Upload → Validate → Detect type → Parse → Clean → Segment → Chunk
                                                              ↓
                    Grounded answer + citations ← gpt-4o-mini ←┘ Embed → FAISS → MMR retrieve
```

| Module | Responsibility |
|---|---|
| `app/api/` | HTTP: routing, upload validation, error envelope, request-id correlation, admission control |
| `app/ingestion/` | Bytes → chunks. PDF layout analysis, JSON structure preservation, token-aware chunking |
| `app/retrieval/` | Embedding, the FAISS index, and MMR selection |
| `app/llm/` | Prompting, structured output, citation resolution, grounding enforcement |
| `app/services/` | Orchestration: document cache, bounded concurrency, timeouts, usage totals |
| `app/core/` | Settings, domain types, exceptions, logging, metrics, retry policy |

Nothing below `app/api/` imports FastAPI, and nothing below `app/llm/` or `app/retrieval/` imports the OpenAI SDK. Both provider-backed dependencies sit behind protocols, which is what lets the whole suite run offline.

## API

### `POST /api/v1/qa`

`multipart/form-data`:

| Field | Type | Description |
|---|---|---|
| `document_file` | PDF or JSON | The source document |
| `questions_file` | JSON | `["question", ...]`, `{"questions": [...]}`, or `[{"question": "..."}]` |

```bash
curl -s -X POST http://localhost:8000/api/v1/qa \
  -F "document_file=@soc2.pdf" \
  -F "questions_file=@questions.json"
```

```json
{
  "document": "soc2.pdf",
  "results": [
    {
      "question": "Which cloud providers do you rely on?",
      "answer": "Amazon Web Services, Google Cloud Platform and Cloudflare.",
      "status": "answered",
      "citations": [
        {
          "chunk_id": "61054f7b33302ad7",
          "source": "soc2.pdf",
          "snippet": "| Provider | Primary Region | Backup Region |…",
          "pages": [2, 3],
          "json_paths": [],
          "section": "A1.2 Infrastructure Providers and Hosting Regions"
        }
      ],
      "latency_ms": 812.4,
      "error_code": null
    }
  ],
  "metadata": {
    "questions_processed": 1,
    "answered": 1,
    "not_found": 0,
    "failed": 0,
    "document_type": "pdf",
    "chunk_count": 3,
    "document_cache_hit": false,
    "processing_time_ms": 1204.7,
    "usage": {
      "input_tokens": 1840,
      "output_tokens": 96,
      "embedding_tokens": 1056,
      "total_tokens": 2992,
      "estimated_cost_usd": 0.000359
    }
  }
}
```

Errors use one envelope, always:

```json
{
  "error": {
    "code": "no_extractable_text",
    "message": "Almost no text could be extracted from this PDF; it appears to be scanned or image-only. OCR is not supported — please supply a text-based PDF.",
    "detail": { "pages": 12, "pages_with_text": 0 }
  },
  "request_id": "9f2c1b7e4a6d4f0e"
}
```

| Status | When |
|---|---|
| 413 | Upload exceeds a size limit |
| 415 | Content is neither PDF nor JSON |
| 422 | Malformed/encrypted/scanned document, bad or oversized questions file |
| 502 | Provider failed after the retry budget |
| 503 | Service at capacity (`Retry-After` set) |
| 504 | Request exceeded its time limit |

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Minimal upload UI |
| `GET /health` | Liveness. Deliberately does **not** call OpenAI — a provider outage must not restart healthy containers |
| `GET /ready` | Readiness: reports `not_ready` when `OPENAI_API_KEY` is missing |
| `GET /metrics` | Request counts, latency p50/p95, token totals, estimated spend |
| `GET /docs` | OpenAPI UI |

## Setup

Requires Python 3.12. (3.13+ is not yet supported by the FAISS/LangChain wheels.)

```bash
git clone <your-repo-url>
cd docqa
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # then add your OPENAI_API_KEY
```

### Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000.

### Docker

```bash
docker build -t docqa .
docker run --env-file .env -p 8000:8000 docqa
```

or `docker compose up --build`.

### Test

```bash
pytest                        # 159 tests
pytest --cov=app              # ~90% coverage
ruff check app tests && ruff format --check app tests
mypy app
```

**No test requires an API key or network access.** The embedder and the model are injected as fakes through FastAPI's dependency overrides.

## Design decisions

**Why FastAPI** — async request handling matters here: a request holds many concurrent LLM calls, and typed validation at the boundary removes a class of hand-written checks.

**Why LangChain, and only partly** — its text splitter and `ChatOpenAI` structured-output support are genuinely useful, so those are used. Orchestration is not: chains hide the two things this service is judged on — bounded concurrency and per-request token accounting. Embeddings call the OpenAI SDK directly because LangChain's wrapper discards the `usage` field.

**Why `text-embedding-3-small`** — ~$0.02 per 1M tokens. A 50-page report costs about $0.0006 to index, versus $0.004 for `-3-large`, for a marginal ranking gain on keyword-dense compliance prose.

**Why token-aware chunking** — 600 tokens with 100 overlap. Character-based sizing makes the context budget unpredictable when packing five chunks into a prompt. 600 tokens fits a typical control description whole; smaller fragments it, larger dilutes the embedding.

**Why separate PDF and JSON handling** — they fail differently. PDFs need layout analysis: running headers stripped (otherwise a 50-page report yields ~50 near-duplicate chunks that crowd out real content), tables rendered as markdown rather than linearised into column soup, and headings recovered from font size. JSON needs the opposite: character-splitting it severs records mid-object, and flattening to `owner.contact` destroys the grouping that says which owner a contact belongs to. Records are rendered as indented text, ancestor scalars are inherited, and foreign keys are resolved to labels — `owner_id: 42` is unreachable by semantic search, `owner_id: 42 (Security Engineering)` is not.

**Why overlap only on prose** — repeating a self-contained JSON record or table row gives it two nearly identical vectors, letting one record win two of five retrieval slots. Structured segments get header repetition instead: a split table repeats its column row, a split record repeats its title and ancestor context.

**Why MMR** — with overlapping chunks, plain top-5 routinely returns several near-copies of one passage. MMR (`fetch_k=20`, `λ=0.5`) buys diversity with no extra API calls.

**Why bounded concurrency** — questions run in parallel so 20 questions take roughly the time of the slowest, not the sum. A per-request semaphore stops one large question list from monopolising the quota; a global cap stops N concurrent requests multiplying into a rate-limit wall.

**Why no semantic chunking or reranking** — both cost additional model calls for a modest gain on this document type. Listed under future work rather than built.

## Grounding and not-found behaviour

The retrieved chunks are the only source of truth. Three mechanisms enforce it:

1. **The model cites by extract number, never by page.** It sees `[1]`, `[2]`, `[3]`; page numbers and JSON paths are attached afterwards from chunk metadata. A fabricated page number is structurally impossible.
2. **Out-of-range indices are discarded.** The source list is model output and is treated as untrusted.
3. **An answer with no valid citation is downgraded to `not_found`.** This is what turns "please don't hallucinate" from a prompt request into an enforced invariant.

Unsupported questions return exactly:

```
Not found in the provided document.
```

A scanned PDF is reported as `no_extractable_text`, not as fifty `not_found` answers — the document is the problem, not the questions.

## Configuration

Every limit is environment-driven; see `.env.example` for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. The only mandatory value |
| `LLM_MODEL` | `gpt-4o-mini` | Answer generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Retrieval embeddings |
| `CHUNK_SIZE_TOKENS` | `600` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `100` | Prose overlap |
| `RETRIEVAL_TOP_K` | `5` | Chunks per question |
| `MMR_FETCH_K` / `MMR_LAMBDA` | `20` / `0.5` | MMR candidate pool and diversity weight |
| `MAX_FILE_SIZE_MB` | `20` | Document upload limit |
| `MAX_PDF_PAGES` | `300` | Page limit |
| `MAX_QUESTIONS` | `50` | Questions per request |
| `MAX_CONCURRENT_QUESTIONS` | `5` | Per-request fan-out cap |
| `MAX_CONCURRENT_REQUESTS` | `32` | Admission control; excess sheds with 503 |
| `LLM_TIMEOUT_SECONDS` | `30` | Per provider call |
| `QUESTION_TIMEOUT_SECONDS` | `60` | Per question |
| `REQUEST_TIMEOUT_SECONDS` | `300` | Whole request |
| `DOCUMENT_CACHE_SIZE` | `32` | Cached indexes; `0` disables |

Settings are cross-validated at startup: overlap must be smaller than chunk size, `MMR_FETCH_K` at least `RETRIEVAL_TOP_K`, and the question timeout at least the LLM timeout.

## Performance

- The document is parsed and embedded **once per request**, in a single batched call — not once per chunk and not once per question.
- Questions run concurrently under a semaphore.
- Blocking work (PyMuPDF parsing, FAISS operations) runs via `asyncio.to_thread`, so an ingest never stalls the event loop for other requests.
- Indexes are cached by content hash, so re-asking against the same document skips parsing and embedding entirely. Concurrent requests for the same document share one build rather than each paying for it.

Measured on the test fixture with fakes: a 6-question request re-run against the same document drops from 1,056 embedding tokens to 46 — only the query embeddings.

## Observability

One JSON object per line on stdout. `request_id` is bound to a context variable, so it appears on every line emitted while handling a request, including inside concurrent fan-out and worker threads.

```json
{"event": "qa.completed", "request_id": "9f2c…", "document_type": "pdf",
 "question_count": 6, "answered": 5, "not_found": 1, "failed": 0,
 "chunk_count": 3, "cache_hit": false, "input_tokens": 640,
 "output_tokens": 128, "embedding_tokens": 1056, "cost_usd": 0.000194,
 "total_latency_ms": 307.02, "level": "info", "timestamp": "…"}
```

Logs carry chunk **ids**, never chunk text. API keys, prompts and document content are never logged, and error responses never include stack traces or provider internals.

`GET /metrics` exposes request counts by status, latency p50/p95/max over a rolling window, token totals, and estimated spend — scoped to one worker process, which the payload states.

## Security

- The API key is read from the environment into a `SecretStr` and never logged. `.env` is gitignored; only `.env.example` is committed.
- Documents are processed **in memory** — nothing is written to disk. Uploaded filenames are reduced to a display label with path separators and traversal segments stripped, and are never used as filesystem paths.
- File type is determined by content sniffing. A `%PDF-` header check is enforced before parsing, because MuPDF will otherwise happily parse HTML — so a PDF URL that has started redirecting to an error page would silently ingest as a valid document.
- Size, page-count, question-count, JSON depth and JSON node limits are all enforced before expensive work begins.
- The prompt instructs the model to treat extract content as data, never instructions, since documents are untrusted input.

## Assumptions and limitations

- **The fixture is synthetic.** The assessment's sample PDF URL (`productfruits.com/docs/soc2-type2.pdf`) now redirects to an HTML page, so tests generate an equivalent SOC 2-style report instead. It exercises the pipeline thoroughly, but answer quality on a real 40–80 page report is not yet measured.
- **Scanned PDFs are rejected**, not OCR'd.
- **One document per request.**
- **The index is in-memory and per-process.** With multiple workers each keeps its own cache; the Dockerfile therefore defaults to a single worker, and scaling is by replica.
- **No authentication or multi-tenancy** — this is a single-tenant service.
- `estimated_cost_usd` comes from a static price table, not from the provider.

## Licensing note

PyMuPDF is **AGPL-3.0**. Fine for an assessment, but a commercial deployment would need either an Artifex licence or a swap to `pypdf` (BSD). The extraction code sits behind a narrow interface for that reason, though the layout analysis (positional header detection, table extraction) would need reworking, since `pypdf` does not expose bounding boxes.

## Future improvements

- OCR for scanned PDFs.
- Shared index store (Redis/S3 + a hosted vector DB) so the cache survives restarts and is shared across workers.
- Hybrid BM25 + vector retrieval, which helps on exact identifiers like control IDs where embeddings are weak.
- Cross-encoder reranking over a larger candidate pool.
- A retrieval evaluation set with recall@k and answer-accuracy scoring, so chunking and `top_k` changes can be measured instead of argued.
