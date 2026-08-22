# Document QA Service

Answers a list of questions from a single PDF or JSON document, using retrieval-augmented generation. Retrieved chunks are the only context given to `gpt-4o-mini`, and any question the document does not support comes back as `not_found` rather than a guess.

---

## Quick start

```bash
cp .env.example .env      # then put your OPENAI_API_KEY in it
docker compose up --build
```

Open http://localhost:8000 and upload a document plus a questions file. The detailed [Setup](#setup) section below covers running without Docker.

## Overview

Upload two files — a document (PDF or JSON) and a JSON list of questions — and get back one answer per question, each with citations pointing at the pages or JSON paths the answer came from.

- **Inputs:** PDF or JSON document, plus a JSON questions file. Format is detected from file content, not the extension.
- **Retrieval:** the document is chunked, embedded once with `text-embedding-3-small`, and indexed in FAISS. Each question retrieves its own top-10 chunks using MMR.
- **Generation:** `gpt-4o-mini` with structured output, constrained to the retrieved extracts.
- **Output:** one result per question with `status` of `answered`, `not_found`, or `error`, plus citations and token/cost metadata.

## Architecture

```
Document  →  Validate  →  Parse  →  Clean  →  Chunk  →  Embed  →  FAISS
                                                                     │
                                                                     ▼
Questions →  Embed  →  MMR retrieval  →  Retrieved context  →  gpt-4o-mini
                                                                     │
                                                                     ▼
                                          Grounding validation  →  Answers + citations
```

The document path runs once per request and is cached by content hash. The question path runs once per question, concurrently under a semaphore.

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

Example response (the document here is the synthetic test fixture):

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

Requires **Python 3.12** — 3.13+ has no FAISS wheel yet.

```bash
git clone https://github.com/CharithaVadamala9/document-qa-service.git
cd document-qa-service
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # then put your OPENAI_API_KEY in it
```

If you do not have Python 3.12 to hand, [uv](https://docs.astral.sh/uv/) will fetch it for you:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

`OPENAI_API_KEY` is the only value you must set; everything else has a working default. The test suite needs neither.

### Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000.

### Docker

```bash
docker build -t document-qa-service .
docker run --rm --env-file .env -p 8000:8000 document-qa-service
```

or `docker compose up --build`, which builds the same image and tags it `docqa:latest`.

The image is multi-stage: dependencies resolve in a builder stage, and the runtime stage carries no build tools or package manager. It runs as a non-root user (`uid 1001`), and the tiktoken vocabulary is baked in at build time so a network-restricted container does not fail its first request.

### Test

```bash
pytest                        # 208 tests, no network, no API key
pytest --cov=app              # 93% coverage
pytest -m live                # optional: 12 tests, real API calls, needs a key
ruff check app tests && ruff format --check app tests
mypy app
```

**No test in the default suite requires an API key or network access.** The embedder and the model are injected as fakes through FastAPI's dependency overrides. Tests marked `live` are deselected by default: model behaviour is probabilistic and cannot be a build gate.

## Evaluation

Prompt tweaks and chunking changes are easy to talk about and hard to judge. `evals/` scores them.

```bash
python -m evals.run --dataset evals/dataset_soc2.json                 # retrieval only, ~$0.001
python -m evals.run --dataset evals/dataset_soc2.json --with-answers  # + LLM grading
python -m evals.run --dataset ... --chunk-tokens 400 --top-k 8        # sweep a setting
```

Two stages, deliberately separate because they fail for different reasons:

- **Retrieval recall@k** — did the chunk holding the answer reach the prompt? Needs embeddings only, no LLM, so it is cheap enough to run on every change. This is the *ceiling* on answer quality: if the evidence never arrives, no amount of prompting recovers it.
- **Status accuracy and fact presence** — did the service answer when it should, refuse when it should, and state the expected fact? One LLM call per case.

Two datasets, deliberately of opposite shape, so a change has to hold on both:

| Document | Shape | Retrieval recall@k | Status accuracy |
|---|---|---|---|
| Bright Defense, 37pp | prose-heavy | 93% (from 60%) | 89–94% (from 67%) |
| Zintlr, 55pp | 34 dense control tables | 88% | 87–90% |

Recall is deterministic and reproduces exactly. Status accuracy is a **range** because generation is sampled: a single figure would misrepresent a number that moves several points on identical input. Judge a change by whether it clears the range, not the midpoint.

Questions that fail with an `upstream_error` are excluded from the denominator and reported separately — scoring an unreachable model as a wrong answer once made rate limiting look like a 45-point quality collapse.

The Bright Defense gap had three independent causes — bold headings missed by size-only detection, chunks too coarse, and a prompt that refused on partial evidence — which separated only because retrieval and generation are scored apart.

The second dataset earned its place by **rejecting** a change. Excluding repeated table headers from the embedded text looked well-motivated: 59 of 80 chunks began with byte-identical text. It held on the prose document and cost 11 points of recall on the table-heavy one, because the column names describe what the table holds, so questions about *test results* or *control activities* match against them. Reverted — a single-document gate would have kept it.

A third category, **grounded but less precise**, is advisory and never scored. It flags an answer properly supported by what it cites but drawn from a weaker source than the best available — answering "who can modify security group rules" from a general statement about the IT Security team rather than the control naming the IT Head. Scoring it as a failure would hide the distinction between *wrong* and *imprecise*.

The dataset points at a document that is **not committed** (`examples/*.pdf` is gitignored; it is a third-party file). Point `--document` at your own, or add a dataset of your own shape.

**On RAGAS and similar frameworks:** worth adding as an opt-in extra, not as the gate. RAGAS scores faithfulness and answer relevancy well, but it is LLM-as-judge — non-deterministic, costed per run, and a heavy dependency tree. This harness is deterministic and free for the retrieval half, which is what you want on every commit. Its weakness is the converse: a substring check cannot see an answer that is fluent, cited, and subtly misstates the source. The two are complements, not alternatives.

## Design decisions

**Why FastAPI** — async request handling matters here: a request holds many concurrent LLM calls, and typed validation at the boundary removes a class of hand-written checks.

**Why LangChain, and only partly** — its text splitter and `ChatOpenAI` structured-output support are genuinely useful, so those are used. Orchestration is not: chains hide the two things this service is judged on — bounded concurrency and per-request token accounting. Embeddings call the OpenAI SDK directly because LangChain's wrapper discards the `usage` field.

**Why `text-embedding-3-small`** — ~$0.02 per 1M tokens. A 50-page report costs about $0.0006 to index, versus $0.004 for `-3-large`, for a marginal ranking gain on keyword-dense compliance prose.

**Why token-aware chunking** — 500 tokens with 100-token overlap on prose. Character-based sizing makes the context budget unpredictable when packing ten chunks into a prompt; token-aware sizing makes it exact. 500 tokens is large enough to hold a typical control description whole — smaller fragments it, larger dilutes the embedding.

**Why separate PDF and JSON handling** — they fail differently. PDFs need layout analysis: running headers stripped (otherwise a 50-page report yields ~50 near-duplicate chunks that crowd out real content), tables rendered as markdown rather than linearised into column soup, and headings recovered from font size. JSON needs the opposite: character-splitting it severs records mid-object, and flattening to `owner.contact` destroys the grouping that says which owner a contact belongs to. Records are rendered as indented text, ancestor scalars are inherited, and foreign keys are resolved to labels — `owner_id: 42` is unreachable by semantic search, `owner_id: 42 (Security Engineering)` is not.

**Why overlap only on prose** — repeating a self-contained JSON record or table row gives it two nearly identical vectors, letting one record win two of the ten retrieval slots. Structured segments get header repetition instead: a split table repeats its column row, a split record repeats its title and ancestor context.

**Why MMR** — with overlapping chunks, plain top-k routinely returns several near-copies of one passage. MMR (`fetch_k=40`, `λ=0.5`) buys diversity with no extra API calls.

**Why 500 tokens and top-10** — measured, not guessed. The original configuration was 600 tokens at top-5, which scored 73% retrieval recall against a public 37-page SOC 2 report. Halving the chunk and doubling `k` to **500/top-10** raised that to 93%, and it is what ships. Smaller chunks localise the evidence; a wider `k` compensates for the finer granularity without lengthening the prompt much, since each chunk is shorter. See `evals/` to re-run the sweep.

**Why bounded concurrency** — questions run in parallel so 20 questions take roughly the time of the slowest, not the sum. A per-request semaphore stops one large question list from monopolising the quota; a global cap stops N concurrent requests multiplying into a rate-limit wall.

**Why no semantic chunking or reranking** — both cost additional model calls for a modest gain on this document type. Listed under future work rather than built.

## Grounding and not-found behaviour

The retrieved chunks are the only source of truth. Three mechanisms enforce it:

1. **The model cites by extract number, never by page.** It sees `[1]`, `[2]`, `[3]`; page numbers and JSON paths are attached afterwards from chunk metadata. A fabricated page number is structurally impossible.
2. **Out-of-range indices are discarded.** The source list is model output and is treated as untrusted.
3. **An answer with no valid citation is downgraded to `not_found`.** This is what turns "please don't hallucinate" from a prompt request into an enforced invariant.
4. **Figures are verified against the cited text.** If an answer asserts a number that appears in neither the extracts it cited nor the question, it is downgraded. A notification SLA of 24 hours where the document says 72 reads as authoritative and is wrong; this catches it deterministically, with no second model call. Disable with `VERIFY_NUMERIC_GROUNDING=false`.

**Partial answers are requested, not guaranteed.** The prompt instructs the model to answer the supported part and name the gap rather than refuse outright, because a flat `not_found` on a partially-answerable question throws away information the caller needs. It usually does. On list-style questions that name several items at once — "do you perform APM, EUM and DEM monitoring?" where the document covers only some — it still sometimes refuses the whole question instead of answering the covered part.

No guardrail is involved; these never reach one. Two prompt revisions were written to fix it and measured on both eval datasets. Each fixed the target case and cost more elsewhere, so neither was kept, and the behaviour is recorded as a non-strict `xfail` in `tests/unit/test_answer_edge_cases.py` rather than described as solved.

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
| `CHUNK_SIZE_TOKENS` | `500` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `100` | Prose overlap |
| `RETRIEVAL_TOP_K` | `10` | Chunks per question |
| `MMR_FETCH_K` / `MMR_LAMBDA` | `40` / `0.5` | MMR candidate pool and diversity weight |
| `VERIFY_NUMERIC_GROUNDING` | `true` | Downgrade answers asserting uncited figures |
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

Re-running a request against the same document bills only the query embeddings; the document's own embedding cost is paid once. `tests/unit/test_qa_service.py` and `tests/integration/test_api.py` assert that relationship directly rather than pinning a token count that would drift with the fixture.

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
- **Prompt injection.** Documents are untrusted, so a supplier's PDF may carry text aimed at the model. The prompt frames extracts as data, never instructions — but that is a request, not a guarantee, so the defences that matter are structural: the model returns extract *numbers* rather than page numbers, out-of-range indices are discarded, uncited answers are downgraded, and figures are checked against the cited text. `tests/unit/test_injection.py` covers both halves: deterministic tests that assume the model *has already been compromised* and check the damage is contained, plus `-m live` tests of whether `gpt-4o-mini` actually resists. Payloads are deliberately *not* stripped during ingestion — the caller is entitled to see what their document contains.

## Assumptions and limitations

- **Test fixtures are synthetic; the eval documents are not.** The assessment's sample PDF URL (`productfruits.com/docs/soc2-type2.pdf`) now redirects to an HTML page, so the test suite generates an equivalent SOC 2-style report — which keeps tests hermetic and the repository free of third-party binaries. Answer quality is measured separately, against two real published SOC 2 reports of deliberately opposite shape (see Evaluation). Those PDFs are gitignored rather than redistributed.
- **Scanned PDFs are rejected**, not OCR'd.
- **One document per request.**
- **Dense retrieval is weak on long, repetitive control tables.** In the 55-page table-heavy report, a handful of questions still fail: an exact control id (`CC6.6.1`) and the specific statement of who may modify security group rules both sit in one chunk that never enters the top-k, while its neighbouring table chunks rank fine. Similarity scores across those chunks bunch into a narrow band (roughly 0.28–0.58), so ranking among them is close to arbitrary. This is a known limitation of single-vector dense retrieval over near-identical structured text, not a defect in extraction — the content is extracted correctly and is present in the index. The fix is lexical matching alongside vectors; see Future Improvements.
- **The index is in-memory and per-process.** With multiple workers each keeps its own cache; the Dockerfile therefore defaults to a single worker, and scaling is by replica.
- **No authentication or multi-tenancy** — this is a single-tenant service.
- `estimated_cost_usd` comes from a static price table, not from the provider.

## Licensing note

PyMuPDF is **AGPL-3.0**. Fine for an assessment, but a commercial deployment would need either an Artifex licence or a swap to `pypdf` (BSD). The extraction code sits behind a narrow interface for that reason, though the layout analysis (positional header detection, table extraction) would need reworking, since `pypdf` does not expose bounding boxes.

## Future improvements

- OCR for scanned PDFs.
- Shared index store (Redis/S3 + a hosted vector DB) so the cache survives restarts and is shared across workers.
- **Hybrid lexical + vector retrieval (BM25 alongside embeddings).** This is the single highest-value next step, and the remaining eval failures point straight at it. Exact identifiers — control ids like `CC6.6.1`, `A1.2.5` — are precisely what dense embeddings handle worst and lexical matching handles best: BM25 would rank the exact-token match first, immediately. It also addresses the repetitive-table weakness above, where dozens of chunks are near-identical in embedding space but trivially distinguishable by their literal content. A reciprocal-rank fusion of the two rankings would need no change to chunking or the vector store.
- An LLM-judge stage (RAGAS or equivalent) alongside the deterministic harness, to catch fluent-but-subtly-wrong answers that substring checks miss.
- Cross-encoder reranking over a larger candidate pool.
