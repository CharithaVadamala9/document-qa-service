"""Offline evaluation harness.

Two stages, because they fail for different reasons and cost differently:

  retrieval  Did the chunk holding the answer reach the top-k? Needs embeddings
             only -- no LLM calls -- so it is cheap enough to run on every
             change. Retrieval recall is the ceiling on answer quality: if the
             evidence never reaches the prompt, no amount of prompting helps.

  answers    Did the service return the right status, and does an answered
             result contain the expected fact? Costs one LLM call per case.

Usage:
    python -m evals.run --dataset evals/dataset_soc2.json               # retrieval only
    python -m evals.run --dataset evals/dataset_soc2.json --with-answers
    python -m evals.run --dataset ... --top-k 8 --chunk-tokens 400      # sweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import openai

from app.core.config import Settings
from app.core.logging import configure_logging
from app.core.models import AnswerStatus
from app.ingestion.loader import load_document
from app.llm.client import OpenAIAnswerGenerator
from app.retrieval.embedder import OpenAIEmbedder
from app.retrieval.retriever import DocumentIndex
from app.services.cache import DocumentCache
from app.services.qa_service import QAService


@dataclass
class CaseResult:
    id: str
    expect: str
    retrieved_evidence: bool
    retrieved_pages: list[int]
    status: str | None = None
    answer_ok: bool | None = None
    # Grounded, but drawn from a weaker source than the best one available.
    # Reported, never scored: the answer is not wrong, and forcing it into a
    # failure would hide the difference between "incorrect" and "imprecise".
    imprecise: bool | None = None


# Typeset documents use non-breaking hyphens and en dashes where a keyboard
# would type "-". Without normalising, "semi-annually" in the dataset fails to
# match "semi\u2011annually" in the PDF and a hit is scored as a miss.
_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})


def _normalise(text: str) -> str:
    return " ".join(text.translate(_DASHES).lower().split())


def _hit(case: dict[str, Any], chunks: list[Any]) -> tuple[bool, list[int]]:
    """Evidence counts as retrieved if any expected phrase appears in the
    retrieved text, or failing that if an expected page was retrieved."""
    text = _normalise(" ".join(c.chunk.text for c in chunks))
    pages = sorted({p for c in chunks for p in c.chunk.pages})

    evidence = [_normalise(e) for e in case.get("evidence", [])]
    if evidence:
        return any(e in text for e in evidence), pages
    expected_pages = set(case.get("pages", []))
    if expected_pages:
        return bool(expected_pages & set(pages)), pages
    return True, pages  # not_found cases have no evidence to find


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--document", help="overrides the path in the dataset")
    parser.add_argument("--with-answers", action="store_true", help="also call the LLM")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--chunk-tokens", type=int)
    return parser.parse_args()


async def main(
    args: argparse.Namespace, dataset: dict[str, Any], document_path: pathlib.Path, data: bytes
) -> int:
    overrides: dict[str, Any] = {}
    if args.top_k:
        overrides["retrieval_top_k"] = args.top_k
        overrides["mmr_fetch_k"] = max(args.top_k * 4, 20)
    if args.chunk_tokens:
        overrides["chunk_size_tokens"] = args.chunk_tokens
        overrides["chunk_overlap_tokens"] = min(100, args.chunk_tokens // 4)

    settings = Settings(**overrides) if overrides else Settings()
    configure_logging(settings.model_copy(update={"log_level": "ERROR"}))
    if not settings.has_openai_key:
        print("OPENAI_API_KEY is required (embeddings at minimum).", file=sys.stderr)
        return 2

    document = load_document(data, filename=document_path.name, settings=settings)
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value(), max_retries=0)
    embedder = OpenAIEmbedder(client, settings)

    try:
        index = await DocumentIndex.build(document.chunks, embedder=embedder, settings=settings)
        cases = dataset["cases"]
        retrievals = await asyncio.gather(*(index.retrieve(c["question"]) for c in cases))

        results = [
            CaseResult(
                id=c["id"], expect=c["expect"], retrieved_evidence=hit, retrieved_pages=pages
            )
            for c, (hit, pages) in (
                (c, _hit(c, r.chunks)) for c, r in zip(cases, retrievals, strict=True)
            )
        ]

        if args.with_answers:
            service = QAService(
                embedder=embedder,
                generator=OpenAIAnswerGenerator(settings),
                settings=settings,
                cache=DocumentCache(4),
            )
            outcome = await service.answer(
                data=data,
                filename=document_path.name,
                questions=[c["question"] for c in cases],
            )
            for result, case, answered in zip(results, cases, outcome.results, strict=True):
                result.status = answered.status.value
                if answered.status is AnswerStatus.ANSWERED:
                    body = _normalise(answered.answer)
                    expected = [_normalise(e) for e in case.get("evidence", [])]
                    result.answer_ok = any(e in body for e in expected) if expected else True
                    preferred = [_normalise(p) for p in case.get("prefer", [])]
                    if preferred:
                        result.imprecise = not any(p in body for p in preferred)
    finally:
        await client.close()

    # --- report ---------------------------------------------------------
    answerable = [r for r in results if r.expect == "answered"]
    recall = sum(r.retrieved_evidence for r in answerable) / len(answerable)

    print(f"\ndocument   {document_path.name}")
    print(
        f"chunks     {len(document.chunks)}  "
        f"(chunk_size={settings.chunk_size_tokens}, top_k={settings.retrieval_top_k})"
    )
    print(
        f"\nretrieval recall@{settings.retrieval_top_k}: "
        f"{recall:.0%}  ({sum(r.retrieved_evidence for r in answerable)}/{len(answerable)})"
    )

    misses = [r for r in answerable if not r.retrieved_evidence]
    if misses:
        print("  evidence NOT retrieved for:")
        for r in misses:
            print(f"    - {r.id}  (got pages {r.retrieved_pages})")

    if args.with_answers:
        correct_status = sum((r.status == "answered") == (r.expect == "answered") for r in results)
        print(
            f"\nstatus accuracy:  {correct_status / len(results):.0%}  "
            f"({correct_status}/{len(results)})"
        )
        wrong = [r for r in results if (r.status == "answered") != (r.expect == "answered")]
        for r in wrong:
            label = "false negative" if r.expect == "answered" else "HALLUCINATION RISK"
            print(f"    - {r.id}: expected {r.expect}, got {r.status}   [{label}]")

        graded = [r for r in results if r.answer_ok is not None]
        if graded:
            ok = sum(bool(r.answer_ok) for r in graded)
            print(f"\nanswer contains expected fact: {ok / len(graded):.0%} ({ok}/{len(graded)})")
            for r in graded:
                if not r.answer_ok:
                    print(f"    - {r.id}: answered but expected fact absent")

        # Advisory only. These answers are supported by what they cite; they
        # simply did not reach the most precise statement in the document.
        checked = [r for r in results if r.imprecise is not None]
        if checked:
            imprecise = [r for r in checked if r.imprecise]
            print(
                f"\ngrounded but less precise than the best source: "
                f"{len(imprecise)}/{len(checked)}   [advisory, not scored]"
            )
            for r in imprecise:
                print(f"    - {r.id}")

    return 0 if recall == 1.0 else 1


if __name__ == "__main__":
    _args = _parse_args()
    _dataset = json.loads(pathlib.Path(_args.dataset).read_text())
    _document = pathlib.Path(_args.document or _dataset["document"])
    if not _document.is_file():
        print(f"document not found: {_document}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(_args, _dataset, _document, _document.read_bytes())))
