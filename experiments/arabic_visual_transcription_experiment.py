"""Run the frozen query-time Arabic visual-transcription page set."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from experiments.arabic_visual_fallback import (
    MAX_VISUAL_PAGES_PER_QUERY,
    MODEL_ID,
    PROMPT_VERSION,
    parse_visual_payload,
    validate_visual_cache_binding,
    visual_payload_is_usable,
)
from experiments.artifacts import sha256_file, write_json_atomic


RENDER_SCALE = 2.0
MIN_REQUEST_INTERVAL_SECONDS = 2.1
SEED = 20260825
MAX_COMPLETION_TOKENS = 4096
_PROMPT = """Inspect this public Tunisian regulatory PDF page visually.

Return one JSON object with exactly these top-level keys:
- "transcription": a faithful verbatim transcription of every visible region containing an answer-bearing number, date, time, percentage, amount, document reference, account/code, or alphanumeric identifier, including enough immediately adjacent text to interpret each literal;
- "items": an array of every such literal, with objects containing exactly "literal", "kind", "context", and "uncertain";
- "uncertain_regions": an array describing any relevant region that cannot be read confidently;
- "complete": true only if every relevant region on the page was inspected, included, and is legible.

For each item:
- "literal" must be the exact visibly printed characters and must occur verbatim in "transcription";
- "kind" must be one of number, date, time, percentage, amount, identifier, other;
- "context" must be a short verbatim phrase around the literal;
- "uncertain" must be true when the literal or its association is not visually certain.

Preserve Arabic reading order, visible digit order, punctuation, and line breaks. Do not summarize, interpret, normalize, reverse, translate, or silently correct anything. Do not infer missing digits. Do not convert Arabic-Indic digits. Return valid JSON only."""


def frozen_routed_pages(routing_receipt: dict[str, Any]) -> list[dict[str, Any]]:
    if routing_receipt.get("status") != "frozen_before_visual_calls":
        raise ValueError("Routing receipt is not frozen before visual calls")
    if routing_receipt.get("policy", {}).get("gold_fields_used_for_routing") != []:
        raise ValueError("Routing receipt is not gold blind")
    max_pages = int(
        routing_receipt.get("policy", {}).get("max_visual_pages_per_query", 0)
    )
    if max_pages != MAX_VISUAL_PAGES_PER_QUERY:
        raise ValueError("Routing page budget differs from the implementation")
    pages: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for route in routing_receipt.get("routes", []):
        route_pages = route.get("pages", [])
        if len(route_pages) > max_pages:
            raise ValueError(f"Route exceeds visual page budget: {route.get('id')}")
        for page in route_pages:
            key = (Path(page["source"]).name.casefold(), int(page["page"]))
            if key in seen:
                continue
            seen.add(key)
            pages.append({"source": Path(page["source"]).name, "page": int(page["page"])})
    counts = routing_receipt.get("counts", {})
    if len(pages) != int(counts.get("unique_pages", -1)):
        raise ValueError("Routing unique-page count differs from its routes")
    return pages


def _render_page(pdf_path: Path, page_number: int) -> bytes:
    with fitz.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(f"Page {page_number} is outside {pdf_path.name}")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE), alpha=False
        )
        return pixmap.tobytes("png")


def _usage(value: Any) -> dict[str, int]:
    return {
        field: int(getattr(value, field, 0) or 0)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _request_page(
    client: Groq,
    *,
    image: bytes,
    source: str,
    page_number: int,
    source_pdf_sha256: str,
) -> dict[str, Any]:
    encoded = base64.b64encode(image).decode("ascii")
    started = time.perf_counter()
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        reasoning_effort="none",
        temperature=0,
        seed=SEED,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    elapsed = time.perf_counter() - started
    content = completion.choices[0].message.content or ""
    common = {
        "source": source,
        "page": page_number,
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "source_pdf_sha256": source_pdf_sha256,
        "image_sha256": hashlib.sha256(image).hexdigest().upper(),
        "response_id": completion.id,
        "latency_seconds": elapsed,
        "usage": _usage(completion.usage),
    }
    try:
        response = parse_visual_payload(content)
    except ValueError as error:
        return {
            **common,
            "validation_status": "invalid",
            "validation_error": str(error),
            "raw_response_sha256": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest().upper(),
        }
    return {**common, "validation_status": "valid", "response": response}


def validate_cached_page(
    payload: dict[str, Any],
    *,
    source_pdf_sha256: str,
    page: int,
    image_sha256: str,
) -> None:
    expected = {
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "source_pdf_sha256": source_pdf_sha256.upper(),
        "page": int(page),
        "image_sha256": image_sha256.upper(),
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(f"Visual cache binding differs: {mismatches}")
    status = payload.get("validation_status")
    if status == "valid":
        validate_visual_cache_binding(
            payload,
            source_pdf_sha256=source_pdf_sha256,
            page=page,
            image_sha256=image_sha256,
        )
    elif status == "invalid":
        if not payload.get("validation_error") or not payload.get("raw_response_sha256"):
            raise ValueError("Invalid visual cache lacks its audit receipt")
    else:
        raise ValueError("Visual cache validation status is invalid")


def run_visual_transcription(
    *,
    routing_receipt_path: Path,
    structured_manifest_path: Path,
    dotenv_path: Path,
    output_dir: Path,
    confirm_public_documents: bool,
) -> dict[str, Any]:
    if not confirm_public_documents:
        raise ValueError("Hosted visual execution requires public-document confirmation")
    routing = json.loads(routing_receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(structured_manifest_path.read_text(encoding="utf-8"))
    pages = frozen_routed_pages(routing)
    records_by_source = {
        record["source"].casefold(): (relative, record)
        for relative, record in manifest["records"].items()
    }
    for page in pages:
        if page["source"].casefold() not in records_by_source:
            raise ValueError(f"Routed source is absent from manifest: {page['source']}")

    load_dotenv(dotenv_path=dotenv_path, override=False)
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured")
    client = Groq(max_retries=3, timeout=120.0)
    available = {model.id for model in client.models.list().data}
    if MODEL_ID not in available:
        raise ValueError(f"Required hosted VLM is not currently available: {MODEL_ID}")

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    last_request_finished = 0.0
    for index, page in enumerate(pages, start=1):
        source = page["source"]
        page_number = page["page"]
        relative, manifest_record = records_by_source[source.casefold()]
        pdf_path = Path(manifest["documents_dir"]) / Path(relative)
        pdf_sha = sha256_file(pdf_path)
        if pdf_sha != manifest_record["sha256"]:
            raise ValueError(f"Source PDF hash drifted: {pdf_path}")
        image = _render_page(pdf_path, page_number)
        image_sha = hashlib.sha256(image).hexdigest().upper()
        cache_path = cache_dir / (
            f"{pdf_sha.casefold()}-p{page_number}-{MODEL_ID.replace('/', '_')}-"
            f"{PROMPT_VERSION}.json"
        )
        cache_hit = cache_path.exists()
        if cache_hit:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            validate_cached_page(
                payload,
                source_pdf_sha256=pdf_sha,
                page=page_number,
                image_sha256=image_sha,
            )
        else:
            remaining = MIN_REQUEST_INTERVAL_SECONDS - (
                time.perf_counter() - last_request_finished
            )
            if remaining > 0:
                time.sleep(remaining)
            try:
                payload = _request_page(
                    client,
                    image=image,
                    source=source,
                    page_number=page_number,
                    source_pdf_sha256=pdf_sha,
                )
            except RateLimitError as error:
                retry_after = None
                if error.response is not None:
                    retry_after = error.response.headers.get("retry-after")
                checkpoint = _build_artifact(
                    status="rate_limited",
                    routing_receipt_path=routing_receipt_path,
                    structured_manifest_path=structured_manifest_path,
                    pages=pages,
                    completed=completed,
                )
                checkpoint["retry_after"] = retry_after
                checkpoint["cache_is_resumable"] = True
                write_json_atomic(output_dir / "arabic_visual_transcription_v1.json", checkpoint)
                return checkpoint
            last_request_finished = time.perf_counter()
            write_json_atomic(cache_path, payload)
        completed.append({**payload, "cache_hit": cache_hit})
        print(
            f"[visual {index}/{len(pages)}] {source} page {page_number} "
            f"validation={payload['validation_status']} cache_hit={cache_hit}",
            flush=True,
        )

    artifact = _build_artifact(
        status="complete",
        routing_receipt_path=routing_receipt_path,
        structured_manifest_path=structured_manifest_path,
        pages=pages,
        completed=completed,
    )
    write_json_atomic(output_dir / "arabic_visual_transcription_v1.json", artifact)
    return artifact


def _build_artifact(
    *,
    status: str,
    routing_receipt_path: Path,
    structured_manifest_path: Path,
    pages: list[dict[str, Any]],
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    valid = [page for page in completed if page["validation_status"] == "valid"]
    usable = [page for page in valid if visual_payload_is_usable(page["response"])]
    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "public_BCT_gold_blind_routed_Arabic_pages_only",
        "external_processing": {
            "provider": "Groq",
            "model": MODEL_ID,
            "document_classification": "public",
            "secret_values_persisted": False,
        },
        "configuration": {
            "prompt_version": PROMPT_VERSION,
            "render_scale": RENDER_SCALE,
            "response_format": "json_object_locally_validated",
            "reasoning_effort": "none",
            "temperature": 0,
            "seed": SEED,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "max_visual_pages_per_query": MAX_VISUAL_PAGES_PER_QUERY,
        },
        "inputs": {
            "routing_receipt_sha256": sha256_file(routing_receipt_path),
            "structured_manifest_sha256": sha256_file(structured_manifest_path),
        },
        "counts": {
            "frozen_pages": len(pages),
            "completed_pages": len(completed),
            "valid_pages": len(valid),
            "usable_pages": len(usable),
            "invalid_or_uncertain_pages": len(completed) - len(usable),
            "hosted_requests_this_run": sum(not page["cache_hit"] for page in completed),
            "cache_hits_this_run": sum(page["cache_hit"] for page in completed),
        },
        "usage": {
            field: sum(int(page.get("usage", {}).get(field, 0)) for page in completed)
            for field in usage_fields
        },
        "latency_seconds": {
            "hosted_total": sum(
                float(page.get("latency_seconds", 0.0))
                for page in completed
                if not page["cache_hit"]
            ),
            "recorded_all_pages": sum(
                float(page.get("latency_seconds", 0.0)) for page in completed
            ),
        },
        "pages": completed,
        "limitations": [
            "Development-only evidence-ingestion experiment on the frozen routed pages.",
            "A valid visual transcription is not automatically a correct legal answer.",
            "Malformed, incomplete, or uncertain visual output is unusable and must fail closed downstream.",
            "Model availability, retention controls, rate limits, and pricing remain external operational dependencies.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-receipt", type=Path, required=True)
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-public-documents", action="store_true")
    args = parser.parse_args()
    run_visual_transcription(
        routing_receipt_path=args.routing_receipt,
        structured_manifest_path=args.structured_manifest,
        dotenv_path=args.dotenv,
        output_dir=args.output_dir,
        confirm_public_documents=args.confirm_public_documents,
    )


if __name__ == "__main__":
    main()
