"""Run Gemini 3.7 Flash on the frozen Arabic visual-fallback pages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from experiments.arabic_visual_fallback import (
    MAX_VISUAL_PAGES_PER_QUERY,
    parse_visual_payload,
    visual_payload_is_usable,
)
from experiments.arabic_visual_transcription_experiment import (
    RENDER_SCALE,
    _PROMPT,
    _render_page,
    frozen_routed_pages,
)
from experiments.artifacts import sha256_file, write_json_atomic


PROVIDER = "Google Gemini API"
MODEL_ID = "gemini-3.7-flash"
PROMPT_VERSION = "bct-arabic-faithful-page-transcription-v1-gemini-3.7"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
INPUT_USD_PER_MILLION = 0.75
OUTPUT_USD_PER_MILLION = 3.75
MAX_OUTPUT_TOKENS = 8192

_VISUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "transcription": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "literal": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "number",
                            "date",
                            "time",
                            "percentage",
                            "amount",
                            "identifier",
                            "other",
                        ],
                    },
                    "context": {"type": "string"},
                    "uncertain": {"type": "boolean"},
                },
                "required": ["literal", "kind", "context", "uncertain"],
            },
        },
        "uncertain_regions": {"type": "array", "items": {"type": "string"}},
        "complete": {"type": "boolean"},
    },
    "required": ["transcription", "items", "uncertain_regions", "complete"],
}


def _visual_configuration() -> dict[str, Any]:
    return {
        "provider": PROVIDER,
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "render_scale": RENDER_SCALE,
        "response_format": "JSON_Schema_plus_strict_local_validation",
        "response_schema": _VISUAL_SCHEMA,
        "thinking_level": "low",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_visual_pages_per_query": MAX_VISUAL_PAGES_PER_QUERY,
    }


def visual_configuration_sha256() -> str:
    encoded = json.dumps(
        _visual_configuration(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def visual_cache_key(
    *,
    source_pdf_sha256: str,
    page: int,
    image_sha256: str,
    configuration_sha256: str,
) -> str:
    binding = {
        "source_pdf_sha256": source_pdf_sha256.upper(),
        "page": int(page),
        "image_sha256": image_sha256.upper(),
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "configuration_sha256": configuration_sha256.upper(),
    }
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().casefold()


def _usage(metadata: dict[str, Any]) -> dict[str, int]:
    return {
        "prompt_tokens": int(metadata.get("promptTokenCount", 0) or 0),
        "completion_tokens": int(metadata.get("candidatesTokenCount", 0) or 0),
        "thinking_tokens": int(metadata.get("thoughtsTokenCount", 0) or 0),
        "total_tokens": int(metadata.get("totalTokenCount", 0) or 0),
    }


def build_generate_content_payload(image: bytes) -> dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": _PROMPT},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(image).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "thinkingConfig": {"thinkingLevel": "low"},
            "responseFormat": {
                "text": {
                    "mimeType": "APPLICATION_JSON",
                    "schema": _VISUAL_SCHEMA,
                }
            },
        },
    }


def extract_generate_content_response(body: dict[str, Any]) -> dict[str, Any]:
    text_parts = [
        part["text"]
        for candidate in body.get("candidates", [])
        for part in candidate.get("content", {}).get("parts", [])
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    if not text_parts:
        raise ValueError("Gemini response contains no text part")
    response = parse_visual_payload("".join(text_parts))
    return {
        "response_id": body.get("responseId"),
        "usage": _usage(body.get("usageMetadata", {})),
        "response": response,
    }


def _api_json(
    *,
    method: str,
    path: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 180.0,
) -> tuple[dict[str, Any], dict[str, str]]:
    request = urllib.request.Request(
        f"{API_ROOT}/{path}",
        data=(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        ),
        method=method,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                json.loads(response.read().decode("utf-8")),
                {key.casefold(): value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise GeminiAPIError(
            status=error.code,
            detail=detail,
            retry_after=error.headers.get("Retry-After"),
        ) from error


class GeminiAPIError(RuntimeError):
    def __init__(self, *, status: int, detail: str, retry_after: str | None) -> None:
        super().__init__(f"Gemini API returned HTTP {status}: {detail}")
        self.status = status
        self.retry_after = retry_after


def _request_page(
    *,
    api_key: str,
    image: bytes,
    source: str,
    page_number: int,
    source_pdf_sha256: str,
    configuration_sha256: str,
) -> dict[str, Any]:
    payload = build_generate_content_payload(image)
    started = time.perf_counter()
    body, _headers = _api_json(
        method="POST",
        path=f"models/{MODEL_ID}:generateContent",
        api_key=api_key,
        payload=payload,
    )
    elapsed = time.perf_counter() - started
    common = {
        "provider": PROVIDER,
        "source": source,
        "page": page_number,
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "source_pdf_sha256": source_pdf_sha256,
        "image_sha256": hashlib.sha256(image).hexdigest().upper(),
        "configuration_sha256": configuration_sha256,
        "latency_seconds": elapsed,
    }
    try:
        parsed = extract_generate_content_response(body)
    except ValueError as error:
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True)
        return {
            **common,
            "validation_status": "invalid",
            "validation_error": str(error),
            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest().upper(),
            "response_id": body.get("responseId"),
            "usage": _usage(body.get("usageMetadata", {})),
        }
    return {**common, "validation_status": "valid", **parsed}


def validate_cached_page(
    payload: dict[str, Any],
    *,
    source_pdf_sha256: str,
    page: int,
    image_sha256: str,
    configuration_sha256: str | None = None,
) -> None:
    expected_configuration = configuration_sha256 or visual_configuration_sha256()
    expected = {
        "provider": PROVIDER,
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "source_pdf_sha256": source_pdf_sha256.upper(),
        "page": int(page),
        "image_sha256": image_sha256.upper(),
        "configuration_sha256": expected_configuration.upper(),
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise ValueError(f"Gemini visual cache binding differs: {mismatches}")
    status = payload.get("validation_status")
    if status == "valid":
        parse_visual_payload(json.dumps(payload["response"], ensure_ascii=False))
    elif status == "invalid":
        if not payload.get("validation_error") or not payload.get("raw_response_sha256"):
            raise ValueError("Invalid Gemini visual cache lacks its audit receipt")
    else:
        raise ValueError("Gemini visual cache validation status is invalid")


def _artifact(
    *,
    status: str,
    routing_receipt_path: Path,
    structured_manifest_path: Path,
    frozen_page_count: int,
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [page for page in completed if page["validation_status"] == "valid"]
    usable = [page for page in valid if visual_payload_is_usable(page["response"])]
    usage = {
        field: sum(int(page.get("usage", {}).get(field, 0)) for page in completed)
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "thinking_tokens",
            "total_tokens",
        )
    }
    billable_output = usage["completion_tokens"] + usage["thinking_tokens"]
    estimated_cost = (
        usage["prompt_tokens"] * INPUT_USD_PER_MILLION
        + billable_output * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "public_BCT_same_frozen_gold_blind_routed_Arabic_pages_only",
        "external_processing": {
            "provider": PROVIDER,
            "model": MODEL_ID,
            "document_classification": "public",
            "secret_values_persisted": False,
        },
        "configuration": {
            "prompt_version": PROMPT_VERSION,
            "render_scale": RENDER_SCALE,
            "response_format": "JSON_Schema_plus_strict_local_validation",
            "thinking_level": "low",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_visual_pages_per_query": MAX_VISUAL_PAGES_PER_QUERY,
            "configuration_sha256": visual_configuration_sha256(),
        },
        "inputs": {
            "routing_receipt_sha256": sha256_file(routing_receipt_path),
            "structured_manifest_sha256": sha256_file(structured_manifest_path),
        },
        "counts": {
            "frozen_pages": frozen_page_count,
            "completed_pages": len(completed),
            "valid_pages": len(valid),
            "usable_pages": len(usable),
            "invalid_or_uncertain_pages": len(completed) - len(usable),
            "hosted_requests_this_run": sum(not page["cache_hit"] for page in completed),
            "cache_hits_this_run": sum(page["cache_hit"] for page in completed),
            "reused_artifact_hits_this_run": sum(
                page.get("cache_origin") == "reused_artifact" for page in completed
            ),
            "unique_hosted_requests_for_cached_result": len(completed),
        },
        "usage": usage,
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
        "approximate_cost": {
            "status": "estimated_paid_standard_introductory_rate",
            "currency": "USD",
            "amount": estimated_cost,
            "input_usd_per_million": INPUT_USD_PER_MILLION,
            "output_including_thinking_usd_per_million": OUTPUT_USD_PER_MILLION,
            "free_tier_actual_charge_may_be_zero": True,
        },
        "pages": completed,
    }


def run_gemini_visual_transcription(
    *,
    routing_receipt_path: Path,
    structured_manifest_path: Path,
    dotenv_path: Path,
    output_dir: Path,
    confirm_public_documents: bool,
    reuse_visual_result_paths: tuple[Path, ...] = (),
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
    load_dotenv(dotenv_path=dotenv_path, override=False)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")
    model, _headers = _api_json(
        method="GET", path=f"models/{MODEL_ID}", api_key=api_key
    )
    if model.get("name") != f"models/{MODEL_ID}":
        raise ValueError(f"Required Gemini model is not currently available: {MODEL_ID}")

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    configuration_sha256 = visual_configuration_sha256()
    reusable_pages: dict[tuple[str, int], dict[str, Any]] = {}
    expected_reuse_configuration = {
        key: value
        for key, value in _visual_configuration().items()
        if key not in {"provider", "model", "response_schema"}
    }
    for reuse_path in reuse_visual_result_paths:
        reusable = json.loads(reuse_path.read_text(encoding="utf-8"))
        if reusable.get("status") != "complete":
            raise ValueError(f"Reusable visual result is incomplete: {reuse_path}")
        actual_configuration = reusable.get("configuration", {})
        mismatches = [
            key
            for key, expected in expected_reuse_configuration.items()
            if actual_configuration.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                f"Reusable visual result configuration differs: {mismatches}"
            )
        for reusable_page in reusable.get("pages", []):
            key = (str(reusable_page["source"]).casefold(), int(reusable_page["page"]))
            reusable_pages.setdefault(key, reusable_page)
    completed: list[dict[str, Any]] = []
    unavailable_pages: list[dict[str, Any]] = []
    for index, page in enumerate(pages, start=1):
        source = page["source"]
        page_number = page["page"]
        if source.casefold() not in records_by_source:
            raise ValueError(f"Routed source is absent from manifest: {source}")
        relative, manifest_record = records_by_source[source.casefold()]
        pdf_path = Path(manifest["documents_dir"]) / Path(relative)
        pdf_sha = sha256_file(pdf_path)
        if pdf_sha != manifest_record["sha256"]:
            raise ValueError(f"Source PDF hash drifted: {pdf_path}")
        image = _render_page(pdf_path, page_number)
        image_sha = hashlib.sha256(image).hexdigest().upper()
        cache_path = cache_dir / (
            visual_cache_key(
                source_pdf_sha256=pdf_sha,
                page=page_number,
                image_sha256=image_sha,
                configuration_sha256=configuration_sha256,
            )
            + ".json"
        )
        cache_hit = cache_path.exists()
        cache_origin = "local_cache" if cache_hit else "hosted"
        if cache_hit:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            validate_cached_page(
                payload,
                source_pdf_sha256=pdf_sha,
                page=page_number,
                image_sha256=image_sha,
                configuration_sha256=configuration_sha256,
            )
        elif (source.casefold(), page_number) in reusable_pages:
            payload = {
                **reusable_pages[(source.casefold(), page_number)],
                "configuration_sha256": configuration_sha256,
            }
            payload.pop("cache_hit", None)
            payload.pop("cache_origin", None)
            validate_cached_page(
                payload,
                source_pdf_sha256=pdf_sha,
                page=page_number,
                image_sha256=image_sha,
                configuration_sha256=configuration_sha256,
            )
            write_json_atomic(cache_path, payload)
            cache_hit = True
            cache_origin = "reused_artifact"
        else:
            try:
                payload = _request_page(
                    api_key=api_key,
                    image=image,
                    source=source,
                    page_number=page_number,
                    source_pdf_sha256=pdf_sha,
                    configuration_sha256=configuration_sha256,
                )
            except GeminiAPIError as error:
                if error.status == 503:
                    unavailable_pages.append(
                        {
                            "source": source,
                            "page": page_number,
                            "status": "provider_temporarily_unavailable",
                        }
                    )
                    print(
                        f"[gemini-visual unavailable] {source} page {page_number} HTTP 503",
                        flush=True,
                    )
                    continue
                if error.status != 429:
                    raise
                checkpoint = _artifact(
                    status="rate_limited",
                    routing_receipt_path=routing_receipt_path,
                    structured_manifest_path=structured_manifest_path,
                    frozen_page_count=len(pages),
                    completed=completed,
                )
                checkpoint["retry_after"] = error.retry_after
                checkpoint["cache_is_resumable"] = True
                write_json_atomic(
                    output_dir / "gemini_visual_transcription_v1.json", checkpoint
                )
                return checkpoint
            write_json_atomic(cache_path, payload)
        completed.append(
            {**payload, "cache_hit": cache_hit, "cache_origin": cache_origin}
        )
        print(
            f"[gemini-visual {index}/{len(pages)}] {source} page {page_number} "
            f"validation={payload['validation_status']} cache_hit={cache_hit}",
            flush=True,
        )

    artifact = _artifact(
        status="partial_provider_unavailable" if unavailable_pages else "complete",
        routing_receipt_path=routing_receipt_path,
        structured_manifest_path=structured_manifest_path,
        frozen_page_count=len(pages),
        completed=completed,
    )
    artifact["unavailable_pages"] = unavailable_pages
    artifact["cache_is_resumable"] = bool(unavailable_pages)
    write_json_atomic(output_dir / "gemini_visual_transcription_v1.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-receipt", type=Path, required=True)
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-public-documents", action="store_true")
    parser.add_argument(
        "--reuse-visual-result", type=Path, action="append", default=[]
    )
    args = parser.parse_args()
    run_gemini_visual_transcription(
        routing_receipt_path=args.routing_receipt,
        structured_manifest_path=args.structured_manifest,
        dotenv_path=args.dotenv,
        output_dir=args.output_dir,
        confirm_public_documents=args.confirm_public_documents,
        reuse_visual_result_paths=tuple(args.reuse_visual_result),
    )


if __name__ == "__main__":
    main()
