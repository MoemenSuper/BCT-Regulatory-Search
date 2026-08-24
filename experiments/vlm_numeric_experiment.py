"""Bounded hosted-VLM benchmark on frozen public Arabic numeric stress pages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.numeric_fidelity_stress import critical_identifiers, critical_numbers


MODEL_ID = "qwen/qwen3.6-27b"
PROMPT_VERSION = "bct-numeric-inventory-v1"
RENDER_SCALE = 2.0
MIN_REQUEST_INTERVAL_SECONDS = 2.1
_KINDS = {"number", "date", "time", "percentage", "amount", "identifier", "other"}
_PROMPT = """Inspect this public Tunisian regulatory PDF page visually.

Return one JSON object with exactly one top-level key, \"items\". The value must be an array containing every visible answer-bearing number, date, time, percentage, amount, document reference, account/code, or alphanumeric identifier on the page.

Each item must be an object with:
- \"literal\": the exact characters as visibly printed, without normalization or correction;
- \"kind\": one of number, date, time, percentage, amount, identifier, other;
- \"context\": a short verbatim phrase around the literal, preserving Arabic reading order;
- \"uncertain\": true only when the image is genuinely unreadable.

Do not summarize. Do not infer missing digits. Do not convert Arabic-Indic digits. Do not silently fix reversed or malformed values. Return valid JSON only."""


def parse_vlm_payload(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("VLM response must be valid JSON") from error
    if not isinstance(value, dict) or set(value) != {"items"}:
        raise ValueError("VLM response must contain exactly the items key")
    if not isinstance(value["items"], list):
        raise ValueError("VLM response items must be an array")
    for index, item in enumerate(value["items"]):
        if not isinstance(item, dict):
            raise ValueError(f"VLM item {index} must be an object")
        if set(item) != {"literal", "kind", "context", "uncertain"}:
            raise ValueError(f"VLM item {index} must contain literal, kind, context, uncertain")
        if not isinstance(item["literal"], str) or not item["literal"].strip():
            raise ValueError(f"VLM item {index} literal must be non-empty text")
        if item["kind"] not in _KINDS:
            raise ValueError(f"VLM item {index} kind is invalid")
        if not isinstance(item["context"], str):
            raise ValueError(f"VLM item {index} context must be text")
        if not isinstance(item["uncertain"], bool):
            raise ValueError(f"VLM item {index} uncertain must be Boolean")
    return value


def _literal_recall(expected: set[str], actual: set[str]) -> float | None:
    return len(expected & actual) / len(expected) if expected else None


def _aggregate_case_metric(cases: list[dict[str, Any]], method: str) -> dict[str, Any]:
    numbers = [case[method]["number_recall"] for case in cases]
    identifiers = [
        case[method]["identifier_recall"]
        for case in cases
        if case[method]["identifier_recall"] is not None
    ]
    return {
        "case_count": len(cases),
        "identifier_case_count": len(identifiers),
        "mean_number_recall": statistics.mean(numbers) if numbers else 0.0,
        "full_number_recall_rate": sum(value == 1.0 for value in numbers) / len(numbers)
        if numbers
        else 0.0,
        "mean_identifier_recall": statistics.mean(identifiers) if identifiers else None,
        "full_identifier_recall_rate": (
            sum(value == 1.0 for value in identifiers) / len(identifiers)
            if identifiers
            else None
        ),
    }


def evaluate_vlm_numeric_cases(
    suite: dict[str, Any],
    pages: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    for case in suite["cases"]:
        key = (case["expected_source"].casefold(), int(case["expected_page"]))
        if key not in pages:
            raise ValueError(f"VLM output is missing {key[0]} page {key[1]}")
        page = pages[key]
        # Context is retained for audit, but only the explicit inventory literal
        # receives credit. Otherwise a missed number mentioned incidentally in
        # another item's context would inflate recall.
        text = "\n".join(item["literal"] for item in page["items"])
        expected_numbers = set(case["expected_numbers"])
        expected_identifiers = set(case["expected_identifiers"])
        actual_numbers = critical_numbers(text)
        actual_identifiers = critical_identifiers(text)
        vlm = {
            "number_recall": _literal_recall(expected_numbers, actual_numbers),
            "identifier_recall": _literal_recall(expected_identifiers, actual_identifiers),
            "matched_numbers": sorted(expected_numbers & actual_numbers),
            "missing_numbers": sorted(expected_numbers - actual_numbers),
            "matched_identifiers": sorted(expected_identifiers & actual_identifiers),
            "missing_identifiers": sorted(expected_identifiers - actual_identifiers),
        }
        cases.append(
            {
                "id": case["id"],
                "expected_source": case["expected_source"],
                "expected_page": case["expected_page"],
                "native": case["native"],
                "ocr": case["ocr"],
                "vlm": vlm,
            }
        )
    metrics = {
        method: _aggregate_case_metric(cases, method)
        for method in ("native", "ocr", "vlm")
    }
    unique_pages = list(pages.values())
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    metrics["usage"] = {
        field: sum(int(page.get("usage", {}).get(field, 0)) for page in unique_pages)
        for field in usage_fields
    }
    metrics["latency"] = {
        "page_count": len(unique_pages),
        "mean_seconds": statistics.mean(
            float(page.get("latency_seconds", 0.0)) for page in unique_pages
        )
        if unique_pages
        else 0.0,
        "median_seconds": statistics.median(
            float(page.get("latency_seconds", 0.0)) for page in unique_pages
        )
        if unique_pages
        else 0.0,
    }
    identifier_ok = (
        metrics["vlm"]["mean_identifier_recall"] is None
        or metrics["native"]["mean_identifier_recall"] is None
        or metrics["vlm"]["mean_identifier_recall"]
        >= metrics["native"]["mean_identifier_recall"] - 0.05
    )
    keep = (
        metrics["vlm"]["mean_number_recall"]
        >= metrics["native"]["mean_number_recall"] + 0.10
        and metrics["vlm"]["full_number_recall_rate"]
        >= metrics["native"]["full_number_recall_rate"]
        and identifier_ok
    )
    improvements = [
        case["id"]
        for case in cases
        if case["vlm"]["number_recall"] > case["native"]["number_recall"]
    ]
    regressions = [
        case["id"]
        for case in cases
        if case["vlm"]["number_recall"] < case["native"]["number_recall"]
    ]
    return {
        "decision_gate": {
            "minimum_mean_number_recall_delta_vs_native": 0.10,
            "minimum_full_number_recall_rate_delta_vs_native": 0.0,
            "minimum_identifier_recall_delta_vs_native": -0.05,
        },
        "metrics": metrics,
        "vlm_number_improvements": improvements,
        "vlm_number_regressions": regressions,
        "vlm_number_unchanged_count": len(cases) - len(improvements) - len(regressions),
        "decision": "KEEP_FOR_FULL_TEXT_VLM_ABLATION" if keep else "REJECT",
        "cases": cases,
    }


def _render_page(pdf_path: Path, page_number: int) -> bytes:
    with fitz.open(pdf_path) as document:
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
        seed=20260824,
        max_completion_tokens=4096,
    )
    elapsed = time.perf_counter() - started
    content = completion.choices[0].message.content or ""
    parsed = parse_vlm_payload(content)
    return {
        "source": source,
        "page": page_number,
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "image_sha256": hashlib.sha256(image).hexdigest().upper(),
        "response_id": completion.id,
        "latency_seconds": elapsed,
        "usage": _usage(completion.usage),
        **parsed,
    }


def run_vlm_numeric_experiment(
    *,
    numeric_suite_path: Path,
    structured_manifest_path: Path,
    dotenv_path: Path,
    output_dir: Path,
    confirm_public_documents: bool,
    limit: int | None = None,
) -> dict[str, Any]:
    if not confirm_public_documents:
        raise ValueError("Hosted VLM execution requires explicit public-document confirmation")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured")
    suite = json.loads(numeric_suite_path.read_text(encoding="utf-8"))
    structured_manifest = json.loads(
        structured_manifest_path.read_text(encoding="utf-8")
    )
    if sha256_file(structured_manifest_path) != suite["inputs"]["structured_manifest_sha256"]:
        raise ValueError("Structured manifest differs from the frozen numeric suite")
    records_by_source = {
        record["source"].casefold(): (relative, record)
        for relative, record in structured_manifest["records"].items()
    }
    all_pages = sorted(
        {
            (case["expected_source"], int(case["expected_page"]))
            for case in suite["cases"]
        },
        key=lambda value: (value[0].casefold(), value[1]),
    )
    selected_pages = all_pages[:limit] if limit is not None else all_pages
    client = Groq(max_retries=3, timeout=90.0)
    available = {model.id for model in client.models.list().data}
    if MODEL_ID not in available:
        raise ValueError(f"Required hosted VLM is not currently available: {MODEL_ID}")
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    page_outputs: dict[tuple[str, int], dict[str, Any]] = {}
    last_request_finished = 0.0
    for index, (source, page_number) in enumerate(selected_pages, start=1):
        relative, record = records_by_source[source.casefold()]
        pdf_path = Path(structured_manifest["documents_dir"]) / Path(relative)
        if sha256_file(pdf_path) != record["sha256"]:
            raise ValueError(f"Source PDF hash drifted: {pdf_path}")
        cache_path = cache_dir / (
            f"{record['sha256'].casefold()}-p{page_number}-{PROMPT_VERSION}.json"
        )
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            parse_vlm_payload(json.dumps({"items": payload["items"]}, ensure_ascii=False))
        else:
            remaining = MIN_REQUEST_INTERVAL_SECONDS - (time.perf_counter() - last_request_finished)
            if remaining > 0:
                time.sleep(remaining)
            image = _render_page(pdf_path, page_number)
            try:
                payload = _request_page(
                    client,
                    image=image,
                    source=source,
                    page_number=page_number,
                )
            except RateLimitError as error:
                retry_after = None
                if error.response is not None:
                    retry_after = error.response.headers.get("retry-after")
                checkpoint = {
                    "status": "rate_limited",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "provider": "Groq",
                    "model": MODEL_ID,
                    "completed_page_count": len(page_outputs),
                    "total_frozen_page_count": len(all_pages),
                    "retry_after": retry_after,
                    "cache_is_resumable": True,
                }
                write_json_atomic(output_dir / "vlm_numeric_result.json", checkpoint)
                print(
                    f"[vlm rate-limited] completed={len(page_outputs)}/{len(all_pages)}; "
                    f"retry_after={retry_after or 'provider_window'}",
                    flush=True,
                )
                return checkpoint
            last_request_finished = time.perf_counter()
            write_json_atomic(cache_path, payload)
        page_outputs[(source.casefold(), page_number)] = payload
        print(
            f"[vlm {index}/{len(selected_pages)}] {source} page {page_number} "
            f"items={len(payload['items'])}",
            flush=True,
        )

    status = "partial_smoke" if limit is not None and limit < len(all_pages) else "complete"
    artifact: dict[str, Any] = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "public_BCT_development_numeric_pages_only",
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
            "seed": 20260824,
            "max_completion_tokens": 4096,
        },
        "inputs": {
            "numeric_suite_sha256": sha256_file(numeric_suite_path),
            "structured_manifest_sha256": sha256_file(structured_manifest_path),
        },
        "selected_page_count": len(selected_pages),
        "total_frozen_page_count": len(all_pages),
    }
    if status == "complete":
        artifact.update(evaluate_vlm_numeric_cases(suite, page_outputs))
        artifact["limitations"] = [
            "Development suite only; the decision permits only a full-text cached VLM ablation.",
            "The benchmark measures recall of verified literals, not completeness or hallucinated extra values.",
            "Hosted processing is permitted here only for the current public corpus; this does not authorize confidential documents.",
            "Model availability, retention controls, rate limits, and pricing remain external operational dependencies.",
        ]
    else:
        artifact["page_smoke_outputs"] = [
            {
                "source": payload["source"],
                "page": payload["page"],
                "item_count": len(payload["items"]),
                "latency_seconds": payload["latency_seconds"],
                "usage": payload["usage"],
            }
            for payload in page_outputs.values()
        ]
    write_json_atomic(output_dir / "vlm_numeric_result.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-suite", type=Path, required=True)
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-public-documents", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    artifact = run_vlm_numeric_experiment(
        numeric_suite_path=args.numeric_suite,
        structured_manifest_path=args.structured_manifest,
        dotenv_path=args.dotenv,
        output_dir=args.output_dir,
        confirm_public_documents=args.confirm_public_documents,
        limit=args.limit,
    )
    if args.summary_output is not None and artifact.get("status") == "complete":
        write_json_atomic(args.summary_output, artifact)


if __name__ == "__main__":
    main()
