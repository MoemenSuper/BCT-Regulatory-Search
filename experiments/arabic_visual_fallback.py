"""Gold-blind routing and evidence contracts for Arabic visual verification."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from experiments.numeric_fidelity_stress import critical_numbers


MODEL_ID = "qwen/qwen3.6-27b"
PROMPT_VERSION = "bct-arabic-faithful-page-transcription-v1"
MAX_VISUAL_PAGES_PER_QUERY = 2

_ARABIC_NUMERIC_OR_DATE = re.compile(
    r"(?:متى|كم|أجل|تاريخ|مدة|ساعة|يوم|شهر|سنة|موسم|نسبة|مبلغ|سقف|فئة|"
    r"تداول|سحب|موعد|دينار|جنيه|فرنك|كرونة|%|\d)",
    re.IGNORECASE,
)
_ANSWERED_DATE_OR_QUANTITATIVE = re.compile(
    r"(?:متى|كم|أجل|تاريخ|مدة|ساعة|يوم|شهر|سنة|موسم|نسبة|مبلغ|سقف|فئة|موعد|%|\d)",
    re.IGNORECASE,
)
_EXPLICIT_CURRENT_OR_FUTURE = re.compile(
    r"(?:اليوم|غد[ًاا]?|المستقبل|سيصدر|ستصدر|سوف|حالي[ًاا]|الآن|سنة\s*2027)",
    re.IGNORECASE,
)
_KINDS = {"number", "date", "time", "percentage", "amount", "identifier", "other"}


def is_arabic_text(text: str) -> bool:
    return any(
        "\u0600" <= character <= "\u06ff"
        or "\u0750" <= character <= "\u077f"
        or "\u08a0" <= character <= "\u08ff"
        for character in text
    )


def is_numeric_or_date_query(query: str) -> bool:
    return is_arabic_text(query) and bool(_ARABIC_NUMERIC_OR_DATE.search(query))


def is_explicit_current_or_future_query(query: str) -> bool:
    return bool(_EXPLICIT_CURRENT_OR_FUTURE.search(query))


def _page_key(source: str, page: int) -> tuple[str, int]:
    return Path(source).name.casefold(), int(page)


def route_visual_pages(
    *,
    suite: dict[str, Any],
    retrieved_result: dict[str, Any],
    risk_result: dict[str, Any],
    max_pages_per_query: int = MAX_VISUAL_PAGES_PER_QUERY,
) -> list[dict[str, Any]]:
    """Select visual pages using query, response, citation, retrieval, and risk data only."""
    if max_pages_per_query < 1:
        raise ValueError("max_pages_per_query must be positive")
    runtime_cases = {
        case["id"]: {"query": case["query"], "language": case["language"]}
        for case in suite["cases"]
    }
    records = {record["id"]: record for record in retrieved_result["records"]}
    if set(runtime_cases) != set(records):
        raise ValueError("Suite and retrieved result IDs differ")
    risky_sources = {
        Path(record["source"]).name.casefold()
        for record in risk_result["records"]
        if record.get("requires_visual_fallback")
    }
    routes = []
    for case_id, runtime_case in runtime_cases.items():
        query = runtime_case["query"]
        record = records[case_id]
        status = record["response"]["status"]
        if (
            runtime_case["language"] != "ar"
            or not is_numeric_or_date_query(query)
            or is_explicit_current_or_future_query(query)
            or status not in {"answered", "insufficient_evidence"}
        ):
            continue
        evidence_by_id = {
            evidence["evidence_id"]: evidence
            for evidence in record["retrieved_evidence"]
        }
        if status == "answered":
            response_text = "\n".join(
                [
                    str(record["response"].get("answer", "")),
                    *(str(claim.get("text", "")) for claim in record["response"].get("claims", [])),
                ]
            )
            if not critical_numbers(response_text) and not _ANSWERED_DATE_OR_QUANTITATIVE.search(query):
                continue
            candidates = [
                evidence_by_id[citation["evidence_id"]]
                for citation in record["response"]["citations"]
                if citation["evidence_id"] in evidence_by_id
            ]
            reason = "answered_numeric_or_date_query_cited_risky_document"
        else:
            candidates = record["retrieved_evidence"]
            reason = "insufficient_numeric_or_date_query_ranked_risky_evidence"
        selected = []
        seen = set()
        for evidence in candidates:
            key = _page_key(evidence["source"], evidence["page"])
            if key[0] not in risky_sources or key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "evidence_id": evidence["evidence_id"],
                    "source": Path(evidence["source"]).name,
                    "page": int(evidence["page"]),
                    "reason": reason,
                }
            )
            if len(selected) == max_pages_per_query:
                break
        if selected:
            routes.append(
                {
                    "id": case_id,
                    "language": runtime_case["language"],
                    "query": query,
                    "original_status": status,
                    "pages": selected,
                }
            )
    return routes


def parse_visual_payload(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Visual response must be valid JSON") from error
    required = {"transcription", "items", "uncertain_regions", "complete"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Visual response has an invalid top-level contract")
    if not isinstance(value["transcription"], str) or not value["transcription"].strip():
        raise ValueError("Visual transcription must be non-empty")
    if not isinstance(value["items"], list):
        raise ValueError("Visual items must be an array")
    if not isinstance(value["uncertain_regions"], list) or not all(
        isinstance(item, str) for item in value["uncertain_regions"]
    ):
        raise ValueError("Visual uncertain_regions must be a text array")
    if not isinstance(value["complete"], bool):
        raise ValueError("Visual complete must be Boolean")
    for index, item in enumerate(value["items"]):
        if not isinstance(item, dict) or set(item) != {
            "literal",
            "kind",
            "context",
            "uncertain",
        }:
            raise ValueError(f"Visual item {index} has an invalid contract")
        if not isinstance(item["literal"], str) or not item["literal"].strip():
            raise ValueError(f"Visual item {index} literal must be non-empty")
        if item["kind"] not in _KINDS:
            raise ValueError(f"Visual item {index} kind is invalid")
        if not isinstance(item["context"], str):
            raise ValueError(f"Visual item {index} context must be text")
        if not isinstance(item["uncertain"], bool):
            raise ValueError(f"Visual item {index} uncertain must be Boolean")
        if item["literal"] not in value["transcription"]:
            raise ValueError(f"Visual item {index} literal is absent from transcription")
    return value


def visual_payload_is_usable(payload: dict[str, Any]) -> bool:
    return (
        payload["complete"]
        and not payload["uncertain_regions"]
        and all(not item["uncertain"] for item in payload["items"])
    )


def validate_visual_cache_binding(
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
    parse_visual_payload(json.dumps(payload["response"], ensure_ascii=False))


def visual_evidence(
    native_evidence: dict[str, Any], visual_cache: dict[str, Any]
) -> dict[str, Any] | None:
    payload = visual_cache["response"]
    if not visual_payload_is_usable(payload):
        return None
    native_text = str(native_evidence["text"])
    visual_text = payload["transcription"]
    native_numbers = critical_numbers(native_text)
    visual_numbers = critical_numbers(visual_text)
    return {
        **native_evidence,
        "text": visual_text,
        "representations": sorted(
            set(native_evidence.get("representations", []))
            | {"vlm_visual_transcription"}
        ),
        "visual_verification": {
            "model": visual_cache["model"],
            "prompt_version": visual_cache["prompt_version"],
            "source_pdf_sha256": visual_cache["source_pdf_sha256"],
            "image_sha256": visual_cache["image_sha256"],
            "native_text_sha256": hashlib.sha256(
                native_text.encode("utf-8")
            ).hexdigest().upper(),
            "numeric_conflict": native_numbers != visual_numbers,
            "native_only_numbers": sorted(native_numbers - visual_numbers),
            "visual_only_numbers": sorted(visual_numbers - native_numbers),
        },
    }
