"""Gold-blind routing and evidence contracts for Arabic visual verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic
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


def build_routing_receipt(
    *,
    suite: dict[str, Any],
    retrieved_result: dict[str, Any],
    risk_result: dict[str, Any],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    routes = route_visual_pages(
        suite=suite,
        retrieved_result=retrieved_result,
        risk_result=risk_result,
    )
    return {
        "status": "frozen_before_visual_calls",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "arabic-query-time-visual-routing-development-v1",
        "inputs": input_hashes,
        "policy": {
            "language": "Arabic query text only",
            "intent": "runtime lexical numeric or date signal",
            "answered": "verify only cited pages from risky documents",
            "insufficient_evidence": "verify at most the first two reranked pages from risky documents",
            "excluded_statuses": ["clarification_needed", "out_of_scope"],
            "excluded_temporal_queries": "explicit current or future lexical requests",
            "max_visual_pages_per_query": MAX_VISUAL_PAGES_PER_QUERY,
            "gold_fields_used_for_routing": [],
        },
        "component_gate": {
            "minimum_known_extraction_repairs_or_safe_corrections": 3,
            "maximum_new_wrong_numeric_or_date_assertions": 0,
            "maximum_routed_control_strict_answer_or_citation_regressions": 0,
            "all_affirmative_changed_claims_require_valid_source_page_links": True,
            "malformed_uncertain_or_unavailable_visual_output": "fail_closed",
            "validation_access_on_component_pass": False,
        },
        "counts": {
            "routed_cases": len(routes),
            "routed_pages": sum(len(route["pages"]) for route in routes),
            "unique_pages": len(
                {
                    _page_key(page["source"], page["page"])
                    for route in routes
                    for page in route["pages"]
                }
            ),
        },
        "routes": routes,
        "limitations": [
            "Development-only routing receipt; gold fields may be used only after this artifact is frozen for scoring.",
            "A routed page is a risk candidate, not proof that native extraction is wrong.",
            "Passing the component gate permits only full development recomposition, not validation or production use.",
        ],
    }


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
            "provider": visual_cache.get("provider", "Groq"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--retrieved-result", type=Path, required=True)
    parser.add_argument("--risk-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    retrieved = json.loads(args.retrieved_result.read_text(encoding="utf-8"))
    risk = json.loads(args.risk_result.read_text(encoding="utf-8"))
    receipt = build_routing_receipt(
        suite=suite,
        retrieved_result=retrieved,
        risk_result=risk,
        input_hashes={
            "answer_suite_sha256": sha256_file(args.suite),
            "retrieved_result_sha256": sha256_file(args.retrieved_result),
            "risk_result_sha256": sha256_file(args.risk_result),
        },
    )
    write_json_atomic(args.output, receipt)


if __name__ == "__main__":
    main()
