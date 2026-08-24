"""Compose cached answer components under a selective context-retry policy."""

from __future__ import annotations

import argparse
import json
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.structured_answer_experiment import _aggregate


def _deduplicate_citations(record: dict[str, Any]) -> None:
    citations = record["response"]["citations"]
    seen = set()
    unique = []
    for citation in citations:
        key = (
            citation["evidence_id"],
            citation["source"].casefold(),
            int(citation["page"]),
        )
        if key not in seen:
            seen.add(key)
            unique.append(citation)
    record["response"]["citations"] = unique
    if isinstance(record.get("structured_diagnostics"), dict):
        record["structured_diagnostics"]["citation_count"] = len(unique)


def compose_selective_answer_candidate(
    *,
    suite: dict[str, Any],
    base_result: dict[str, Any],
    retry_suite: dict[str, Any],
    retry_result: dict[str, Any],
    negative_result: dict[str, Any],
) -> dict[str, Any]:
    suite_ids = [case["id"] for case in suite["cases"]]
    base_by_id = {record["id"]: record for record in base_result["records"]}
    retry_ids = {case["id"] for case in retry_suite["cases"]}
    retry_by_id = {record["id"]: record for record in retry_result["records"]}
    negative_by_id = {record["id"]: record for record in negative_result["records"]}
    expected_negative_ids = {
        case["id"] for case in suite["cases"] if not case.get("relevant")
    }
    if set(base_by_id) != set(suite_ids):
        raise ValueError("Base answer result must exactly cover the full suite")
    if set(retry_by_id) != retry_ids:
        raise ValueError("Retry result and retry suite IDs differ")
    if set(negative_by_id) != expected_negative_ids:
        raise ValueError("Query-state result must exactly cover negative suite IDs")

    records = []
    retried_ids = []
    for case in suite["cases"]:
        case_id = case["id"]
        if not case.get("relevant"):
            selected = deepcopy(negative_by_id[case_id])
            selected["answer_path"] = "query_state_then_deterministic_status"
        else:
            base = base_by_id[case_id]
            base_status = base["response"]["status"]
            if base_status == "answered":
                if case_id in retry_ids:
                    raise ValueError(f"Answered base case was unexpectedly selected for retry: {case_id}")
                selected = deepcopy(base)
                selected["answer_path"] = "verified_excerpt_claim_linked_answer"
            elif base_status == "insufficient_evidence":
                if case_id not in retry_ids:
                    raise ValueError(f"Insufficient-evidence case lacks a frozen retry: {case_id}")
                selected = deepcopy(retry_by_id[case_id])
                selected["answer_path"] = "single_full_page_context_retry"
                retried_ids.append(case_id)
            else:
                raise ValueError(
                    f"Relevant base case has unsupported status {base_status}: {case_id}"
                )
        _deduplicate_citations(selected)
        records.append(selected)

    latency = [float(record["latency_seconds"]) for record in records]
    return {
        "status": "complete",
        "experiment_id": "selective-context-query-state-answer-development-v1",
        "configuration": {
            "relevant_default": "verified excerpt with claim-linked answer v1",
            "relevant_retry": (
                "one full labeled-page retry only after insufficient_evidence"
            ),
            "negative_path": "explicit query state plus deterministic status and text",
            "model": "openai/gpt-oss-120b",
            "citation_normalization": "exact duplicate citations removed locally",
        },
        "automatic_metrics": {
            "overall": _aggregate(records),
            "fr": _aggregate(
                [record for record in records if record["language"] == "fr"]
            ),
            "ar": _aggregate(
                [record for record in records if record["language"] == "ar"]
            ),
        },
        "usage": {
            field: sum(int(record["usage"][field]) for record in records)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "latency_seconds": {
            "mean_selected_call": statistics.mean(latency),
            "median_selected_call": statistics.median(latency),
            "note": "Retry cases also incurred the base insufficient-evidence call before the selected retry call.",
        },
        "policy_diagnostics": {
            "retried_case_ids": retried_ids,
            "retried_case_count": len(retried_ids),
            "negative_query_state_case_count": len(expected_negative_ids),
        },
        "limitations": [
            "Development-only cached composition selected after inspecting prior answer failures.",
            "Verified gold evidence isolates answer behavior and does not establish end-to-end evidence sufficiency.",
            "A single full-page retry still requires conflict checks before production use.",
        ],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--retry-suite", type=Path, required=True)
    parser.add_argument("--retry-result", type=Path, required=True)
    parser.add_argument("--negative-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = compose_selective_answer_candidate(
        suite=json.loads(args.suite.read_text(encoding="utf-8")),
        base_result=json.loads(args.base_result.read_text(encoding="utf-8")),
        retry_suite=json.loads(args.retry_suite.read_text(encoding="utf-8")),
        retry_result=json.loads(args.retry_result.read_text(encoding="utf-8")),
        negative_result=json.loads(args.negative_result.read_text(encoding="utf-8")),
    )
    artifact["inputs"] = {
        "answer_suite_sha256": sha256_file(args.suite),
        "base_result_sha256": sha256_file(args.base_result),
        "retry_suite_sha256": sha256_file(args.retry_suite),
        "retry_result_sha256": sha256_file(args.retry_result),
        "negative_result_sha256": sha256_file(args.negative_result),
    }
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
