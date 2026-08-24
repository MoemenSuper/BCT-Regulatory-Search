"""Apply the frozen development gate to the selective answer candidate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


EXPECTED_RETRY_IDS = {
    "cir_2019_02_fr_amount_or_rate_02",
    "note_2022_16_ar_ceramic_importer_02",
}


def _number_present(expected: str, answer: str) -> bool:
    answer_numbers = re.findall(r"\d+(?:[\s.,:\u00a0\u202f]\d+)*", answer)
    expected_digits = re.sub(r"\D", "", expected)
    return expected_digits in {
        re.sub(r"\D", "", value) for value in answer_numbers
    }


def _identifier_present(expected: str, answer: str) -> bool:
    expected = expected.casefold()
    tokens = {value.casefold() for value in re.findall(r"[A-Za-z0-9]+", answer)}
    if expected in tokens:
        return True
    if re.fullmatch(r"\d{1,2}h", expected):
        return any(token.startswith(expected) and token[len(expected):].isdigit() for token in tokens)
    return False


def evaluate_selective_answer_gate(result: dict[str, Any]) -> dict[str, Any]:
    records = result["records"]
    relevant = [record for record in records if record["relevant"]]
    negative = [record for record in records if not record["relevant"]]

    relevant_semantic_pass = all(
        all(record["answer_evaluation"][field] for field in (
            "answer_correct",
            "citation_correct",
            "grounded",
        ))
        for record in relevant
    )
    relevant_structural_pass = all(
        record["response"]["status"] == "answered"
        and record["structured_diagnostics"]["exact_structured_citation"]
        and record["structured_diagnostics"]["claim_evidence_links_valid"]
        for record in relevant
    )
    expected_literals_preserved = all(
        all(
            _number_present(value, record["response"]["answer"])
            for value in record["automatic_audit"]["expected_numbers"]
        )
        and all(
            _identifier_present(value, record["response"]["answer"])
            for value in record["automatic_audit"]["expected_identifiers"]
        )
        for record in relevant
    )
    negative_policy_pass = all(
        record["answer_evaluation"]["safe_response"]
        and record["answer_evaluation"]["expected_behavior_met"]
        and not record["response"]["claims"]
        and not record["response"]["citations"]
        for record in negative
    )
    clarification_count = sum(
        record["response"]["status"] == "clarification_needed"
        and record["answer_evaluation"]["clarification_requested"]
        for record in negative
    )
    retry_ids = {
        record["id"]
        for record in relevant
        if record.get("answer_path") == "single_full_page_context_retry"
    }

    checks = {
        "frozen_case_counts": len(records) == 40 and len(relevant) == 32 and len(negative) == 8,
        "all_relevant_semantically_correct": relevant_semantic_pass,
        "all_relevant_structurally_cited": relevant_structural_pass,
        "all_expected_literals_preserved": expected_literals_preserved,
        "all_negatives_safe_and_expected": negative_policy_pass,
        "exactly_two_clarifications": clarification_count == 2,
        "only_frozen_context_retries": retry_ids == EXPECTED_RETRY_IDS,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failed_checks else "failed",
        "checks": checks,
        "failed_checks": failed_checks,
        "counts": {
            "cases": len(records),
            "relevant": len(relevant),
            "negative": len(negative),
            "clarifications": clarification_count,
            "context_retries": len(retry_ids),
        },
        "retry_ids": sorted(retry_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    receipt = {
        "experiment_id": "selective-answer-candidate-development-gate-v1",
        "input_sha256": sha256_file(args.result),
        **evaluate_selective_answer_gate(result),
        "qualification": "development_only_gold_evidence_components",
        "limitations": [
            "Passing does not establish retrieval-time evidence sufficiency.",
            "The semantic labels are agent evidence review, not independent human adjudication.",
        ],
    }
    write_json_atomic(args.output, receipt)
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
