"""Apply the frozen, artifact-bound gate to the selective answer candidate."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.numeric_fidelity_stress import critical_identifiers


EXPECTED_SUITE_SHA256 = "5E3D840B6DF8FDF3046AA2A5A67B84DF8CC0E0E61D4EC810E5D51446C89917A1"
EXPECTED_CANDIDATE_SHA256 = "DD3F3F35335243E2511EE1F945717B4462FCDDCC837F3C1AEE7D11670A5D6E70"
EXPECTED_GENERATED_SHA256 = "3CFF3B8FF91217F082969D25DA985E8AB195C7ABA31725BEE40D0E9B3F2D9A74"
EXPECTED_REVIEW_SHA256 = "6668B53D89CBAD4A278E90FF58E99EFC9366C4B08AB84DB2E41D390EB79EE910"
EXPECTED_RETRY_IDS = {
    "cir_2019_02_fr_amount_or_rate_02",
    "note_2022_16_ar_ceramic_importer_02",
}
_EXPECTED_NEGATIVE_STATUS = {
    "clarify": "clarification_needed",
    "reject_out_of_scope": "out_of_scope",
    "abstain": "insufficient_evidence",
}


def _ascii_digits(text: str) -> str:
    characters = []
    for character in unicodedata.normalize("NFKC", text):
        if character.isdigit():
            characters.append(str(unicodedata.digit(character)))
        else:
            characters.append(character)
    return "".join(characters)


def _number_literals(text: str) -> set[str]:
    normalized = _ascii_digits(text)
    normalized = re.sub(r"(?<=\d)[\s\u00a0\u202f](?=\d{3}\b)", "", normalized)
    literals = set()
    for value in re.findall(r"\d+(?:[.,]\d+)?", normalized):
        if "," in value:
            integer, fraction = value.split(",", 1)
            literals.add(f"{integer}.{fraction}")
        elif "." in value:
            integer, fraction = value.split(".", 1)
            literals.add(
                integer + fraction
                if len(fraction) == 3 and len(integer) <= 3
                else value
            )
        else:
            literals.add(value)
    return literals


def _identifier_literals(text: str) -> set[str]:
    return {
        value.strip("./-").casefold()
        for value in critical_identifiers(_ascii_digits(text))
    }


def _identifier_present(expected: str, answer_identifiers: set[str]) -> bool:
    if expected in answer_identifiers:
        return True
    if re.fullmatch(r"\d{1,2}h", expected):
        return f"{expected}00" in answer_identifiers
    return False


def _unique_records(records: list[dict[str, Any]]) -> bool:
    return len({record.get("id") for record in records}) == len(records)


def _record_metadata_matches(case: dict[str, Any], record: dict[str, Any]) -> bool:
    return (
        record.get("id") == case["id"]
        and record.get("language") == case["language"]
        and record.get("relevant") is bool(case["relevant"])
        and record.get("answer_suite_role") == case["answer_suite_role"]
        and record.get("expected_behavior") == case.get("expected_behavior")
    )


def _relevant_structure_matches(case: dict[str, Any], record: dict[str, Any]) -> bool:
    response = record.get("response", {})
    claims = response.get("claims")
    citations = response.get("citations")
    if (
        response.get("status") != "answered"
        or not isinstance(response.get("answer"), str)
        or not response["answer"].strip()
        or not isinstance(claims, list)
        or not claims
        or not isinstance(citations, list)
    ):
        return False
    expected_citation = (
        "E1",
        Path(str(case["expected_source"])).name.casefold(),
        int(case["expected_page"]),
    )
    citation_keys = [
        (
            citation.get("evidence_id"),
            Path(str(citation.get("source", ""))).name.casefold(),
            citation.get("page"),
        )
        for citation in citations
        if isinstance(citation, dict)
    ]
    if citation_keys != [expected_citation]:
        return False
    return all(
        isinstance(claim, dict)
        and isinstance(claim.get("text"), str)
        and claim["text"].strip()
        and claim.get("evidence_ids")
        and set(claim["evidence_ids"]) == {"E1"}
        for claim in claims
    )


def _negative_structure_matches(case: dict[str, Any], record: dict[str, Any]) -> bool:
    response = record.get("response", {})
    expected_status = _EXPECTED_NEGATIVE_STATUS[case["expected_behavior"]]
    return (
        response.get("status") == expected_status
        and isinstance(response.get("answer"), str)
        and bool(response["answer"].strip())
        and response.get("claims") == []
        and response.get("citations") == []
    )


def _review_labels_pass(case: dict[str, Any], record: dict[str, Any]) -> bool:
    evaluation = record.get("answer_evaluation", {})
    fields = (
        ("answer_correct", "citation_correct", "grounded")
        if case["relevant"]
        else ("safe_response", "expected_behavior_met")
    )
    return all(evaluation.get(field) is True for field in fields)


def _literals_preserved(case: dict[str, Any], record: dict[str, Any]) -> bool:
    expected_answer = str(case.get("expected_answer") or "")
    answer = str(record.get("response", {}).get("answer") or "")
    answer_numbers = _number_literals(answer)
    answer_identifiers = _identifier_literals(answer)
    return _number_literals(expected_answer).issubset(answer_numbers) and all(
        _identifier_present(identifier, answer_identifiers)
        for identifier in _identifier_literals(expected_answer)
    )


def evaluate_selective_answer_gate(
    result: dict[str, Any],
    suite: dict[str, Any],
    *,
    candidate_sha256: str,
    suite_sha256: str,
) -> dict[str, Any]:
    records = result.get("records", [])
    cases = suite.get("cases", [])
    record_by_id = {record.get("id"): record for record in records}
    exact_ids = (
        _unique_records(records)
        and [record.get("id") for record in records] == [case["id"] for case in cases]
    )
    paired = [(case, record_by_id.get(case["id"], {})) for case in cases]
    relevant = [(case, record) for case, record in paired if case.get("relevant")]
    negative = [(case, record) for case, record in paired if not case.get("relevant")]
    retry_ids = {
        case["id"]
        for case, record in relevant
        if record.get("answer_path") == "single_full_page_context_retry"
    }
    clarification_count = sum(
        record.get("response", {}).get("status") == "clarification_needed"
        for _case, record in negative
    )
    inputs = result.get("inputs", {})
    checks = {
        "frozen_artifact_binding": (
            candidate_sha256 == EXPECTED_CANDIDATE_SHA256
            and suite_sha256 == EXPECTED_SUITE_SHA256
            and inputs.get("generated_result_sha256") == EXPECTED_GENERATED_SHA256
            and inputs.get("review_sha256") == EXPECTED_REVIEW_SHA256
        ),
        "exact_frozen_case_ids_and_order": exact_ids,
        "frozen_case_counts": len(cases) == 40 and len(relevant) == 32 and len(negative) == 8,
        "all_case_metadata_matches_suite": exact_ids and all(
            _record_metadata_matches(case, record) for case, record in paired
        ),
        "all_bound_review_labels_pass": exact_ids and all(
            _review_labels_pass(case, record) for case, record in paired
        ),
        "all_relevant_responses_recompute_as_structurally_cited": exact_ids and all(
            _relevant_structure_matches(case, record) for case, record in relevant
        ),
        "all_expected_literals_recomputed_from_suite": exact_ids and all(
            _literals_preserved(case, record) for case, record in relevant
        ),
        "all_negative_statuses_and_payloads_recomputed": exact_ids and all(
            _negative_structure_matches(case, record) for case, record in negative
        ),
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
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    result_hash = sha256_file(args.result)
    suite_hash = sha256_file(args.suite)
    receipt = {
        "experiment_id": "selective-answer-candidate-development-gate-v2",
        "input_sha256": result_hash,
        "suite_sha256": suite_hash,
        **evaluate_selective_answer_gate(
            result,
            suite,
            candidate_sha256=result_hash,
            suite_sha256=suite_hash,
        ),
        "qualification": "development_only_gold_evidence_components",
        "limitations": [
            "Passing binds only the exact frozen candidate and suite; semantic correctness remains agent evidence review.",
            "Passing does not establish retrieval-time evidence sufficiency or release readiness.",
        ],
    }
    write_json_atomic(args.output, receipt)
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
