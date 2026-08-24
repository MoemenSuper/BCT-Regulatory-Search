import json
from pathlib import Path

from experiments.artifacts import sha256_file
from experiments.selective_answer_gate import (
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_SUITE_SHA256,
    _identifier_present,
    _number_literals,
    evaluate_selective_answer_gate,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments/results/selective_answer_candidate_v2.json"
SUITE_PATH = ROOT / "experiments/stress_suites/answer_safety_development_v1.json"


def _inputs():
    return (
        json.loads(RESULT_PATH.read_text(encoding="utf-8")),
        json.loads(SUITE_PATH.read_text(encoding="utf-8")),
    )


def _evaluate(result, suite, candidate_sha256=EXPECTED_CANDIDATE_SHA256):
    return evaluate_selective_answer_gate(
        result,
        suite,
        candidate_sha256=candidate_sha256,
        suite_sha256=EXPECTED_SUITE_SHA256,
    )


def test_frozen_selective_answer_gate_passes_exact_candidate():
    result, suite = _inputs()

    receipt = _evaluate(result, suite, sha256_file(RESULT_PATH))

    assert receipt["status"] == "passed"
    assert receipt["failed_checks"] == []


def test_gate_binds_semantic_labels_to_exact_reviewed_candidate_hash():
    result, suite = _inputs()
    result["records"][0]["response"]["answer"] = "A wholly false answer."

    receipt = _evaluate(result, suite, candidate_sha256="0" * 64)

    assert receipt["status"] == "failed"
    assert receipt["checks"]["frozen_artifact_binding"] is False


def test_gate_recomputes_claims_citations_and_negative_statuses():
    result, suite = _inputs()
    result["records"][0]["response"]["claims"] = []
    result["records"][0]["response"]["citations"] = []
    negative = next(record for record in result["records"] if not record["relevant"])
    negative["response"].update(status="answered", answer="", claims=[{}])

    receipt = _evaluate(result, suite)

    assert receipt["checks"]["all_relevant_responses_recompute_as_structurally_cited"] is False
    assert receipt["checks"]["all_negative_statuses_and_payloads_recomputed"] is False


def test_gate_requires_exact_unique_frozen_ids_and_suite_metadata():
    result, suite = _inputs()
    result["records"][0]["id"] = "bogus_not_in_suite"
    result["records"][1]["language"] = "ar"

    receipt = _evaluate(result, suite)

    assert receipt["checks"]["exact_frozen_case_ids_and_order"] is False
    assert receipt["checks"]["all_case_metadata_matches_suite"] is False


def test_literal_requirements_are_derived_from_suite_not_candidate_audit():
    result, suite = _inputs()
    for record in result["records"]:
        if record["relevant"]:
            record["automatic_audit"]["expected_numbers"] = []
            record["automatic_audit"]["expected_identifiers"] = []

    assert _evaluate(result, suite)["checks"][
        "all_expected_literals_recomputed_from_suite"
    ] is True


def test_number_normalization_preserves_decimal_scale_and_grouping():
    assert "100000" in _number_literals("100.000 DT")
    assert "1.00" in _number_literals("1,00 %")
    assert "1.00" not in _number_literals("100 %")
    assert "6.50" not in _number_literals("650 %")


def test_time_identifier_accepts_only_exact_zero_minute_suffix():
    assert _identifier_present("14h", {"14h00"}) is True
    assert _identifier_present("14h", {"14h999"}) is False
