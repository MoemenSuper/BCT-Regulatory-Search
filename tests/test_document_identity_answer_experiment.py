import pytest

from experiments.document_identity_answer_experiment import (
    answer_cache_binding,
    derive_changed_answer_case_ids,
    parse_or_fail_closed,
    recompose_records,
    validate_top5_signature,
)


def _suite():
    return {"cases": [{"id": "unchanged"}, {"id": "changed"}]}


def _raw():
    return {"records": [{"id": "unchanged"}, {"id": "changed"}]}


def test_changed_answer_cases_are_derived_from_the_frozen_intersection():
    identity = {"changed_top5_ids": ["outside_suite", "changed"]}

    assert derive_changed_answer_case_ids(_suite(), _raw(), identity) == ["changed"]


def test_changed_answer_case_derivation_fails_on_suite_raw_mismatch():
    raw = {"records": [{"id": "changed"}]}

    with pytest.raises(ValueError, match="exactly cover"):
        derive_changed_answer_case_ids(_suite(), raw, {"changed_top5_ids": []})


def test_frozen_top5_validation_checks_source_page_and_score():
    expected = [
        {"source": "Cir_2018_03_ar.pdf", "page": 2, "score": 0.5},
        {"source": "Cir_2019_03_ar.pdf", "page": 2, "score": 0.25},
    ]
    actual = [dict(item) for item in expected]

    validate_top5_signature(actual, expected)
    actual[1]["page"] = 3
    with pytest.raises(ValueError, match="top-five replay differs"):
        validate_top5_signature(actual, expected)


def test_recomposition_replaces_only_the_frozen_changed_cases():
    raw_records = _raw()["records"]
    replacement = {"id": "changed", "response": {"status": "answered"}}

    composed = recompose_records(raw_records, {"changed": replacement}, ["changed"])

    assert composed[0] is raw_records[0]
    assert composed[1] is replacement
    with pytest.raises(ValueError, match="exactly match"):
        recompose_records(raw_records, {}, ["changed"])


def test_answer_cache_binding_includes_both_frozen_identity_artifacts():
    binding = answer_cache_binding(
        suite_sha256="A" * 64,
        raw_result_sha256="B" * 64,
        identity_result_sha256="C" * 64,
        routing_receipt_sha256="D" * 64,
        user_payload="payload",
    )

    assert binding["identity_result_sha256"] == "C" * 64
    assert binding["routing_receipt_sha256"] == "D" * 64
    assert binding["user_payload"] == "payload"


def test_structurally_inconsistent_nonanswer_fails_closed_without_claims():
    malformed = """{
      "status": "insufficient_evidence",
      "answer": "لا تكفي الأدلة.",
      "claims": [{"claim_id": "C1", "text": "ادعاء", "evidence_ids": ["E1"]}],
      "citations": [{"evidence_id": "E1", "source": "x.pdf", "page": 0}]
    }"""

    response, status, error = parse_or_fail_closed(malformed, "ar")

    assert status == "fail_closed_structured_validation"
    assert error
    assert response["status"] == "insufficient_evidence"
    assert response["claims"] == []
    assert response["citations"] == []
