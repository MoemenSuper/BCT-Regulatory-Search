import json

import pytest

from experiments.artifacts import sha256_file
from experiments.validation_human_approval import (
    approve_validation_candidate,
    build_review_template,
)


def _inputs(tmp_path):
    candidate_path = tmp_path / "candidate.json"
    cases = [{"id": "case-1", "verification_status": "pending"}]
    candidate_path.write_text(json.dumps(cases), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit = {"inputs": {"validation_candidate_sha256": sha256_file(candidate_path)}}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return candidate_path, audit_path, cases, audit


def test_pending_template_cannot_approve(tmp_path):
    candidate_path, audit_path, cases, audit = _inputs(tmp_path)
    review = build_review_template(
        candidate_path=candidate_path,
        audit_path=audit_path,
        cases=cases,
        audit=audit,
    )

    with pytest.raises(ValueError, match="reviewer name"):
        approve_validation_candidate(
            candidate_path=candidate_path,
            audit_path=audit_path,
            cases=cases,
            audit=audit,
            review=review,
        )


def test_complete_independent_review_approves_all_cases(tmp_path):
    candidate_path, audit_path, cases, audit = _inputs(tmp_path)
    review = build_review_template(
        candidate_path=candidate_path,
        audit_path=audit_path,
        cases=cases,
        audit=audit,
    )
    review["reviewer"] = {
        "name": "Independent reviewer",
        "reviewed_at": "2026-08-24",
        "independent_of_candidate_curation": True,
        "confirmed_candidate_hash": True,
        "reviewed_source_documents": True,
        "saw_no_retrieval_or_model_outputs": True,
    }
    review["case_decisions"][0]["decision"] = "approve"

    approved, receipt = approve_validation_candidate(
        candidate_path=candidate_path,
        audit_path=audit_path,
        cases=cases,
        audit=audit,
        review=review,
    )

    assert approved[0]["verification_status"] == "independently_human_verified"
    assert receipt["approved_case_count"] == 1


def test_correction_requires_new_candidate(tmp_path):
    candidate_path, audit_path, cases, audit = _inputs(tmp_path)
    review = build_review_template(
        candidate_path=candidate_path,
        audit_path=audit_path,
        cases=cases,
        audit=audit,
    )
    review["reviewer"] = {
        "name": "Independent reviewer",
        "reviewed_at": "2026-08-24",
        "independent_of_candidate_curation": True,
        "confirmed_candidate_hash": True,
        "reviewed_source_documents": True,
        "saw_no_retrieval_or_model_outputs": True,
    }
    review["case_decisions"][0]["decision"] = "correct"

    with pytest.raises(ValueError, match="new candidate"):
        approve_validation_candidate(
            candidate_path=candidate_path,
            audit_path=audit_path,
            cases=cases,
            audit=audit,
            review=review,
        )
