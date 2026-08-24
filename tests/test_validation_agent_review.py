import json

import pytest

from experiments.artifacts import sha256_file
from experiments.validation_agent_review import verify_agent_review


def _inputs(tmp_path):
    candidate_path = tmp_path / "candidate.json"
    cases = [{"id": "case-1"}, {"id": "case-2"}]
    candidate_path.write_text(json.dumps(cases), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit = {"inputs": {"validation_candidate_sha256": sha256_file(candidate_path)}}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    review = {
        "candidate_sha256": sha256_file(candidate_path),
        "audit_sha256": sha256_file(audit_path),
        "reviewer": {
            "agent_label": "blind-review-agent",
            "reviewed_at": "2026-08-24",
            "independent_of_candidate_curation": True,
            "confirmed_candidate_hash": True,
            "reviewed_source_documents": True,
            "saw_no_retrieval_or_model_outputs": True,
            "is_human_reviewer": False,
        },
        "approved_case_ids": ["case-1", "case-2"],
    }
    return candidate_path, audit_path, cases, audit, review


def test_complete_agent_review_permits_only_provisional_validation(tmp_path):
    candidate_path, audit_path, cases, audit, review = _inputs(tmp_path)

    receipt = verify_agent_review(
        candidate_path=candidate_path,
        audit_path=audit_path,
        cases=cases,
        audit=audit,
        review=review,
    )

    assert receipt["approved_case_count"] == 2
    assert receipt["allowed_use"] == "provisional_validation_only"
    assert receipt["independent_human_approval"] is False


def test_agent_review_rejects_incomplete_case_coverage(tmp_path):
    candidate_path, audit_path, cases, audit, review = _inputs(tmp_path)
    review["approved_case_ids"] = ["case-1"]

    with pytest.raises(ValueError, match="exactly cover"):
        verify_agent_review(
            candidate_path=candidate_path,
            audit_path=audit_path,
            cases=cases,
            audit=audit,
            review=review,
        )


def test_agent_review_rejects_a_human_claim(tmp_path):
    candidate_path, audit_path, cases, audit, review = _inputs(tmp_path)
    review["reviewer"]["is_human_reviewer"] = True

    with pytest.raises(ValueError, match="not human review"):
        verify_agent_review(
            candidate_path=candidate_path,
            audit_path=audit_path,
            cases=cases,
            audit=audit,
            review=review,
        )
