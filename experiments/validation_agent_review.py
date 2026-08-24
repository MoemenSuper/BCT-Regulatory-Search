"""Verify a hash-bound second-agent source review without calling it human approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


def verify_agent_review(
    *,
    candidate_path: Path,
    audit_path: Path,
    cases: list[dict[str, Any]],
    audit: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    candidate_hash = sha256_file(candidate_path)
    audit_hash = sha256_file(audit_path)
    if audit["inputs"]["validation_candidate_sha256"] != candidate_hash:
        raise ValueError("Audit and validation candidate hashes differ")
    if review.get("candidate_sha256") != candidate_hash or review.get(
        "audit_sha256"
    ) != audit_hash:
        raise ValueError("Agent review hashes do not match the candidate and audit")
    reviewer = review.get("reviewer", {})
    if not str(reviewer.get("agent_label", "")).strip() or not str(
        reviewer.get("reviewed_at", "")
    ).strip():
        raise ValueError("Agent review requires an agent label and review date")
    attestations = (
        "independent_of_candidate_curation",
        "confirmed_candidate_hash",
        "reviewed_source_documents",
        "saw_no_retrieval_or_model_outputs",
    )
    if any(reviewer.get(field) is not True for field in attestations):
        raise ValueError("Every agent reviewer attestation must be true")
    if reviewer.get("is_human_reviewer") is not False:
        raise ValueError("Agent review must explicitly state that it is not human review")
    approved_ids = review.get("approved_case_ids", [])
    case_ids = [case["id"] for case in cases]
    if len(set(approved_ids)) != len(approved_ids) or set(approved_ids) != set(case_ids):
        raise ValueError("Agent approvals must exactly cover unique candidate IDs")
    return {
        "status": "provisionally_double_agent_verified_not_human_adjudicated",
        "candidate_sha256": candidate_hash,
        "audit_sha256": audit_hash,
        "review_sha256": None,
        "reviewer": reviewer,
        "approved_case_count": len(case_ids),
        "independent_human_approval": False,
        "allowed_use": "provisional_validation_only",
        "rule": (
            "This source-only second-agent review permits provisional validation; it does "
            "not replace independent human adjudication or a future family-disjoint holdout."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.candidate.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    receipt = verify_agent_review(
        candidate_path=args.candidate,
        audit_path=args.audit,
        cases=cases,
        audit=audit,
        review=review,
    )
    receipt["review_sha256"] = sha256_file(args.review)
    write_json_atomic(args.receipt, receipt)


if __name__ == "__main__":
    main()
