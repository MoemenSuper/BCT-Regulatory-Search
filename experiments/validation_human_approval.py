"""Create and verify independent human approval for a validation candidate."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


def build_review_template(
    *, candidate_path: Path, audit_path: Path, cases: list[dict[str, Any]], audit: dict[str, Any]
) -> dict[str, Any]:
    candidate_hash = sha256_file(candidate_path)
    audit_hash = sha256_file(audit_path)
    if audit["inputs"]["validation_candidate_sha256"] != candidate_hash:
        raise ValueError("Audit and validation candidate hashes differ")
    return {
        "review_version": 1,
        "candidate_sha256": candidate_hash,
        "audit_sha256": audit_hash,
        "reviewer": {
            "name": "",
            "reviewed_at": "",
            "independent_of_candidate_curation": False,
            "confirmed_candidate_hash": False,
            "reviewed_source_documents": False,
            "saw_no_retrieval_or_model_outputs": False,
        },
        "case_decisions": [
            {"id": case["id"], "decision": "pending", "note": ""} for case in cases
        ],
    }


def approve_validation_candidate(
    *,
    candidate_path: Path,
    audit_path: Path,
    cases: list[dict[str, Any]],
    audit: dict[str, Any],
    review: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_hash = sha256_file(candidate_path)
    audit_hash = sha256_file(audit_path)
    if audit["inputs"]["validation_candidate_sha256"] != candidate_hash:
        raise ValueError("Audit and validation candidate hashes differ")
    if review.get("candidate_sha256") != candidate_hash or review.get("audit_sha256") != audit_hash:
        raise ValueError("Human review hashes do not match the candidate and audit")
    reviewer = review.get("reviewer", {})
    required_text = ("name", "reviewed_at")
    if any(not isinstance(reviewer.get(field), str) or not reviewer[field].strip() for field in required_text):
        raise ValueError("Human review requires reviewer name and review date")
    attestations = (
        "independent_of_candidate_curation",
        "confirmed_candidate_hash",
        "reviewed_source_documents",
        "saw_no_retrieval_or_model_outputs",
    )
    if any(reviewer.get(field) is not True for field in attestations):
        raise ValueError("Every human reviewer attestation must be true")
    decisions = review.get("case_decisions", [])
    decision_by_id = {value.get("id"): value for value in decisions}
    case_ids = [case["id"] for case in cases]
    if len(decision_by_id) != len(decisions) or set(decision_by_id) != set(case_ids):
        raise ValueError("Human decisions must exactly cover unique candidate IDs")
    unapproved = [
        case_id
        for case_id in case_ids
        if decision_by_id[case_id].get("decision") != "approve"
    ]
    if unapproved:
        raise ValueError(
            "Every case must be approved; corrections or rejections require a new candidate: "
            + ", ".join(unapproved)
        )
    approved = deepcopy(cases)
    for case in approved:
        case["verification_status"] = "independently_human_verified"
    receipt = {
        "status": "independently_human_verified",
        "candidate_sha256": candidate_hash,
        "audit_sha256": audit_hash,
        "review_sha256": None,
        "reviewer": reviewer,
        "approved_case_count": len(approved),
        "rule": "Any correction or rejection requires a new candidate and audit before approval.",
    }
    return approved, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("template")
    template.add_argument("--candidate", type=Path, required=True)
    template.add_argument("--audit", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    approve = subparsers.add_parser("approve")
    approve.add_argument("--candidate", type=Path, required=True)
    approve.add_argument("--audit", type=Path, required=True)
    approve.add_argument("--review", type=Path, required=True)
    approve.add_argument("--output", type=Path, required=True)
    approve.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.candidate.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if args.command == "template":
        write_json_atomic(
            args.output,
            build_review_template(
                candidate_path=args.candidate,
                audit_path=args.audit,
                cases=cases,
                audit=audit,
            ),
        )
        return
    review = json.loads(args.review.read_text(encoding="utf-8"))
    approved, receipt = approve_validation_candidate(
        candidate_path=args.candidate,
        audit_path=args.audit,
        cases=cases,
        audit=audit,
        review=review,
    )
    receipt["review_sha256"] = sha256_file(args.review)
    write_json_atomic(args.output, approved)
    write_json_atomic(args.receipt, receipt)


if __name__ == "__main__":
    main()
