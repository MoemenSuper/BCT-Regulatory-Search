"""Freeze the negative/ambiguous answer cases for a bounded status-policy gate."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


def build_answer_status_suite(
    *, base_suite: dict[str, Any], base_suite_sha256: str
) -> dict[str, Any]:
    cases = [deepcopy(case) for case in base_suite["cases"] if not case.get("relevant")]
    if not cases:
        raise ValueError("Base answer suite has no negative or ambiguous cases")
    allowed = {"clarify", "abstain", "reject_out_of_scope"}
    invalid = [case["id"] for case in cases if case.get("expected_behavior") not in allowed]
    if invalid:
        raise ValueError(f"Negative cases have unsupported expected behavior: {invalid}")
    by_behavior = {
        behavior: sum(case["expected_behavior"] == behavior for case in cases)
        for behavior in sorted(allowed)
    }
    return {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"base_answer_suite_sha256": base_suite_sha256},
        "suite_type": "negative_and_ambiguous_answer_status_development",
        "selection": "All negative and ambiguous cases from the frozen gold-evidence answer suite.",
        "evaluation_protocol": (
            "Supply no evidence. Hold model and decoding fixed while changing only the response "
            "schema's non-empty answer constraint and the status decision policy."
        ),
        "answer_experiment": {
            "experiment_id": "claim-linked-status-policy-development-v2",
            "candidate_and_evidence": "all frozen negative and ambiguous cases with no evidence",
            "prompt_version": "bct-claim-linked-answer-v2",
        },
        "counts": {
            "total": len(cases),
            "relevant": 0,
            "negative_or_ambiguous": len(cases),
            "by_language": {
                language: sum(case["language"] == language for case in cases)
                for language in ("ar", "fr")
            },
            "by_expected_behavior": by_behavior,
        },
        "limitations": [
            "Development-only status gate selected from already inspected cases.",
            "Prompt behavior on eight cases cannot establish generalization.",
            "No evidence or retrieval is supplied, so this does not evaluate relevant-answer quality.",
        ],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_suite = json.loads(args.base_suite.read_text(encoding="utf-8"))
    artifact = build_answer_status_suite(
        base_suite=base_suite,
        base_suite_sha256=sha256_file(args.base_suite),
    )
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
