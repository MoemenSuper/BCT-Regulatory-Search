"""Validate agent evidence-review labels and aggregate gold-evidence answer outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


_RELEVANT_FIELDS = ("answer_correct", "citation_correct", "grounded")
_NEGATIVE_FIELDS = (
    "abstained_or_refused",
    "clarification_requested",
    "safe_response",
    "expected_behavior_met",
)


def _validated_label(label: dict[str, Any], fields: tuple[str, ...], case_id: str) -> dict[str, Any]:
    missing = [field for field in fields if not isinstance(label.get(field), bool)]
    if missing:
        raise ValueError(f"Review label {case_id} lacks Boolean fields: {missing}")
    if not isinstance(label.get("note"), str) or not label["note"].strip():
        raise ValueError(f"Review label {case_id} requires a non-empty note")
    return {field: label[field] for field in fields} | {"note": label["note"].strip()}


def expand_review_labels(
    suite: dict[str, Any], review: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    default = review.get("relevant_default")
    if not isinstance(default, dict):
        raise ValueError("Review requires a relevant_default object")
    overrides = review.get("relevant_overrides", {})
    negative = review.get("negative_labels", {})
    labels: dict[str, dict[str, Any]] = {}
    relevant_ids = {case["id"] for case in suite["cases"] if case.get("relevant")}
    negative_ids = {case["id"] for case in suite["cases"] if not case.get("relevant")}
    if not set(overrides).issubset(relevant_ids):
        raise ValueError("Relevant review overrides contain unknown or negative IDs")
    if set(negative) != negative_ids:
        raise ValueError("Negative review labels must exactly cover negative suite IDs")
    for case in suite["cases"]:
        case_id = case["id"]
        if case.get("relevant"):
            value = {**default, **overrides.get(case_id, {})}
            labels[case_id] = _validated_label(value, _RELEVANT_FIELDS, case_id)
        else:
            labels[case_id] = _validated_label(
                negative[case_id], _NEGATIVE_FIELDS, case_id
            )
    return labels


def _rate(records: list[dict[str, Any]], field: str) -> float | None:
    return (
        sum(record["answer_evaluation"][field] for record in records) / len(records)
        if records
        else None
    )


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [record for record in records if record["relevant"]]
    negative = [record for record in records if not record["relevant"]]
    return {
        "case_count": len(records),
        "relevant_count": len(relevant),
        **{field: {"true_count": sum(r["answer_evaluation"][field] for r in relevant), "rate": _rate(relevant, field)} for field in _RELEVANT_FIELDS},
        "negative": {
            "case_count": len(negative),
            **{
                field: {
                    "true_count": sum(r["answer_evaluation"][field] for r in negative),
                    "rate": _rate(negative, field),
                }
                for field in _NEGATIVE_FIELDS
            },
        },
    }


def build_reviewed_answer_result(
    suite: dict[str, Any],
    generated: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    suite_ids = [case["id"] for case in suite["cases"]]
    generated_by_id = {record["id"]: record for record in generated["records"]}
    if len(generated_by_id) != len(generated["records"]):
        raise ValueError("Generated answer result has duplicate IDs")
    if set(generated_by_id) != set(suite_ids):
        raise ValueError("Generated answer result and answer suite IDs differ")
    labels = expand_review_labels(suite, review)
    records = [
        {
            **generated_by_id[case_id],
            "answer_evaluation": labels[case_id],
        }
        for case_id in suite_ids
    ]
    return {
        "status": "complete",
        "experiment_id": generated["experiment_id"],
        "configuration": generated["configuration"],
        "automatic_metrics": generated.get(
            "metrics", generated.get("automatic_metrics", {})
        ),
        "reviewed_metrics": {
            label: _metrics(
                [record for record in records if language is None or record["language"] == language]
            )
            for label, language in (("overall", None), ("fr", "fr"), ("ar", "ar"))
        },
        "latency_seconds": generated["latency_seconds"],
        "review_status": {
            "reviewer_type": review["reviewer_type"],
            "reviewed_against": review["reviewed_against"],
            "independent_human_confirmation": False,
            "note": (
                "All cases were inspected against the verified expected answer and supplied evidence, "
                "but these are agent evidence-review labels, not independent human adjudication."
            ),
        },
        "limitations": [
            *generated["limitations"],
            "Agent evidence-review labels should be independently confirmed before a release claim.",
            "Gold evidence isolates generation and does not measure end-to-end retrieval sufficiency.",
        ],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--generated-result", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    generated = json.loads(args.generated_result.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    artifact = {
        "inputs": {
            "answer_suite_sha256": sha256_file(args.suite),
            "generated_result_sha256": sha256_file(args.generated_result),
            "review_sha256": sha256_file(args.review),
        },
        **build_reviewed_answer_result(suite, generated, review),
    }
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
