"""Access-logged evaluation of frozen validation and final-holdout splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.evaluation_protocol import _load_dataset


_ROLES = {"validation", "final_holdout"}
_RELEVANT_OUTCOMES = ("answer_correct", "citation_correct", "grounded")
_NEGATIVE_OUTCOMES = ("abstained", "clarification_requested", "safe_response")


def _nonempty(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _record_value(record: dict[str, Any], field: str) -> Any:
    for container in (record, record.get("result", {}), record.get("answer_evaluation", {})):
        if field in container:
            return container[field]
    return None


def _rate(values: Iterable[bool], total: int) -> float:
    return sum(values) / total if total else 0.0


def _observed_boolean_metric(
    records: list[dict[str, Any]], field: str
) -> dict[str, int | float]:
    values = [_record_value(record, field) for record in records]
    observed = [value for value in values if isinstance(value, bool)]
    return {
        "observed_count": len(observed),
        "true_count": sum(observed),
        "rate": _rate(observed, len(observed)),
    }


def _metrics_for_group(
    cases: list[dict[str, Any]], records_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    relevant = [case for case in cases if case["relevant"]]
    negative = [case for case in cases if not case["relevant"]]
    relevant_records = [records_by_id[case["id"]] for case in relevant]
    negative_records = [records_by_id[case["id"]] for case in negative]
    metrics: dict[str, Any] = {
        "case_count": len(cases),
        "relevant_count": len(relevant),
        "negative_or_ambiguous_count": len(negative),
    }
    for label, field, cutoffs in (
        ("source", "source_rank", (1, 5, 20)),
        ("exact_page", "exact_page_rank", (1, 5, 20)),
    ):
        ranks = [_record_value(record, field) for record in relevant_records]
        for cutoff in cutoffs:
            metrics[f"{label}_at_{cutoff}"] = _rate(
                (isinstance(rank, int) and 0 < rank <= cutoff for rank in ranks),
                len(relevant),
            )
        metrics[f"mrr_{label}"] = _rate(
            (1 / rank if isinstance(rank, int) and rank > 0 else 0.0 for rank in ranks),
            len(relevant),
        )
    for field in _RELEVANT_OUTCOMES:
        metrics[field] = _observed_boolean_metric(relevant_records, field)
    metrics["negative"] = {
        "case_count": len(negative),
        "categories": dict(
            sorted(Counter(case.get("category", "unknown") for case in negative).items())
        ),
        **{
            field: _observed_boolean_metric(negative_records, field)
            for field in _NEGATIVE_OUTCOMES
        },
    }
    return metrics


def _append_access(path: Path, event: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    prefix = existing if not existing or existing.endswith("\n") else existing + "\n"
    temporary.write_text(
        prefix + json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_frozen_split(
    *,
    protocol_path: Path,
    role: str,
    result_path: Path,
    output_path: Path,
    ledger_path: Path,
    purpose: str,
    accessed_by: str,
    code_commit: str,
    include_case_details: bool = False,
    accessed_at: str | None = None,
) -> dict[str, Any]:
    """Verify frozen inputs, emit metrics, and append a successful-access record."""
    if role not in _ROLES:
        raise ValueError("role must be validation or final_holdout")
    if role == "final_holdout" and include_case_details:
        raise ValueError("Final holdout evaluation is aggregate-only")
    purpose = _nonempty(purpose, "purpose")
    accessed_by = _nonempty(accessed_by, "accessed_by")
    code_commit = _nonempty(code_commit, "code_commit")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    try:
        frozen = protocol["sets"][role]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Protocol does not define frozen role {role}") from error
    dataset_path = Path(frozen["path"])
    actual_dataset_hash = sha256_file(dataset_path)
    if actual_dataset_hash != frozen["sha256"]:
        raise ValueError(f"Frozen {role} dataset hash does not match the protocol")
    cases = _load_dataset(dataset_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = result.get("records")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("Result artifact records must be a JSON array of objects")
    case_ids = [case["id"] for case in cases]
    record_ids = [record.get("id") for record in records]
    duplicates = sorted(
        case_id for case_id, count in Counter(record_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Result artifact has duplicate case IDs: {duplicates}")
    missing = sorted(set(case_ids) - set(record_ids))
    extra = sorted(set(record_ids) - set(case_ids), key=str)
    if missing:
        raise ValueError(f"Result artifact is missing {len(missing)} frozen evaluation cases")
    if extra:
        raise ValueError(f"Result artifact has {len(extra)} unexpected evaluation cases")
    records_by_id = {record["id"]: record for record in records}

    metrics = {
        label: _metrics_for_group(
            [case for case in cases if language is None or case["language"] == language],
            records_by_id,
        )
        for label, language in (("overall", None), ("fr", "fr"), ("ar", "ar"))
    }
    timestamp = accessed_at or datetime.now(timezone.utc).isoformat()
    protocol_hash = sha256_file(protocol_path)
    result_hash = sha256_file(result_path)
    detail_policy = (
        "case_details_explicitly_requested" if include_case_details else "aggregate_only"
    )
    artifact: dict[str, Any] = {
        "status": "complete",
        "evaluated_at": timestamp,
        "role": role,
        "detail_policy": detail_policy,
        "protocol_sha256": protocol_hash,
        "dataset_sha256": actual_dataset_hash,
        "result_sha256": result_hash,
        "code_commit": code_commit,
        "metrics": metrics,
    }
    if include_case_details:
        artifact["case_details"] = [
            {
                "id": case["id"],
                "language": case["language"],
                "relevant": case["relevant"],
                "result": records_by_id[case["id"]].get("result", {}),
                "answer_evaluation": records_by_id[case["id"]].get(
                    "answer_evaluation", {}
                ),
                "failure_categories": records_by_id[case["id"]].get(
                    "failure_categories", []
                ),
            }
            for case in cases
        ]

    event = {
        "access_id": str(uuid4()),
        "accessed_at": timestamp,
        "accessed_by": accessed_by,
        "purpose": purpose,
        "role": role,
        "detail_policy": detail_policy,
        "protocol_sha256": protocol_hash,
        "dataset_sha256": actual_dataset_hash,
        "result_sha256": result_hash,
        "code_commit": code_commit,
        "output_path": str(output_path.resolve()),
    }
    _append_access(ledger_path, event)
    write_json_atomic(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(_ROLES), required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--accessed-by", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--include-case-details", action="store_true")
    args = parser.parse_args()
    evaluate_frozen_split(
        protocol_path=args.protocol,
        role=args.role,
        result_path=args.result,
        output_path=args.output,
        ledger_path=args.ledger,
        purpose=args.purpose,
        accessed_by=args.accessed_by,
        code_commit=args.code_commit,
        include_case_details=args.include_case_details,
    )


if __name__ == "__main__":
    main()
