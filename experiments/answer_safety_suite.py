"""Freeze a bounded development suite for gold-evidence answer safety tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


def _order(seed: str, case: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}|{case['id']}".encode()).hexdigest()


def _select(
    *,
    role: str,
    pool: list[dict[str, Any]],
    count: int,
    used: set[str],
) -> list[dict[str, Any]]:
    eligible = [case for case in pool if case["id"] not in used]
    if len(eligible) < count:
        raise ValueError(f"Answer suite role {role} needs {count} cases but has {len(eligible)}")
    selected = sorted(eligible, key=lambda case: _order(role, case))[:count]
    used.update(case["id"] for case in selected)
    return [{**case, "answer_suite_role": role} for case in selected]


def build_answer_safety_suite(
    evaluation: list[dict[str, Any]],
    current_result: dict[str, Any],
    targeted_catalog: dict[str, Any],
    numeric_suite: dict[str, Any],
) -> dict[str, Any]:
    """Select disjoint high-risk and clean cases before answer-model testing."""
    by_id = {case["id"]: case for case in evaluation}
    records = {record["id"]: record for record in current_result["records"]}
    if set(by_id) != set(records):
        raise ValueError("Evaluation and current result IDs differ")
    catalog_ids = {
        name: [case["id"] for case in suite["cases"]]
        for name, suite in targeted_catalog["suites"].items()
    }
    missing_catalog = sorted(
        case_id
        for ids in catalog_ids.values()
        for case_id in ids
        if case_id not in by_id
    )
    if missing_catalog:
        raise ValueError(f"Targeted catalog contains unknown IDs: {missing_catalog[:3]}")

    used: set[str] = set()
    selected: list[dict[str, Any]] = []
    numeric_ids = [case["id"] for case in numeric_suite["cases"]]
    selected += _select(
        role="arabic_numeric",
        pool=[by_id[case_id] for case_id in numeric_ids],
        count=4,
        used=used,
    )
    selected += _select(
        role="arabic_visual_non_table",
        pool=[
            by_id[case_id]
            for case_id in catalog_ids["visual_non_table"]
            if by_id[case_id].get("language") == "ar"
        ],
        count=4,
        used=used,
    )
    selected += _select(
        role="french_table",
        pool=[
            by_id[case_id]
            for case_id in catalog_ids["table_pages"]
            if by_id[case_id].get("language") == "fr"
        ],
        count=4,
        used=used,
    )
    selected += _select(
        role="french_long_document",
        pool=[by_id[case_id] for case_id in catalog_ids["long_documents"]],
        count=4,
        used=used,
    )
    selected += _select(
        role="context_or_version",
        pool=[
            by_id[case_id]
            for name in ("context_dependence", "temporal_near_duplicate")
            for case_id in catalog_ids[name]
            if next(
                item
                for item in targeted_catalog["suites"][name]["cases"]
                if item["id"] == case_id
            ).get("role") == "failure"
        ],
        count=4,
        used=used,
    )
    selected += _select(
        role="latin_alphanumeric_identifier",
        pool=[by_id[case_id] for case_id in catalog_ids["identifiers"]],
        count=4,
        used=used,
    )

    hard_ids = set().union(*(set(values) for values in catalog_ids.values()))
    for language in ("ar", "fr"):
        selected += _select(
            role=f"clean_top1_{language}",
            pool=[
                case
                for case in evaluation
                if case.get("relevant")
                and case.get("language") == language
                and case["id"] not in hard_ids
                and records[case["id"]].get("result", {}).get("exact_page_rank") == 1
                and case.get("evidence_method") == "text_extraction"
            ],
            count=4,
            used=used,
        )

    negative = [case for case in evaluation if not case.get("relevant")]
    if len(negative) != 8:
        raise ValueError(f"Expected all 8 frozen negative/ambiguous cases, found {len(negative)}")
    selected += [{**case, "answer_suite_role": "negative_or_ambiguous"} for case in negative]
    ids = [case["id"] for case in selected]
    if len(ids) != len(set(ids)):
        raise AssertionError("Answer safety suite roles must be disjoint")
    selected.sort(key=lambda case: case["id"])
    return {
        "suite_type": "gold_evidence_answer_safety_development",
        "selection": (
            "Disjoint deterministic 32-case relevant sample spanning numeric, visual, table, "
            "long-document, context/version, identifier, and clean Top-1 roles, plus all 8 "
            "negative/ambiguous development cases."
        ),
        "evaluation_protocol": (
            "Feed only the verified evidence snippet and exact source/page for relevant cases; "
            "feed no regulatory evidence for negative cases. This isolates generation from retrieval."
        ),
        "counts": {
            "total": len(selected),
            "relevant": sum(case.get("relevant") for case in selected),
            "negative_or_ambiguous": sum(not case.get("relevant") for case in selected),
            "by_language": dict(sorted(Counter(case["language"] for case in selected).items())),
            "by_role": dict(
                sorted(Counter(case["answer_suite_role"] for case in selected).items())
            ),
        },
        "limitations": [
            "Development-only suite selected from already inspected questions; it cannot establish generalization.",
            "Gold-evidence evaluation measures generation, citation, and refusal behavior but not retrieval sufficiency.",
            "Automated outcome labels must remain separate from human-verified answer correctness and claim support.",
        ],
        "cases": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--current-result", type=Path, required=True)
    parser.add_argument("--targeted-catalog", type=Path, required=True)
    parser.add_argument("--numeric-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "evaluation_sha256": sha256_file(args.evaluation),
            "current_result_sha256": sha256_file(args.current_result),
            "targeted_catalog_sha256": sha256_file(args.targeted_catalog),
            "numeric_suite_sha256": sha256_file(args.numeric_suite),
        },
        **build_answer_safety_suite(
            json.loads(args.evaluation.read_text(encoding="utf-8")),
            json.loads(args.current_result.read_text(encoding="utf-8")),
            json.loads(args.targeted_catalog.read_text(encoding="utf-8")),
            json.loads(args.numeric_suite.read_text(encoding="utf-8")),
        ),
    }
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
