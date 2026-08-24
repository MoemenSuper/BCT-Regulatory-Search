"""Freeze targeted development cohorts without consulting model experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.numeric_fidelity_stress import critical_identifiers


WRONG_VERSION = "wrong_temporal_or_document_version"
CONTEXT_PROBLEM = "chunk_boundary_or_context_problem"


def _source_name(value: Any) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _selection_key(target_id: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{target_id}|{candidate_id}".encode()).hexdigest()


def _case_entry(
    case: dict[str, Any],
    *,
    role: str | None = None,
    page_count: int | None = None,
    has_table: bool | None = None,
) -> dict[str, Any]:
    entry = {
        "id": case["id"],
        "language": case.get("language"),
        "category": case.get("category"),
        "evidence_method": case.get("evidence_method"),
        "relevant": bool(case.get("relevant")),
        "expected_source": _source_name(case.get("expected_source", "")),
        "expected_page": case.get("expected_page"),
    }
    if role is not None:
        entry["role"] = role
    if page_count is not None:
        entry["source_page_count"] = page_count
    if has_table is not None:
        entry["expected_page_has_table_block"] = has_table
    return entry


def _counts(cases: list[dict[str, Any]]) -> dict[str, Any]:
    languages = Counter(str(case.get("language")) for case in cases)
    categories = Counter(str(case.get("category")) for case in cases)
    roles = Counter(str(case.get("role")) for case in cases if case.get("role"))
    return {
        "total": len(cases),
        "by_language": dict(sorted(languages.items())),
        "by_category": dict(sorted(categories.items())),
        "by_role": dict(sorted(roles.items())),
    }


def _matched_controls(
    targets: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    *,
    excluded: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], int]:
    eligible = [
        case
        for case in evaluation
        if case.get("relevant")
        and not excluded(records_by_id[case["id"]])
        and records_by_id[case["id"]].get("result", {}).get("exact_page_rank") is not None
        and int(records_by_id[case["id"]]["result"]["exact_page_rank"]) <= 5
    ]
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    unmatched = 0
    for target in sorted(targets, key=lambda value: value["id"]):
        same_language = [
            candidate
            for candidate in eligible
            if candidate["id"] not in used_ids
            and candidate.get("language") == target.get("language")
        ]
        same_category = [
            candidate
            for candidate in same_language
            if candidate.get("category") == target.get("category")
        ]
        pool = same_category or same_language
        if not pool:
            unmatched += 1
            continue
        chosen = min(pool, key=lambda value: _selection_key(target["id"], value["id"]))
        selected.append(chosen)
        used_ids.add(chosen["id"])
    return selected, unmatched


def _paired_failure_suite(
    *,
    suite_type: str,
    selection: str,
    limitation: str,
    targets: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    excluded: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    controls, unmatched = _matched_controls(
        targets,
        evaluation,
        records_by_id,
        excluded=excluded,
    )
    cases = [
        *(_case_entry(case, role="failure") for case in targets),
        *(_case_entry(case, role="control") for case in controls),
    ]
    cases.sort(key=lambda value: (value.get("language") or "", value["role"], value["id"]))
    return {
        "suite_type": suite_type,
        "selection": selection,
        "counts": _counts(cases),
        "unmatched_control_count": unmatched,
        "limitations": [limitation],
        "cases": cases,
    }


def _load_structured_pages(
    manifest: dict[str, Any],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, int]]:
    pages: dict[tuple[str, int], dict[str, Any]] = {}
    page_counts: dict[str, int] = {}
    for record in manifest["records"].values():
        source = _source_name(record["source"]).casefold()
        page_counts[source] = int(record["pages"])
        document = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        for page in document["pages"]:
            pages[(source, int(page["page_number"]))] = page
    return pages, page_counts


def build_targeted_stress_catalog(
    evaluation: list[dict[str, Any]],
    result: dict[str, Any],
    structured_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build fixed cohorts from labels and artifacts that predate future experiments."""
    records_by_id = {record["id"]: record for record in result["records"]}
    missing_records = sorted(case["id"] for case in evaluation if case["id"] not in records_by_id)
    if missing_records:
        raise ValueError(f"Result is missing evaluation IDs: {missing_records[:3]}")
    pages, page_counts = _load_structured_pages(structured_manifest)

    enriched: list[tuple[dict[str, Any], bool, int]] = []
    for case in evaluation:
        if not case.get("relevant"):
            continue
        source = _source_name(case["expected_source"]).casefold()
        page_number = int(case["expected_page"])
        page = pages.get((source, page_number))
        if page is None:
            raise ValueError(f"Structured expected page is missing for {case['id']}")
        has_table = any(block.get("type") == "table" for block in page.get("blocks", []))
        enriched.append((case, has_table, page_counts[source]))

    table_cases = [
        _case_entry(case, page_count=count, has_table=True)
        for case, has_table, count in enriched
        if has_table
    ]
    visual_non_table_cases = [
        _case_entry(case, page_count=count, has_table=False)
        for case, has_table, count in enriched
        if case.get("evidence_method") == "visual_review" and not has_table
    ]
    long_document_cases = [
        _case_entry(case, page_count=count, has_table=has_table)
        for case, has_table, count in enriched
        if count >= 20
    ]
    identifier_cases = [
        _case_entry(case, page_count=count, has_table=has_table)
        for case, has_table, count in enriched
        if critical_identifiers(str(case.get("evidence_quote", "")))
    ]
    ambiguity_cases = [
        _case_entry(case, role="negative_or_ambiguous")
        for case in evaluation
        if not case.get("relevant")
    ]

    for cases in (
        table_cases,
        visual_non_table_cases,
        long_document_cases,
        identifier_cases,
        ambiguity_cases,
    ):
        cases.sort(key=lambda value: value["id"])

    version_targets = [
        case
        for case in evaluation
        if case.get("relevant")
        and WRONG_VERSION in records_by_id[case["id"]].get("failure_categories", [])
    ]
    context_targets = [
        case
        for case in evaluation
        if case.get("relevant")
        and records_by_id[case["id"]].get("primary_failure_category") == CONTEXT_PROBLEM
    ]

    suites = {
        "table_pages": {
            "suite_type": "development_structured_table_pages",
            "selection": "All relevant cases whose expected StructuredDocument page contains at least one table block.",
            "counts": _counts(table_cases),
            "limitations": [
                "Table-block presence does not prove that the labeled answer itself is in the table."
            ],
            "cases": table_cases,
        },
        "visual_non_table": {
            "suite_type": "development_visual_review_without_table_block",
            "selection": "All relevant visual-review cases whose expected StructuredDocument page has no table block.",
            "counts": _counts(visual_non_table_cases),
            "limitations": [
                "This is an image/visual-annex proxy; visual_review does not guarantee a full-page image or annex."
            ],
            "cases": visual_non_table_cases,
        },
        "temporal_near_duplicate": _paired_failure_suite(
            suite_type="development_wrong_version_diagnostic_with_controls",
            selection=(
                "All cases carrying the frozen wrong-version diagnostic plus unique deterministic "
                "same-language, same-category Top-5 controls where available."
            ),
            limitation=(
                "The failure cohort is retrieval-diagnostic, not a complete corpus-level map of amendments, "
                "supersession, or document families."
            ),
            targets=version_targets,
            evaluation=evaluation,
            records_by_id=records_by_id,
            excluded=lambda record: WRONG_VERSION in record.get("failure_categories", []),
        ),
        "long_documents": {
            "suite_type": "development_documents_at_least_20_pages",
            "selection": "All relevant cases whose source StructuredDocument contains at least 20 pages.",
            "counts": _counts(long_document_cases),
            "limitations": [
                "The current development cohort is French-only, so it cannot support an Arabic long-document claim."
            ],
            "cases": long_document_cases,
        },
        "context_dependence": _paired_failure_suite(
            suite_type="development_primary_context_failure_with_controls",
            selection=(
                "All cases whose first frozen failure stage is chunk/context plus unique deterministic "
                "same-language, same-category Top-5 controls where available."
            ),
            limitation=(
                "Only primary diagnosed context failures are included; this is a small hard-case cohort, "
                "not a complete semantic annotation of context dependence."
            ),
            targets=context_targets,
            evaluation=evaluation,
            records_by_id=records_by_id,
            excluded=lambda record: record.get("primary_failure_category") == CONTEXT_PROBLEM,
        ),
        "ambiguity_abstention": {
            "suite_type": "development_negative_and_ambiguous_queries",
            "selection": "All frozen cases labeled relevant=false.",
            "counts": _counts(ambiguity_cases),
            "limitations": [
                "Eight curated negatives cannot estimate open-world abstention performance on their own."
            ],
            "cases": ambiguity_cases,
        },
        "identifiers": {
            "suite_type": "development_latin_alphanumeric_identifier_evidence",
            "selection": (
                "All relevant cases whose verified evidence contains a Latin alphanumeric token with both "
                "letters and digits."
            ),
            "counts": _counts(identifier_cases),
            "limitations": [
                "This deterministic detector excludes numeric-only legal references such as Article 15."
            ],
            "cases": identifier_cases,
        },
    }
    return {
        "suite_type": "targeted_development_catalog",
        "selection_independence": (
            "Cohorts use frozen evaluation labels, the reproduced baseline diagnostics, and StructuredDocument "
            "artifacts only; no OCR-fusion or VLM output influenced membership."
        ),
        "suite_counts": {name: suite["counts"]["total"] for name, suite in suites.items()},
        "limitations": [
            "All cohorts are development data and may be inspected for diagnosis; they do not establish generalization.",
            "Cases may belong to more than one suite, so suite totals must not be summed as unique questions.",
        ],
        "suites": suites,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    manifest = json.loads(args.structured_manifest.read_text(encoding="utf-8"))
    catalog = build_targeted_stress_catalog(evaluation, result, manifest)
    artifact = {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "evaluation_sha256": sha256_file(args.evaluation),
            "result_sha256": sha256_file(args.result),
            "structured_manifest_sha256": sha256_file(args.structured_manifest),
        },
        **catalog,
    }
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
