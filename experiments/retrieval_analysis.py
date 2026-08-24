"""Stage-aware analysis for frozen retrieval experiment artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.artifacts import sha256_file, write_json_atomic


def _source_name(value: Any) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _page_numbers(metadata: dict[str, Any]) -> set[int]:
    value = metadata.get("pages", metadata.get("page"))
    if isinstance(value, list):
        return {int(item) for item in value}
    if isinstance(value, str):
        return {
            int(item.strip())
            for item in value.split(",")
            if item.strip().lstrip("-").isdigit()
        }
    return {int(value)} if value is not None else set()


def _rate(values: Iterable[bool], total: int) -> float:
    return sum(values) / total if total else 0.0


def _candidate_metrics(
    cases: list[dict[str, Any]], candidate_cache: dict[str, Any]
) -> dict[str, Any]:
    source_hits: list[bool] = []
    page_hits: list[bool] = []
    dense_ranks: list[int | None] = []
    bm25_ranks: list[int | None] = []
    pool_sizes: list[int] = []

    for case in cases:
        if case["id"] not in candidate_cache:
            raise ValueError(f"Candidate cache is missing evaluation case {case['id']}")
        candidates = candidate_cache[case["id"]].get("candidates", [])
        pool_sizes.append(len(candidates))
        expected_source = _source_name(case["expected_source"])
        expected_page = int(case["expected_page"])
        source_matches = []
        page_matches = []
        for candidate in candidates:
            metadata = candidate["document"]["metadata"]
            if _source_name(metadata.get("source")) != expected_source:
                continue
            source_matches.append(candidate)
            if expected_page in _page_numbers(metadata):
                page_matches.append(candidate)

        source_hits.append(bool(source_matches))
        page_hits.append(bool(page_matches))
        dense = []
        bm25 = []
        for candidate in page_matches:
            for ranks in candidate.get("ranks", {}).values():
                if ranks.get("dense") is not None:
                    dense.append(int(ranks["dense"]))
                if ranks.get("bm25") is not None:
                    bm25.append(int(ranks["bm25"]))
        dense_ranks.append(min(dense) if dense else None)
        bm25_ranks.append(min(bm25) if bm25 else None)

    total = len(cases)
    metrics = {
        "n": total,
        "source_pool_recall": _rate(source_hits, total),
        "exact_page_pool_recall": _rate(page_hits, total),
        "mean_candidate_pool_size": statistics.mean(pool_sizes) if pool_sizes else 0.0,
        "median_candidate_pool_size": statistics.median(pool_sizes) if pool_sizes else 0.0,
    }
    for cutoff in (5, 10, 20):
        metrics[f"dense_exact_page_recall_at_{cutoff}"] = _rate(
            (rank is not None and rank <= cutoff for rank in dense_ranks), total
        )
    for cutoff in (5, 10, 15):
        metrics[f"bm25_exact_page_recall_at_{cutoff}"] = _rate(
            (rank is not None and rank <= cutoff for rank in bm25_ranks), total
        )
    return metrics


def _ranking_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    metrics: dict[str, Any] = {"n": total}
    for label, field, cutoffs in (
        ("source", "source_rank", (1, 5, 20, 50)),
        ("exact_page", "exact_page_rank", (1, 5, 10, 20, 50)),
    ):
        ranks = [record["result"].get(field) for record in records]
        for cutoff in cutoffs:
            metrics[f"{label}_at_{cutoff}"] = _rate(
                (rank is not None and rank <= cutoff for rank in ranks), total
            )
        metrics[f"mrr_{label}"] = (
            sum(1 / rank if rank else 0 for rank in ranks) / total if total else 0.0
        )
    return metrics


def analyze_retrieval(
    evaluation: list[dict[str, Any]],
    result: dict[str, Any],
    candidate_cache: dict[str, Any],
) -> dict[str, Any]:
    """Return stage-aware metrics for overall, French, and Arabic cases."""
    cases_by_id = {case["id"]: case for case in evaluation}
    if len(cases_by_id) != len(evaluation):
        raise ValueError("Evaluation case IDs must be unique")
    records_by_id = {record["id"]: record for record in result["records"]}
    missing_records = [case_id for case_id in cases_by_id if case_id not in records_by_id]
    if missing_records:
        raise ValueError(f"Result artifact is missing {len(missing_records)} evaluation cases")

    groups: dict[str, Any] = {}
    for label, language in (("overall", None), ("fr", "fr"), ("ar", "ar")):
        cases = [
            case
            for case in evaluation
            if case.get("relevant") and (language is None or case.get("language") == language)
        ]
        records = [records_by_id[case["id"]] for case in cases]
        failures = Counter(
            category
            for record in records
            for category in record.get("failure_categories", [])
        )
        primary_failures = Counter(
            record["primary_failure_category"]
            for record in records
            if record.get("primary_failure_category")
        )
        negative_cases = [
            case
            for case in evaluation
            if not case.get("relevant")
            and (language is None or case.get("language") == language)
        ]
        negative_primary = Counter(
            records_by_id[case["id"]].get("primary_failure_category")
            or "negative_or_ambiguous_query"
            for case in negative_cases
            if case["id"] in records_by_id
        )
        groups[label] = {
            "candidate": _candidate_metrics(cases, candidate_cache),
            "ranking": _ranking_metrics(records),
            "primary_failures": dict(sorted(primary_failures.items())),
            "all_failure_labels": dict(sorted(failures.items())),
            "negative_or_ambiguous": {
                "case_count": len(negative_cases),
                "categories": dict(
                    sorted(Counter(case.get("category", "unknown") for case in negative_cases).items())
                ),
                "primary_categories": dict(sorted(negative_primary.items())),
            },
        }
    return {"groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    candidate_cache = json.loads(args.candidate_cache.read_text(encoding="utf-8"))
    analysis = analyze_retrieval(evaluation, result, candidate_cache)
    artifact = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "evaluation": {"path": str(args.evaluation.resolve()), "sha256": sha256_file(args.evaluation)},
            "result": {"path": str(args.result.resolve()), "sha256": sha256_file(args.result)},
            "candidate_cache": {
                "path": str(args.candidate_cache.resolve()),
                "sha256": sha256_file(args.candidate_cache),
            },
        },
        "configuration": result.get("configuration", {}),
        **analysis,
    }
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
