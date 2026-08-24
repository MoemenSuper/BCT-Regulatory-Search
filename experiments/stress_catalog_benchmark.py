"""Report retrieval ranks on frozen targeted development cohorts."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    def hit(rank: int | None, cutoff: int) -> bool:
        return rank is not None and rank <= cutoff

    count = len(records)
    if not count:
        return {
            "n": 0,
            "current_page_at_1": None,
            "current_page_at_5": None,
            "current_page_at_20": None,
            "current_mrr_page": None,
            "fusion_page_at_1": None,
            "fusion_page_at_5": None,
            "fusion_page_at_20": None,
            "fusion_mrr_page": None,
            "repairs_at_5": 0,
            "regressions_at_5": 0,
            "net_at_5": 0,
        }

    current = [record["current_page"] for record in records]
    fusion = [record["fusion_page"] for record in records]
    repairs = sum(
        not hit(before, 5) and hit(after, 5)
        for before, after in zip(current, fusion)
    )
    regressions = sum(
        hit(before, 5) and not hit(after, 5)
        for before, after in zip(current, fusion)
    )

    def arm(values: list[int | None], prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_page_at_1": sum(hit(rank, 1) for rank in values) / count,
            f"{prefix}_page_at_5": sum(hit(rank, 5) for rank in values) / count,
            f"{prefix}_page_at_20": sum(hit(rank, 20) for rank in values) / count,
            f"{prefix}_mrr_page": statistics.mean(
                1.0 / rank if rank is not None else 0.0 for rank in values
            ),
        }

    return {
        "n": count,
        **arm(current, "current"),
        **arm(fusion, "fusion"),
        "repairs_at_5": repairs,
        "regressions_at_5": regressions,
        "net_at_5": repairs - regressions,
    }


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        "overall": _metrics(records),
        "fr": _metrics([record for record in records if record["language"] == "fr"]),
        "ar": _metrics([record for record in records if record["language"] == "ar"]),
    }
    roles = sorted({record["role"] for record in records if record.get("role")})
    if roles:
        output["by_role"] = {
            role: _metrics([record for record in records if record.get("role") == role])
            for role in roles
        }
    return output


def benchmark_stress_catalog(
    catalog: dict[str, Any],
    retrieval_result: dict[str, Any],
) -> dict[str, Any]:
    rank_records = retrieval_result["rank_records"]
    if catalog["inputs"]["evaluation_sha256"] != retrieval_result["inputs"]["evaluation_sha256"]:
        raise ValueError("Catalog and retrieval result use different evaluation hashes")
    if catalog["inputs"]["result_sha256"] != retrieval_result["inputs"]["current_result_sha256"]:
        raise ValueError("Catalog baseline and retrieval current-result hashes differ")

    suites: dict[str, Any] = {}
    for name, suite in catalog["suites"].items():
        relevant_cases = [case for case in suite["cases"] if case.get("relevant")]
        missing = sorted(case["id"] for case in relevant_cases if case["id"] not in rank_records)
        if missing:
            raise ValueError(f"Retrieval result is missing {name} IDs: {missing[:3]}")
        if not relevant_cases:
            suites[name] = {
                "status": "not_evaluated",
                "reason": "Retrieval rank metrics do not measure abstention or clarification quality.",
                "case_count": len(suite["cases"]),
            }
            continue
        records = [
            {
                "id": case["id"],
                "language": case.get("language"),
                "role": case.get("role"),
                "current_page": rank_records[case["id"]]["current_page"],
                "fusion_page": rank_records[case["id"]]["fusion_page"],
            }
            for case in relevant_cases
        ]
        suites[name] = {
            "status": "development_retrieval_only",
            "metrics": _group_metrics(records),
            "changed_at_5": [
                record
                for record in records
                if (record["current_page"] is not None and record["current_page"] <= 5)
                != (record["fusion_page"] is not None and record["fusion_page"] <= 5)
            ],
        }
    return {
        "status": "complete",
        "suite_type": "targeted_development_retrieval_benchmark",
        "configuration": {
            "current": "reproduced StructuredDocument sequential-1000/200 hybrid winner",
            "fusion": "additive Arabic OCR dense-5/BM25-5 with unchanged reranker",
            "metric": "exact expected page rank",
        },
        "limitations": [
            "All cohorts and both retrieval configurations were selected or analyzed on development data.",
            "Overlapping cohorts are descriptive slices and must not be combined as independent samples.",
            "Negative/ambiguous cases require answer-level abstention evaluation and receive no retrieval score.",
        ],
        "suites": suites,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--retrieval-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    retrieval = json.loads(args.retrieval_result.read_text(encoding="utf-8"))
    artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "catalog_sha256": sha256_file(args.catalog),
            "retrieval_result_sha256": sha256_file(args.retrieval_result),
        },
        **benchmark_stress_catalog(catalog, retrieval),
    }
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
