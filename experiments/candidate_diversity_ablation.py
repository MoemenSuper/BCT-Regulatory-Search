"""Test stable source-page diversity after additive OCR reranking."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.ocr_fusion_retrieval import (
    CURRENT_CANDIDATES_SHA256,
    CURRENT_RESULT_SHA256,
    EVALUATION_SHA256,
    OCR_BM25_K,
    OCR_DENSE_K,
    _deserialize_candidates,
    _load_search_representation,
    _merge_candidates,
    _ranks,
    _retrieve,
    is_arabic_query,
)
from reranker import create_reranker, score_documents


def _page_key(candidate: dict[str, Any]) -> tuple[str, int]:
    metadata = candidate["document"].metadata
    return str(metadata.get("source", "")).casefold(), int(metadata.get("page", -1))


def diversify_ranked_pages(
    ranked: list[tuple[dict[str, Any], float]],
) -> list[tuple[dict[str, Any], float]]:
    """Keep the highest-scored chunk for each source page, preserving score order."""
    seen: set[tuple[str, int]] = set()
    output = []
    for candidate, score in ranked:
        key = _page_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        output.append((candidate, score))
    return output


def _hit(rank: int | None, cutoff: int) -> bool:
    return rank is not None and rank <= cutoff


def _metrics(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    relevant = [record for record in records if record["relevant"]]
    ranks = [record[f"{arm}_page"] for record in relevant]
    return {
        "n": len(relevant),
        "exact_page_top1": sum(_hit(rank, 1) for rank in ranks) / len(ranks),
        "exact_page_top5": sum(_hit(rank, 5) for rank in ranks) / len(ranks),
        "exact_page_top20": sum(_hit(rank, 20) for rank in ranks) / len(ranks),
        "mrr_page": statistics.mean(1.0 / rank if rank is not None else 0.0 for rank in ranks),
    }


def _paired(records: list[dict[str, Any]], before: str, after: str, cutoff: int) -> dict[str, int]:
    relevant = [record for record in records if record["relevant"]]
    repairs = sum(
        not _hit(record[f"{before}_page"], cutoff)
        and _hit(record[f"{after}_page"], cutoff)
        for record in relevant
    )
    regressions = sum(
        _hit(record[f"{before}_page"], cutoff)
        and not _hit(record[f"{after}_page"], cutoff)
        for record in relevant
    )
    return {"repairs": repairs, "regressions": regressions, "net": repairs - regressions}


def summarize_diversity(records: list[dict[str, Any]]) -> dict[str, Any]:
    def group(language: str | None) -> dict[str, Any]:
        selected = [
            record for record in records if language is None or record["language"] == language
        ]
        return {
            arm: _metrics(selected, arm) for arm in ("current", "fusion", "diverse")
        } | {
            "diverse_vs_fusion_at_1": _paired(selected, "fusion", "diverse", 1),
            "diverse_vs_fusion_at_5": _paired(selected, "fusion", "diverse", 5),
            "diverse_vs_current_at_5": _paired(selected, "current", "diverse", 5),
        }

    return {
        "overall": group(None),
        "fr": group("fr"),
        "ar": group("ar"),
        "queries_with_page_duplicates": sum(
            record["duplicate_page_candidate_count"] > 0 for record in records
        ),
        "duplicate_page_candidates_removed": sum(
            record["duplicate_page_candidate_count"] for record in records
        ),
        "changed_page_ranks": [
            record["id"]
            for record in records
            if record["fusion_page"] != record["diverse_page"]
        ],
    }


def build_slim_result(full_result: dict[str, Any], *, full_result_sha256: str) -> dict[str, Any]:
    if full_result.get("status") != "complete":
        raise ValueError("Only a complete diversity result can be summarized")
    changed_ids = set(full_result["summary"]["changed_page_ranks"])
    return {
        key: full_result[key]
        for key in (
            "status",
            "timestamp",
            "experiment_id",
            "decision",
            "deployment_status",
            "hypothesis",
            "predeclared_gate",
            "configuration",
            "inputs",
            "summary",
            "latency_seconds",
            "limitations",
        )
    } | {
        "artifact_hashes": {"full_result_sha256": full_result_sha256},
        "changed_rank_records": [
            record for record in full_result["records"] if record["id"] in changed_ids
        ],
        "rank_records": {
            record["id"]: {
                "language": record["language"],
                "current_page": record["current_page"],
                "fusion_page": record["fusion_page"],
                "diverse_page": record["diverse_page"],
            }
            for record in full_result["records"]
        },
    }


def run_diversity_ablation(
    *,
    evaluation_path: Path,
    current_result_path: Path,
    current_candidates_path: Path,
    fusion_slim_result_path: Path,
    representation_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    for path, expected, label in (
        (evaluation_path, EVALUATION_SHA256, "evaluation"),
        (current_result_path, CURRENT_RESULT_SHA256, "current result"),
        (current_candidates_path, CURRENT_CANDIDATES_SHA256, "current candidates"),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen {label} hash differs: {actual}")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    current = json.loads(current_result_path.read_text(encoding="utf-8"))
    current_by_id = {record["id"]: record for record in current["records"]}
    candidates_by_id = json.loads(current_candidates_path.read_text(encoding="utf-8"))
    fusion = json.loads(fusion_slim_result_path.read_text(encoding="utf-8"))
    fusion_ranks = fusion["rank_records"]
    expected_ids = {case["id"] for case in evaluation}
    for values, label in (
        (set(current_by_id), "current result"),
        (set(candidates_by_id), "candidate cache"),
        (set(fusion_ranks), "fusion result"),
    ):
        if values != expected_ids:
            raise ValueError(f"{label} does not exactly cover evaluation IDs")
    if fusion["inputs"]["current_result_sha256"] != CURRENT_RESULT_SHA256:
        raise ValueError("Fusion result does not use the frozen current result")
    if fusion["inputs"]["current_candidates_sha256"] != CURRENT_CANDIDATES_SHA256:
        raise ValueError("Fusion result does not use the frozen candidate cache")
    if (
        fusion["inputs"]["ocr_representation_manifest_sha256"]
        != sha256_file(representation_manifest_path)
    ):
        raise ValueError("Fusion result and OCR representation manifest hashes differ")

    if output_path.exists():
        checkpoint = json.loads(output_path.read_text(encoding="utf-8"))
        expected_inputs = {
            "evaluation_sha256": sha256_file(evaluation_path),
            "current_result_sha256": sha256_file(current_result_path),
            "current_candidates_sha256": sha256_file(current_candidates_path),
            "fusion_slim_result_sha256": sha256_file(fusion_slim_result_path),
            "ocr_representation_manifest_sha256": sha256_file(
                representation_manifest_path
            ),
        }
        if checkpoint.get("status") == "complete":
            if checkpoint.get("inputs") != expected_inputs:
                raise ValueError("Complete diversity checkpoint input hashes differ")
            return checkpoint
        records = checkpoint.get("records", []) if checkpoint.get("status") == "partial" else []
    else:
        records = []
    completed = {record["id"] for record in records}
    representation = _load_search_representation(
        json.loads(representation_manifest_path.read_text(encoding="utf-8"))
    )
    reranker = create_reranker()

    for index, case in enumerate(evaluation, start=1):
        if case["id"] in completed:
            continue
        tracked = fusion_ranks[case["id"]]
        if not is_arabic_query(case["query"]):
            current_rank = tracked["current_page"]
            records.append(
                {
                    "id": case["id"],
                    "language": case.get("language"),
                    "relevant": bool(case["relevant"]),
                    "current_page": current_rank,
                    "fusion_page": tracked["fusion_page"],
                    "diverse_page": tracked["fusion_page"],
                    "candidate_count": len(candidates_by_id[case["id"]]["candidates"]),
                    "unique_page_count": len(candidates_by_id[case["id"]]["candidates"]),
                    "duplicate_page_candidate_count": 0,
                    "latency_seconds": 0.0,
                }
            )
            continue

        started = time.perf_counter()
        base = _deserialize_candidates(candidates_by_id[case["id"]])
        ocr = _retrieve(representation, case["query"], OCR_DENSE_K, OCR_BM25_K)
        candidates = _merge_candidates((base, ocr))
        scored = score_documents(
            reranker,
            case["query"],
            [candidate["document"] for candidate in candidates],
        )
        ranked = sorted(
            zip(candidates, (float(score) for _document, score in scored)),
            key=lambda item: item[1],
            reverse=True,
        )
        diverse = diversify_ranked_pages(ranked)
        fusion_page = diverse_page = None
        if case["relevant"]:
            _source_rank, fusion_page = _ranks(
                ranked, case["expected_source"], int(case["expected_page"])
            )
            _source_rank, diverse_page = _ranks(
                diverse, case["expected_source"], int(case["expected_page"])
            )
        if fusion_page != tracked["fusion_page"]:
            raise ValueError(
                f"Undiversified rank mismatch for {case['id']}: "
                f"rerun={fusion_page}, tracked={tracked['fusion_page']}"
            )
        if diverse_page is not None and fusion_page is not None and diverse_page > fusion_page:
            raise AssertionError(f"Page diversity worsened exact-page rank for {case['id']}")
        records.append(
            {
                "id": case["id"],
                "language": case.get("language"),
                "relevant": bool(case["relevant"]),
                "current_page": tracked["current_page"],
                "fusion_page": fusion_page,
                "diverse_page": diverse_page,
                "candidate_count": len(ranked),
                "unique_page_count": len(diverse),
                "duplicate_page_candidate_count": len(ranked) - len(diverse),
                "latency_seconds": time.perf_counter() - started,
            }
        )
        write_json_atomic(
            output_path,
            {
                "status": "partial",
                "configuration": {"page_key": "casefolded source plus start page"},
                "records": records,
            },
        )
        print(
            f"[diversity {index}/{len(evaluation)}] {case['id']} "
            f"rank={fusion_page}->{diverse_page}",
            flush=True,
        )

    records.sort(key=lambda record: record["id"])
    summary = summarize_diversity(records)
    gate_passed = all(
        summary[language][comparison]["regressions"] == 0
        for language in ("overall", "fr", "ar")
        for comparison in ("diverse_vs_fusion_at_1", "diverse_vs_fusion_at_5")
    ) and summary["overall"]["diverse_vs_fusion_at_5"]["repairs"] > 0
    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "post-reranker-source-page-diversity-development-v1",
        "decision": "KEEP_FOR_UNSEEN_VALIDATION" if gate_passed else "REJECT",
        "deployment_status": "PROHIBITED_PENDING_UNSEEN_VALIDATION",
        "hypothesis": (
            "Stable source-page diversity after reranking removes redundant native/OCR chunks "
            "from final slots without sacrificing exact-page recall."
        ),
        "predeclared_gate": (
            "No exact Page@1 or Page@5 regression overall or by language versus additive OCR, "
            "and at least one Page@5 repair."
        ),
        "configuration": {
            "candidate_pool": "identical additive Arabic OCR exact union",
            "reranker": "BAAI/bge-reranker-v2-m3 unchanged",
            "diversity": "stable first occurrence per casefolded source plus start page",
            "final_page_slots": 5,
        },
        "inputs": {
            "evaluation_sha256": sha256_file(evaluation_path),
            "current_result_sha256": sha256_file(current_result_path),
            "current_candidates_sha256": sha256_file(current_candidates_path),
            "fusion_slim_result_sha256": sha256_file(fusion_slim_result_path),
            "ocr_representation_manifest_sha256": sha256_file(
                representation_manifest_path
            ),
        },
        "summary": summary,
        "latency_seconds": {
            "mean_arabic_rerun": statistics.mean(
                record["latency_seconds"]
                for record in records
                if record["language"] == "ar"
            ),
            "diversity_postprocess": "included but not separately measurable; linear over ranked candidates",
        },
        "limitations": [
            "Development-only post-hoc ablation; this does not establish generalization.",
            "Page diversity improves citation-page coverage mechanically but may discard a second useful chunk from the same page.",
            "A generation context builder should retain provenance and may reattach additional same-page or neighboring context after page selection.",
        ],
        "records": records,
    }
    write_json_atomic(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--current-result", type=Path, required=True)
    parser.add_argument("--current-candidates", type=Path, required=True)
    parser.add_argument("--fusion-slim-result", type=Path, required=True)
    parser.add_argument("--representation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slim-output", type=Path)
    args = parser.parse_args()
    artifact = run_diversity_ablation(
        evaluation_path=args.evaluation,
        current_result_path=args.current_result,
        current_candidates_path=args.current_candidates,
        fusion_slim_result_path=args.fusion_slim_result,
        representation_manifest_path=args.representation_manifest,
        output_path=args.output,
    )
    if args.slim_output is not None:
        write_json_atomic(
            args.slim_output,
            build_slim_result(artifact, full_result_sha256=sha256_file(args.output)),
        )


if __name__ == "__main__":
    main()
