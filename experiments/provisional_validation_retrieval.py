"""Run the development-selected retrieval configuration once on frozen validation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.evaluation_protocol import _load_dataset
from experiments.retrieval_ablations import (
    BM25_K,
    DENSE_K,
    _load_search_representation,
    _merge_candidates,
    _metrics,
    _page_matches,
    _ranks,
    _retrieve,
)
from reranker import create_reranker, score_documents


NATIVE_MANIFEST_SHA256 = "920159F84E87353256899370A2E8B7854BAB87FBBBDC05184FED7C3657B439C1"
OCR_MANIFEST_SHA256 = "8CBD9A468B2D017DB46218252B555231460A64381245786CCA020C84F3CD05AB"
OCR_DENSE_K = 5
OCR_BM25_K = 5


def is_arabic_query(query: str) -> bool:
    return any(
        "\u0600" <= character <= "\u06ff"
        or "\u0750" <= character <= "\u077f"
        or "\u08a0" <= character <= "\u08ff"
        for character in query
    )


def diversify_ranked_pages(
    ranked: list[tuple[dict[str, Any], float]],
) -> list[tuple[dict[str, Any], float]]:
    seen: set[tuple[str, int]] = set()
    output = []
    for candidate, score in ranked:
        metadata = candidate["document"].metadata
        key = (
            str(metadata.get("source", "")).casefold(),
            int(metadata.get("page", -1)),
        )
        if key not in seen:
            seen.add(key)
            output.append((candidate, score))
    return output


def load_frozen_validation(
    *, protocol_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    try:
        frozen = protocol["sets"]["validation"]
    except (KeyError, TypeError) as error:
        raise ValueError("Protocol does not define frozen validation") from error
    validation_path = Path(frozen["path"])
    if sha256_file(validation_path) != frozen["sha256"]:
        raise ValueError("Frozen validation dataset hash does not match the protocol")
    return _load_dataset(validation_path), frozen


def _failure_categories(
    case: dict[str, Any], candidates: list[dict[str, Any]], exact_page_rank: int | None
) -> list[str]:
    if not case["relevant"] or (
        exact_page_rank is not None and exact_page_rank <= 5
    ):
        return []
    source_matches = [
        candidate
        for candidate in candidates
        if str(candidate["document"].metadata.get("source", "")).casefold()
        == str(case["expected_source"]).casefold()
    ]
    if not source_matches:
        return ["correct_document_missing_from_candidate_set"]
    if not any(
        _page_matches(candidate["document"].metadata, int(case["expected_page"]))
        for candidate in source_matches
    ):
        return ["correct_page_missing_from_candidate_set"]
    return ["correct_page_reranked_below_top_5"]


def _top_item(candidate: dict[str, Any], score: float) -> dict[str, Any]:
    metadata = candidate["document"].metadata
    return {
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "page_end": metadata.get("page_end", metadata.get("page")),
        "representations": sorted(candidate["representations"]),
        "reranker_score": score,
    }


def run_provisional_validation(
    *,
    protocol_path: Path,
    native_manifest_path: Path,
    ocr_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if sha256_file(native_manifest_path) != NATIVE_MANIFEST_SHA256:
        raise ValueError("Native representation manifest differs from the selected control")
    if sha256_file(ocr_manifest_path) != OCR_MANIFEST_SHA256:
        raise ValueError("OCR representation manifest differs from the selected control")
    cases, frozen = load_frozen_validation(protocol_path=protocol_path)
    native = _load_search_representation(
        json.loads(native_manifest_path.read_text(encoding="utf-8"))
    )
    ocr = _load_search_representation(
        json.loads(ocr_manifest_path.read_text(encoding="utf-8"))
    )
    reranker = create_reranker()
    records = []
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        native_candidates = _retrieve(native, case["query"], DENSE_K, BM25_K)
        ocr_candidates = (
            _retrieve(ocr, case["query"], OCR_DENSE_K, OCR_BM25_K)
            if is_arabic_query(case["query"])
            else []
        )
        candidates = _merge_candidates((native_candidates, ocr_candidates))
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
        source_rank = exact_page_rank = None
        if case["relevant"]:
            source_rank, exact_page_rank = _ranks(
                diverse, case["expected_source"], int(case["expected_page"])
            )
        failure_categories = _failure_categories(case, candidates, exact_page_rank)
        records.append(
            {
                "id": case["id"],
                "language": case["language"],
                "category": case["category"],
                "relevant": case["relevant"],
                "result": {
                    "source_rank": source_rank,
                    "exact_page_rank": exact_page_rank,
                    "top5": [
                        _top_item(candidate, score) for candidate, score in diverse[:5]
                    ],
                },
                "candidate_count": len(candidates),
                "unique_page_count": len(diverse),
                "ocr_candidate_count": len(ocr_candidates),
                "failure_categories": failure_categories,
                "latency_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"[validation {index}/{len(cases)}] {case['id']} page={exact_page_rank}",
            flush=True,
        )
    latency = [record["latency_seconds"] for record in records]
    artifact = {
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_role": "provisional_validation",
        "decision_policy": "observe_once_and_do_not_tune_on_individual_failures",
        "configuration": {
            "native_candidate_union": {"dense_k": DENSE_K, "bm25_k": BM25_K},
            "arabic_additive_ocr": {
                "routed_by_arabic_script": True,
                "dense_k": OCR_DENSE_K,
                "bm25_k": OCR_BM25_K,
            },
            "reranker": "BAAI/bge-reranker-v2-m3",
            "post_reranker_diversity": "stable first occurrence per source page",
            "final_page_slots": 5,
        },
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "validation_sha256": frozen["sha256"],
            "native_manifest_sha256": sha256_file(native_manifest_path),
            "ocr_manifest_sha256": sha256_file(ocr_manifest_path),
        },
        "summary": {
            "overall": _metrics(records, "result"),
            "fr": _metrics(
                [record for record in records if record["language"] == "fr"],
                "result",
            ),
            "ar": _metrics(
                [record for record in records if record["language"] == "ar"],
                "result",
            ),
            "failure_category_counts": {
                category: sum(
                    category in record["failure_categories"] for record in records
                )
                for category in sorted(
                    {
                        category
                        for record in records
                        for category in record["failure_categories"]
                    }
                )
            },
            "latency_seconds": {
                "mean": statistics.mean(latency),
                "median": statistics.median(latency),
            },
        },
        "limitations": [
            "This is a page-disjoint but source/family-overlapping provisional validation set.",
            "The labels have a blind second-agent source review, not independent human adjudication.",
            "This retrieval-only run does not measure answer, citation, grounding, or abstention quality.",
            "No final family-disjoint holdout is currently available.",
        ],
        "records": records,
    }
    write_json_atomic(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--ocr-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_provisional_validation(
        protocol_path=args.protocol,
        native_manifest_path=args.native_manifest,
        ocr_manifest_path=args.ocr_manifest,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
