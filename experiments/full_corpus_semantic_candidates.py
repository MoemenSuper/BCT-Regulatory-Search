from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from time import perf_counter

from neo4j import GraphDatabase

from regulatory_graph.corpus_structure import inventory_corpus_cache
from regulatory_graph.neo4j_store import Neo4jRegulatoryGraph
from regulatory_graph.semantic_candidates import (
    CandidateType,
    extract_corpus_candidates,
    write_candidate_review_queue,
)


def run(
    *,
    documents_dir: Path,
    manifest_path: Path,
    chunks_path: Path,
    queue_path: Path,
    coverage_path: Path,
    result_path: Path,
    neo4j_uri: str,
) -> dict[str, object]:
    started = perf_counter()
    inventory = inventory_corpus_cache(documents_dir, manifest_path, chunks_path)
    with GraphDatabase.driver(neo4j_uri, auth=None) as driver:
        driver.verify_connectivity()
        graph = Neo4jRegulatoryGraph(driver)
        graph_before = graph.snapshot()
        candidates = extract_corpus_candidates(inventory)
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_receipt = write_candidate_review_queue(candidates, queue_path)
        with tempfile.TemporaryDirectory(prefix="bct-semantic-queue-") as directory:
            repeat_path = Path(directory) / "repeat.jsonl"
            repeat_receipt = write_candidate_review_queue(candidates, repeat_path)
            deterministic_repeat = (
                repeat_receipt == queue_receipt
                and repeat_path.read_bytes() == queue_path.read_bytes()
            )
        graph_after = graph.snapshot()

    source_pages = {
        (candidate.filename, candidate.page_number) for candidate in candidates
    }
    source_documents = {candidate.filename for candidate in candidates}
    all_pages = {
        (edition.filename, int(page["page_number"]))
        for edition in inventory.editions
        for page in edition.document["pages"]
    }
    all_documents = {edition.filename for edition in inventory.editions}
    type_counts = Counter(item.candidate_type.value for item in candidates)
    action_counts = Counter(
        item.proposed_action.value
        for item in candidates
        if item.proposed_action is not None
    )
    language_counts = Counter(item.language for item in candidates)
    offset_failures = 0
    page_text = {
        (edition.filename, int(page["page_number"])): str(page.get("raw_text", ""))
        for edition in inventory.editions
        for page in edition.document["pages"]
    }
    for item in candidates:
        text = page_text[(item.filename, item.page_number)]
        if (
            text[item.match_start:item.match_end] != item.signal
            or item.evidence_quote not in text
        ):
            offset_failures += 1

    legal_actions = type_counts[CandidateType.LEGAL_ACTION.value]
    candidate_counts_by_page = Counter(
        (item.filename, item.page_number) for item in candidates
    )
    candidate_counts_by_document = Counter(item.filename for item in candidates)
    pages_by_document = Counter(filename for filename, _ in all_pages)
    signal_pages_by_document = Counter(filename for filename, _ in source_pages)
    coverage = {
        "schema_version": "semantic-coverage-gaps-v1",
        "documents": [
            {
                "filename": filename,
                "candidate_count": candidate_counts_by_document[filename],
                "pages_total": pages_by_document[filename],
                "pages_with_signals": signal_pages_by_document[filename],
                "pages_without_signals": (
                    pages_by_document[filename] - signal_pages_by_document[filename]
                ),
            }
            for filename in sorted(all_documents, key=str.casefold)
        ],
        "documents_without_signals": sorted(
            all_documents - source_documents,
            key=str.casefold,
        ),
        "pages": [
            {
                "filename": filename,
                "page_number": page_number,
                "candidate_count": candidate_counts_by_page[(filename, page_number)],
            }
            for filename, page_number in sorted(
                all_pages,
                key=lambda item: (item[0].casefold(), item[1]),
            )
        ],
        "pages_without_signals": [
            {"filename": filename, "page_number": page_number}
            for filename, page_number in sorted(
                all_pages - source_pages,
                key=lambda item: (item[0].casefold(), item[1]),
            )
        ],
        "unmeasured_review_categories": {
            "broad_or_partial_replacements": "not_attempted",
            "conflicting_dates": "not_attempted",
            "malformed_arabic": "not_evaluated",
        },
    }
    coverage_bytes = (
        json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_bytes(coverage_bytes)
    coverage_sha256 = sha256(coverage_bytes).hexdigest().upper()
    result = {
        "experiment_id": "full-corpus-graph-checkpoint-c-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "hypothesis": (
            "A deterministic corpus-wide pass can create an exact-source semantic "
            "review queue without promoting unresolved legal relationships."
        ),
        "changed_variable": "deterministic_semantic_candidate_discovery_only",
        "configuration": {
            "documents_dir": str(documents_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "chunks_path": str(chunks_path.resolve()),
            "neo4j_uri": neo4j_uri,
            "runtime_gold_access": False,
            "hosted_calls": 0,
            "automatic_promotion": False,
        },
        "dataset_hashes": {
            "structured_manifest": inventory.manifest_sha256,
            "retained_chunks": inventory.chunks_sha256,
            "structured_artifact_aggregate": inventory.artifact_hash_aggregate,
            "candidate_review_queue": queue_receipt.content_sha256,
            "semantic_coverage_gaps": coverage_sha256,
        },
        "candidate_metrics": {
            "total": len(candidates),
            "by_type": dict(sorted(type_counts.items())),
            "by_action": dict(sorted(action_counts.items())),
            "by_language": dict(sorted(language_counts.items())),
            "documents_with_signals": len(source_documents),
            "documents_without_signals": len(all_documents - source_documents),
            "pages_with_signals": len(source_pages),
            "pages_without_signals": len(all_pages - source_pages),
            "verified": 0,
            "candidate": 0,
            "needs_review": len(candidates),
        },
        "failure_distribution": {
            "source_offset_or_quote_failures": offset_failures,
            "legal_action_target_resolution_not_attempted": legal_actions,
            "legal_action_visual_verification_not_attempted": legal_actions,
            "replacement_predecessor_verification_not_attempted": (
                action_counts.get("REPLACE", 0)
            ),
            "effective_date_resolution_not_attempted": legal_actions,
            "malformed_arabic": "not_evaluated",
            "conflicting_dates": "not_attempted",
            "broad_or_partial_replacements": "not_attempted",
        },
        "graph_safety": {
            "before": asdict(graph_before),
            "after": asdict(graph_after),
            "content_unchanged": graph_before == graph_after,
            "new_semantic_nodes_written": 0,
            "new_semantic_relationships_written": 0,
        },
        "verification": {
            "queue_deterministic_repeat": deterministic_repeat,
            "source_offsets_exact": offset_failures == 0,
            "all_new_candidates_need_review": all(
                item.verification_status.value == "NEEDS_REVIEW"
                for item in candidates
            ),
        },
        "latency": {"wall_seconds": perf_counter() - started},
        "approximate_cost": {
            "currency": "USD",
            "amount": 0.0,
            "hosted_calls": 0,
        },
        "decision": "REJECT_CHECKPOINT_C_INCOMPLETE_VERIFICATION",
        "conclusion": (
            "Preserve the exact-source review queue, but do not persist or use "
            "unresolved candidates as temporal graph facts."
        ),
        "validation_metrics": {
            "status": "not_run",
            "provisional_answer_validation": "unopened",
            "final_holdout": "unopened",
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:17687")
    args = parser.parse_args()
    result = run(
        documents_dir=args.documents_dir,
        manifest_path=args.manifest,
        chunks_path=args.chunks,
        queue_path=args.queue,
        coverage_path=args.coverage,
        result_path=args.result,
        neo4j_uri=args.neo4j_uri,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
