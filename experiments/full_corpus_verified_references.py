"""Verify explicit corpus citations and optionally backfill them into Neo4j."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from neo4j import GraphDatabase

from experiments.artifacts import sha256_file, write_json_atomic
from regulatory_graph.corpus_structure import (
    build_structural_bundle,
    inventory_corpus_cache,
)
from regulatory_graph.neo4j_store import Neo4jGraphWriter, Neo4jRegulatoryGraph
from regulatory_graph.reference_ingestion import (
    ReferencePage,
    VerifiedInstrumentCatalog,
    VerifiedReferencePromotion,
    enrich_bundle_with_verified_references,
    extract_document_reference_candidates,
    verify_reference_candidate,
)


EXPERIMENT_ID = "full-corpus-verified-references-v1"
PERSISTENT_URI = "bolt://127.0.0.1:17687"
PERSISTENT_CONFIRMATION = "ADD_VERIFIED_REFERENCES"


def _snapshot(snapshot: Any) -> dict[str, Any]:
    return {
        "nodes": snapshot.nodes,
        "relationships": snapshot.relationships,
        "content_sha256": snapshot.content_sha256,
    }


def _reference_pages(cached: Any) -> tuple[ReferencePage, ...]:
    return tuple(
        ReferencePage(
            page_number=int(page["page_number"]),
            text=str(page.get("raw_text", "")),
            extraction_method=str(page.get("extraction_method") or "structured"),
        )
        for page in cached.document["pages"]
    )


def run(
    *,
    documents_dir: Path,
    manifest_path: Path,
    chunks_path: Path,
    review_output_path: Path,
    result_path: Path,
    neo4j_uri: str,
    write_graph: bool,
    persistent_confirmation: str | None,
) -> dict[str, Any]:
    started = perf_counter()
    resolved_uri = neo4j_uri.strip()
    if write_graph and resolved_uri == PERSISTENT_URI:
        if persistent_confirmation != PERSISTENT_CONFIRMATION:
            raise ValueError(
                "persistent reference backfill requires explicit confirmation"
            )

    inventory = inventory_corpus_cache(
        documents_dir,
        manifest_path,
        chunks_path,
    )
    structural_bundle = build_structural_bundle(inventory)
    catalog = VerifiedInstrumentCatalog.from_bundle(structural_bundle)
    editions_by_filename = {
        edition.filename: edition for edition in structural_bundle.source_editions
    }

    candidates = []
    decisions = []
    review_rows = []
    for cached in inventory.editions:
        edition = editions_by_filename[cached.filename]
        pdf_path = documents_dir / cached.relative_path
        document_candidates = extract_document_reference_candidates(
            edition,
            _reference_pages(cached),
            instrument_catalog=catalog,
        )
        candidates.extend(document_candidates)
        for candidate in document_candidates:
            decision = verify_reference_candidate(
                candidate,
                source_pdf_path=pdf_path,
                instrument_catalog=catalog,
            )
            decisions.append(decision)
            review_rows.append(
                {
                    "candidate_uid": candidate.uid,
                    "source_filename": candidate.source_filename,
                    "page_number": candidate.page_number,
                    "target_instrument_uid": candidate.target_instrument_uid,
                    "target_corpus_present": candidate.target_corpus_present,
                    "status": decision.status.value,
                    "reasons": list(decision.reasons),
                }
            )

    verified = tuple(
        decision
        for decision in decisions
        if isinstance(decision, VerifiedReferencePromotion)
    )
    enriched_bundle = enrich_bundle_with_verified_references(
        structural_bundle,
        verified,
    )
    review_output_path.parent.mkdir(parents=True, exist_ok=True)
    review_output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in review_rows
        ),
        encoding="utf-8",
    )

    graph_data: dict[str, Any] = {"write_requested": write_graph}
    if write_graph:
        with GraphDatabase.driver(resolved_uri, auth=None) as driver:
            driver.verify_connectivity()
            graph = Neo4jRegulatoryGraph(driver)
            before = graph.snapshot()
            writer = Neo4jGraphWriter(driver)
            first_receipt = writer.write_bundle(enriched_bundle)
            after_first = graph.snapshot()
            second_receipt = writer.write_bundle(enriched_bundle)
            after_second = graph.snapshot()
            filename_by_edition = {
                edition.uid: edition.filename
                for edition in structural_bundle.source_editions
            }
            retrieval_seed_filename = None
            evidence = ()
            if verified:
                seed_edition_uid = verified[0].evidence_span.source_edition_uid
                retrieval_seed_filename = filename_by_edition[seed_edition_uid]
                evidence = graph.relationship_evidence(
                    (retrieval_seed_filename,),
                    limit=10,
                )
        graph_data = {
            "write_requested": True,
            "before": _snapshot(before),
            "after_first": _snapshot(after_first),
            "after_second": _snapshot(after_second),
            "idempotent_repeat": after_first == after_second,
            "first_write_bundle_sha256": first_receipt.bundle_sha256,
            "second_write_bundle_sha256": second_receipt.bundle_sha256,
            "retrieval_seed_filename": retrieval_seed_filename,
            "retrieval_evidence_count": len(evidence),
        }

    verified_source_documents = {
        row["source_filename"]
        for row in review_rows
        if row["status"] == "VERIFIED"
    }
    reason_counts = Counter(
        reason
        for row in review_rows
        for reason in row["reasons"]
    )
    metrics = {
        "candidate_count": len(candidates),
        "verified_reference_count": len(verified),
        "needs_review_reference_count": len(candidates) - len(verified),
        "documents_with_candidates": len(
            {candidate.source_filename for candidate in candidates}
        ),
        "documents_with_verified_references": len(verified_source_documents),
        "verified_local_target_count": sum(
            decision.target_instrument.corpus_present for decision in verified
        ),
        "verified_external_target_count": sum(
            not decision.target_instrument.corpus_present for decision in verified
        ),
        "needs_review_reason_counts": dict(sorted(reason_counts.items())),
    }
    gates = {
        "verified_documents_at_least_100": (
            metrics["documents_with_verified_references"] >= 100
        ),
        "only_verified_decisions_written": all(
            reference.verification_status.value == "VERIFIED"
            for reference in enriched_bundle.instrument_references
        ),
        "write_idempotent": (
            not write_graph or bool(graph_data["idempotent_repeat"])
        ),
        "verified_relationship_retrievable": (
            not write_graph or graph_data["retrieval_evidence_count"] > 0
        ),
    }
    result = {
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "configuration": {
            "documents_dir": str(documents_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "chunks_path": str(chunks_path.resolve()),
            "neo4j_uri": resolved_uri,
            "write_graph": write_graph,
            "runtime_gold_access": False,
            "hosted_calls": 0,
            "automatic_legal_effect_promotion": False,
        },
        "input_hashes": {
            "manifest": inventory.manifest_sha256,
            "chunks": inventory.chunks_sha256,
            "artifact_aggregate": inventory.artifact_hash_aggregate,
            "review_output": sha256_file(review_output_path),
        },
        "metrics": metrics,
        "graph": graph_data,
        "gates": gates,
        "decision": (
            "KEEP_CONNECTED_REFERENCE_GRAPH"
            if all(gates.values())
            else "REJECT_CONNECTED_REFERENCE_GRAPH"
        ),
        "latency_seconds": perf_counter() - started,
        "limitations": [
            "Verified CITES relationships do not imply amendment, replacement, or abrogation.",
            "Scanned or extraction-divergent references remain NEEDS_REVIEW.",
            "Temporal legal-effect lineage remains separately review-gated.",
        ],
    }
    write_json_atomic(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--neo4j-uri", default=PERSISTENT_URI)
    parser.add_argument("--write-graph", action="store_true")
    parser.add_argument("--confirm-persistent-write")
    args = parser.parse_args()
    result = run(
        documents_dir=args.documents_dir,
        manifest_path=args.manifest,
        chunks_path=args.chunks,
        review_output_path=args.review_output,
        result_path=args.result,
        neo4j_uri=args.neo4j_uri,
        write_graph=args.write_graph,
        persistent_confirmation=args.confirm_persistent_write,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
