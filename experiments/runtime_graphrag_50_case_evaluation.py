"""Run the frozen 50-case slice through the current read-only GraphRAG runtime."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any

from langchain_core.documents import Document
from neo4j import GraphDatabase

from experiments.artifacts import sha256_file, write_json_atomic
from regulatory_graph.neo4j_store import Neo4jRegulatoryGraph
from regulatory_graph.runtime import (
    GraphRetrievalStatus,
    RegulatoryGraphRetriever,
    is_relationship_query,
    is_temporal_rule_query,
)


EXPERIMENT_ID = "runtime-graphrag-50-case-evaluation-v1"
SLICE_SHA256 = "F77A6B0F5727CB2FD8820A787AA4BBC2906A8E3FE012A872553729065A6E167E"
CONTROL_SHA256 = "57E605B3582A4D09164D538291F5F314E3EDA8E9F99C492B48578F6CF409EA86"
EXPECTED_CASE_COUNT = 50
FIXED_CURRENT_DATE = date(2026, 9, 1)


def _source_key(value: Any) -> str:
    return Path(str(value)).name.casefold()


def _page_pair(item: dict[str, Any]) -> tuple[str, int]:
    return (_source_key(item["source"]), int(item["page"]))


def _required_page_pairs(case: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (_source_key(item["source"]), int(page))
        for item in case.get("expected_sources", [])
        for page in item.get("pages", [])
    }


def _snapshot_dict(snapshot: Any) -> dict[str, Any]:
    return {
        "nodes": snapshot.nodes,
        "relationships": snapshot.relationships,
        "content_sha256": snapshot.content_sha256,
    }


def _runtime_inputs(
    cases: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    controls_by_id = {record["id"]: record for record in controls}
    if len(controls_by_id) != len(controls):
        raise ValueError("duplicate control record IDs")
    if {case["id"] for case in cases} != set(controls_by_id):
        raise ValueError("slice and control IDs differ")
    return [
        {
            "id": case["id"],
            "query": case["query"],
            "language": case.get("language"),
            "category": case.get("category"),
            "seeds": [
                {"source": item["source"], "page": int(item["page"])}
                for item in controls_by_id[case["id"]]["top20"][:5]
            ],
        }
        for case in cases
    ]


def _documents(seeds: list[dict[str, Any]]) -> tuple[Document, ...]:
    return tuple(
        Document(
            page_content="",
            metadata={
                "source": seed["source"],
                "page": int(seed["page"]) - 1,
                "page_label": int(seed["page"]),
            },
        )
        for seed in seeds
    )


def _document_record(document: Document) -> dict[str, Any]:
    metadata = dict(document.metadata)
    return {
        "source": Path(str(metadata.get("source", ""))).name,
        "page": int(metadata["page_label"]),
        "retrieval_source": metadata.get("retrieval_source"),
        "graph_path": metadata.get("graph_path"),
        "graph_relation": metadata.get("graph_relation"),
        "temporal_resolution": metadata.get("temporal_resolution"),
        "as_of": metadata.get("as_of"),
        "provision_uid": metadata.get("provision_uid"),
        "page_content": document.page_content,
    }


def _run_runtime(
    retriever: RegulatoryGraphRetriever,
    runtime_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outputs = []
    for case in runtime_inputs:
        started = time.perf_counter()
        result = retriever.retrieve(case["query"], _documents(case["seeds"]))
        latency = time.perf_counter() - started
        outputs.append(
            {
                "id": case["id"],
                "query": case["query"],
                "language": case["language"],
                "category": case["category"],
                "relationship_intent": is_relationship_query(case["query"]),
                "temporal_intent": is_temporal_rule_query(case["query"]),
                "seeds": case["seeds"],
                "latency_seconds": latency,
                "trace": result.trace.as_dict(),
                "requires_temporal_abstention": result.requires_temporal_abstention,
                "documents": [_document_record(document) for document in result.documents],
            }
        )
    return outputs


def _score(
    cases: list[dict[str, Any]],
    runtime_outputs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = {case["id"]: case for case in cases}
    scored = []
    for output in runtime_outputs:
        required = _required_page_pairs(gold_by_id[output["id"]])
        seeds = {_page_pair(item) for item in output["seeds"]}
        graph = {_page_pair(item) for item in output["documents"]}
        missing_before = required - seeds
        found_by_graph = missing_before & graph
        scored.append(
            {
                "id": output["id"],
                "required_page_pair_count": len(required),
                "ordinary_top5_required_pair_recall": (
                    len(required & seeds) / len(required) if required else 1.0
                ),
                "graph_required_pair_recall": (
                    len(required & graph) / len(required) if required else 1.0
                ),
                "augmented_required_pair_recall": (
                    len(required & (seeds | graph)) / len(required) if required else 1.0
                ),
                "ordinary_top5_complete": required <= seeds,
                "augmented_complete": required <= (seeds | graph),
                "missing_required_pairs_added_by_graph": [
                    {"source": source, "page": page}
                    for source, page in sorted(found_by_graph)
                ],
            }
        )

    status_counts = Counter(output["trace"]["status"] for output in runtime_outputs)
    temporal_counts = Counter(
        output["trace"]["temporal_status"] for output in runtime_outputs
    )
    reason_counts = Counter(
        output["trace"]["temporal_reason"]
        for output in runtime_outputs
        if output["trace"]["temporal_reason"] is not None
    )
    relationship_routes = sum(output["relationship_intent"] for output in runtime_outputs)
    temporal_routes = sum(output["temporal_intent"] for output in runtime_outputs)
    routed = sum(
        output["trace"]["status"] != GraphRetrievalStatus.NOT_REQUESTED.value
        for output in runtime_outputs
    )
    evidence_cases = sum(bool(output["documents"]) for output in runtime_outputs)
    metrics = {
        "case_count": len(runtime_outputs),
        "relationship_intent_count": relationship_routes,
        "temporal_intent_count": temporal_routes,
        "routed_case_count": routed,
        "routing_coverage": routed / len(runtime_outputs),
        "graph_status_counts": dict(sorted(status_counts.items())),
        "temporal_status_counts": dict(sorted(temporal_counts.items())),
        "temporal_reason_counts": dict(sorted(reason_counts.items())),
        "graph_evidence_case_count": evidence_cases,
        "graph_evidence_coverage": evidence_cases / len(runtime_outputs),
        "temporal_abstention_count": sum(
            output["requires_temporal_abstention"] for output in runtime_outputs
        ),
        "ordinary_top5_complete_required_pairs": sum(
            record["ordinary_top5_complete"] for record in scored
        ),
        "augmented_complete_required_pairs": sum(
            record["augmented_complete"] for record in scored
        ),
        "complete_pair_repairs_from_graph": sum(
            not record["ordinary_top5_complete"] and record["augmented_complete"]
            for record in scored
        ),
        "cases_with_missing_required_pair_added_by_graph": sum(
            bool(record["missing_required_pairs_added_by_graph"])
            for record in scored
        ),
        "mean_graph_latency_seconds": statistics.mean(
            output["latency_seconds"] for output in runtime_outputs
        ),
        "median_graph_latency_seconds": statistics.median(
            output["latency_seconds"] for output in runtime_outputs
        ),
        "max_graph_latency_seconds": max(
            output["latency_seconds"] for output in runtime_outputs
        ),
    }
    return metrics, scored


def run(
    *,
    slice_path: Path,
    control_path: Path,
    runtime_output_path: Path,
    result_path: Path,
    neo4j_uri: str,
) -> dict[str, Any]:
    if sha256_file(slice_path) != SLICE_SHA256:
        raise ValueError("frozen 50-case slice hash differs")
    if sha256_file(control_path) != CONTROL_SHA256:
        raise ValueError("frozen control result hash differs")

    slice_data = json.loads(slice_path.read_text(encoding="utf-8"))
    control_data = json.loads(control_path.read_text(encoding="utf-8"))
    cases = slice_data["cases"]
    controls = control_data["records"]
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"expected {EXPECTED_CASE_COUNT} cases, found {len(cases)}")
    runtime_inputs = _runtime_inputs(cases, controls)

    driver = GraphDatabase.driver(neo4j_uri, auth=None, connection_timeout=2.0)
    try:
        driver.verify_connectivity()
        graph = Neo4jRegulatoryGraph(driver)
        before = graph.snapshot()
        runtime_outputs = _run_runtime(
            RegulatoryGraphRetriever(graph, current_date=FIXED_CURRENT_DATE),
            runtime_inputs,
        )
        runtime_artifact = {
            "status": "frozen_before_scoring",
            "experiment_id": EXPERIMENT_ID,
            "runtime_gold_access": False,
            "fixed_current_date": FIXED_CURRENT_DATE.isoformat(),
            "case_count": len(runtime_outputs),
            "records": runtime_outputs,
        }
        write_json_atomic(runtime_output_path, runtime_artifact)

        metrics, scored = _score(cases, runtime_outputs)
        after = graph.snapshot()
    finally:
        driver.close()

    before_data = _snapshot_dict(before)
    after_data = _snapshot_dict(after)
    unchanged = before_data == after_data
    result = {
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "decision": (
            "CURRENT_GRAPH_INVENTORY_DOES_NOT_COVER_50_CASE_SLICE"
            if metrics["graph_evidence_case_count"] == 0
            else "CURRENT_GRAPH_INVENTORY_PARTIALLY_COVERS_50_CASE_SLICE"
        ),
        "inputs": {
            "slice_sha256": sha256_file(slice_path),
            "control_sha256": sha256_file(control_path),
            "runtime_output_sha256": sha256_file(runtime_output_path),
            "runtime_gold_access": False,
            "hosted_calls": 0,
            "answer_calls": 0,
            "ordinary_seed_count": 5,
            "fixed_current_date": FIXED_CURRENT_DATE.isoformat(),
        },
        "metrics": metrics,
        "persistent_graph": {
            "before": before_data,
            "after": after_data,
            "unchanged": unchanged,
        },
        "records": scored,
        "limitations": [
            "This measures graph routing and verified graph evidence retrieval, not final answer correctness.",
            "Ordinary retrieval seeds were reused from the prior frozen benchmark; embeddings and reranking were not rerun.",
            "The graph layer can return only verified relationships and complete verified temporal lineages already persisted in Neo4j.",
        ],
    }
    if not unchanged:
        result["status"] = "failed"
        result["decision"] = "FAIL_PERSISTENT_GRAPH_CHANGED"
    write_json_atomic(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slice",
        type=Path,
        default=Path("experiments/stress_suites/graph_temporal_evaluation_v1.json"),
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=Path("experiments/results/graph_temporal_enriched_benchmark_v1.json"),
    )
    parser.add_argument(
        "--runtime-output",
        type=Path,
        default=Path("experiments/artifacts/runtime_graphrag_50_case_outputs_v1.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("experiments/results/runtime_graphrag_50_case_evaluation_v1.json"),
    )
    parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:17687")
    args = parser.parse_args()
    result = run(
        slice_path=args.slice,
        control_path=args.control,
        runtime_output_path=args.runtime_output,
        result_path=args.result,
        neo4j_uri=args.neo4j_uri,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(json.dumps(result["persistent_graph"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
