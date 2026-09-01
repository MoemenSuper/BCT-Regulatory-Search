"""Run the audited enriched GraphRAG slice through retrieval, Neo4j, and answers."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from langchain_core.documents import Document
from neo4j import GraphDatabase

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.document_identity_candidate_experiment import ranked_signature
from experiments.gold_evidence_answer_experiment import automatic_answer_audit
from experiments.graph_temporal_benchmark import _candidate_pages, _rank, score_case
from experiments.ocr_fusion_retrieval import _merge_candidates, _retrieve, is_arabic_query
from experiments.provisional_validation_retrieval import (
    NATIVE_MANIFEST_SHA256,
    OCR_BM25_K,
    OCR_DENSE_K,
    OCR_MANIFEST_SHA256,
)
from experiments.retrieval_ablations import BM25_K, DENSE_K, _load_search_representation
from experiments.retrieved_context_answer_experiment import _evidence_payload
from experiments.structured_answer_experiment import (
    MODEL,
    _SCHEMA_V2,
    _SYSTEM_V5,
    _usage,
    parse_structured_answer,
)
from regulatory_graph.neo4j_store import Neo4jRegulatoryGraph
from regulatory_graph.runtime import RegulatoryGraphRetriever
from reranker import create_reranker


EXPERIMENT_ID = "enriched-graphrag-end-to-end-v1"
SUITE_SHA256 = "F77A6B0F5727CB2FD8820A787AA4BBC2906A8E3FE012A872553729065A6E167E"
EXPECTED_CASE_COUNT = 50
PROMPT_VERSION = "bct-enriched-graphrag-answer-v1-structured-v5"
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 1536
FIXED_CURRENT_DATE = date(2026, 9, 1)


def runtime_inputs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: case.get(key)
            for key in ("id", "query", "language", "category")
        }
        for case in cases
    ]


def _load_cases(suite: Any) -> list[dict[str, Any]]:
    if isinstance(suite, list):
        cases = suite
    elif isinstance(suite, dict):
        cases = suite.get("cases")
    else:
        cases = None
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("Suite must be a case list or an object with a case list")
    return cases


def _expected_page_pairs(case: dict[str, Any]) -> set[tuple[str, int]]:
    expected_sources = case.get("expected_sources") or []
    if expected_sources:
        return {
            _pair(item["source"], page)
            for item in expected_sources
            for page in item.get("pages", [])
        }
    if (
        case.get("relevant", True)
        and case.get("expected_source") is not None
        and case.get("expected_page") is not None
    ):
        return {_pair(case["expected_source"], case["expected_page"])}
    return set()


def graph_candidate(document: Document) -> dict[str, Any]:
    metadata = dict(document.metadata)
    page = int(metadata.get("page_label", int(metadata.get("page", -1)) + 1))
    metadata.update({"page": page, "page_end": page, "pages": [page], "page_label": page})
    representation = str(metadata.get("retrieval_source", "neo4j"))
    return {
        "document": Document(page_content=document.page_content, metadata=metadata),
        "representations": {representation},
        "ranks": {},
    }


def seed_documents(candidates: list[dict[str, Any]]) -> tuple[Document, ...]:
    seeds = []
    for candidate in candidates:
        document = candidate["document"]
        metadata = dict(document.metadata)
        page = int(metadata["page"])
        metadata.update({"page": page - 1, "page_label": page})
        seeds.append(Document(page_content=document.page_content, metadata=metadata))
    return tuple(seeds)


def _pair(source: Any, page: Any) -> tuple[str, int]:
    return (Path(str(source)).name.casefold(), int(page))


def structured_diagnostics(
    case: dict[str, Any],
    response: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    citations_match = all(
        citation["evidence_id"] in evidence_by_id
        and _pair(citation["source"], citation["page"])
        == _pair(
            evidence_by_id[citation["evidence_id"]]["source"],
            evidence_by_id[citation["evidence_id"]]["page"],
        )
        for citation in response["citations"]
    )
    claim_links = (
        bool(response["claims"])
        and all(
            bool(claim["evidence_ids"])
            and set(claim["evidence_ids"]).issubset(evidence_by_id)
            for claim in response["claims"]
        )
        if response["status"] == "answered"
        else not response["claims"]
    )
    expected_pairs = _expected_page_pairs(case)
    cited_pairs = {
        _pair(item["source"], item["page"])
        for item in response["citations"]
    }
    if case.get("relevant", True):
        status_expected = response["status"] == "answered"
    else:
        expected_status = {
            "clarify": "clarification_needed",
            "reject_out_of_scope": "out_of_scope",
            "abstain": "insufficient_evidence",
        }.get(case.get("expected_behavior"))
        status_expected = response["status"] == expected_status
    return {
        "answered": response["status"] == "answered",
        "status_expected": status_expected,
        "citations_match_evidence": citations_match,
        "claim_evidence_links_valid": claim_links,
        "complete_required_page_citations": expected_pairs <= cited_pairs,
        "required_page_citation_recall": (
            len(expected_pairs & cited_pairs) / len(expected_pairs)
            if expected_pairs
            else 1.0
        ),
        "claim_count": len(response["claims"]),
        "citation_count": len(response["citations"]),
    }


def _temporal_abstention(language: str) -> dict[str, Any]:
    answer = (
        "لا يمكن تحديد الحكم المنطبق بأمان لأن التسلسل الزمني الموثق غير مكتمل أو التاريخ المطلوب غير محدد."
        if language == "ar"
        else "Je ne peux pas déterminer la disposition applicable de manière fiable car la chronologie vérifiée est incomplète ou la date nécessaire n'est pas précisée."
    )
    return {
        "status": "insufficient_evidence",
        "answer": answer,
        "claims": [],
        "citations": [],
    }


def _answer_ranked(ranked: list[tuple[dict[str, Any], float]]) -> list[tuple[dict[str, Any], float]]:
    temporal = [
        item
        for item in ranked
        if item[0]["document"].metadata.get("temporal_resolution") == "VERIFIED"
    ]
    if not temporal:
        return ranked[:5]
    mandatory = temporal[0]
    return [mandatory, *(item for item in ranked if item is not mandatory)][:5]


def _rate_limit_wait_seconds(error: Any) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) if response is not None else {}
    try:
        retry_after = float(headers.get("retry-after", 60.0))
    except (TypeError, ValueError):
        retry_after = 60.0
    return max(retry_after, 1.0)


def _answer_with_cache(
    *,
    client: Groq,
    case_id: str,
    query: str,
    evidence: list[dict[str, Any]],
    suite_hash: str,
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, int], bool]:
    payload = json.dumps(
        {
            "question": query,
            "evidence": [
                {key: item[key] for key in ("evidence_id", "source", "page", "text")}
                for item in evidence
            ],
        },
        ensure_ascii=False,
    )
    cache_path = cache_dir / f"{case_id}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        binding = (cached.get("model"), cached.get("prompt_version"), cached.get("suite_sha256"), cached.get("user_payload"))
        if binding != (MODEL, PROMPT_VERSION, suite_hash, payload):
            raise ValueError(f"Answer cache binding differs for {case_id}")
        response = parse_structured_answer(
            json.dumps(cached["response"], ensure_ascii=False),
            require_nonempty_answer=True,
        )
        return response, cached["usage"], True

    for attempt in range(24):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_V5},
                    {"role": "user", "content": payload},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bct_enriched_graphrag_answer",
                        "strict": True,
                        "schema": _SCHEMA_V2,
                    },
                },
                reasoning_effort=REASONING_EFFORT,
                temperature=0,
                seed=20260901,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            break
        except RateLimitError as error:
            if attempt == 23:
                raise
            wait = _rate_limit_wait_seconds(error)
            print(f"[rate-limit] waiting {wait:.1f}s", flush=True)
            time.sleep(wait)

    response = parse_structured_answer(
        completion.choices[0].message.content or "",
        require_nonempty_answer=True,
    )
    usage = _usage(completion.usage)
    write_json_atomic(
        cache_path,
        {
            "id": case_id,
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "suite_sha256": suite_hash,
            "user_payload": payload,
            "response_id": completion.id,
            "usage": usage,
            "response": response,
        },
    )
    return response, usage, False


def _snapshot(snapshot: Any) -> dict[str, Any]:
    return {
        "nodes": snapshot.nodes,
        "relationships": snapshot.relationships,
        "content_sha256": snapshot.content_sha256,
    }


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"case_count": 0}
    relevant = [record for record in records if record["relevant"]]
    negative = [record for record in records if not record["relevant"]]
    relevant_count = len(relevant)
    return {
        "case_count": len(records),
        "relevant_count": relevant_count,
        "negative_count": len(negative),
        "answered_rate": (
            sum(r["diagnostics"]["answered"] for r in relevant) / relevant_count
            if relevant_count
            else None
        ),
        "complete_required_page_retrieval_at_5": (
            sum(r["retrieval"]["complete_required_page_pairs_at_5"] for r in relevant) / relevant_count
            if relevant_count
            else None
        ),
        "complete_required_page_citation_rate": (
            sum(r["diagnostics"]["complete_required_page_citations"] for r in relevant) / relevant_count
            if relevant_count
            else None
        ),
        "mean_required_page_citation_recall": (
            statistics.mean(r["diagnostics"]["required_page_citation_recall"] for r in relevant)
            if relevant_count
            else None
        ),
        "citation_to_evidence_integrity_rate": sum(r["diagnostics"]["citations_match_evidence"] for r in records) / len(records),
        "claim_to_evidence_integrity_rate": sum(r["diagnostics"]["claim_evidence_links_valid"] for r in records) / len(records),
        "negative_expected_status_rate": (
            sum(r["diagnostics"]["status_expected"] for r in negative) / len(negative)
            if negative
            else None
        ),
        "graph_expansion_rate": sum(r["graph_status"] == "EXPANDED" for r in records) / len(records),
        "temporal_abstention_count": sum(r["answer_path"] == "safe_temporal_abstention" for r in records),
        "graph_complete_pair_repairs": sum(
            not r["ordinary_retrieval"]["complete_required_page_pairs_at_5"]
            and r["retrieval"]["complete_required_page_pairs_at_5"]
            for r in relevant
        ),
    }


def run(
    *,
    suite_path: Path,
    native_manifest_path: Path,
    ocr_manifest_path: Path,
    dotenv_path: Path,
    cache_dir: Path,
    runtime_output_path: Path,
    result_path: Path,
    neo4j_uri: str,
    confirm_public_documents: bool,
    expected_suite_sha256: str = SUITE_SHA256,
    expected_case_count: int = EXPECTED_CASE_COUNT,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    if not confirm_public_documents:
        raise ValueError("Public-document confirmation is required")
    for path, expected, label in (
        (suite_path, expected_suite_sha256, "suite"),
        (native_manifest_path, NATIVE_MANIFEST_SHA256, "native manifest"),
        (ocr_manifest_path, OCR_MANIFEST_SHA256, "OCR manifest"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen {label} hash differs")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured")

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = _load_cases(suite)
    if len(cases) != expected_case_count:
        raise ValueError(f"Expected {expected_case_count} cases")
    inputs = runtime_inputs(cases)
    native = _load_search_representation(json.loads(native_manifest_path.read_text(encoding="utf-8")))
    ocr = _load_search_representation(json.loads(ocr_manifest_path.read_text(encoding="utf-8")))
    reranker = create_reranker()
    client = Groq(max_retries=3, timeout=90.0)
    cache_dir.mkdir(parents=True, exist_ok=True)

    driver = GraphDatabase.driver(neo4j_uri, auth=None, connection_timeout=2.0)
    runtime_records = []
    try:
        driver.verify_connectivity()
        graph = Neo4jRegulatoryGraph(driver)
        retriever = RegulatoryGraphRetriever(graph, current_date=FIXED_CURRENT_DATE)
        before = _snapshot(graph.snapshot())
        for index, case in enumerate(inputs, start=1):
            started = time.perf_counter()
            native_candidates = _retrieve(native, case["query"], DENSE_K, BM25_K)
            ocr_candidates = _retrieve(ocr, case["query"], OCR_DENSE_K, OCR_BM25_K) if is_arabic_query(case["query"]) else []
            ordinary_candidates = _merge_candidates((native_candidates, ocr_candidates))
            ordinary_ranked = _rank(reranker, case["query"], ordinary_candidates)
            graph_result = retriever.retrieve(
                case["query"],
                seed_documents([item[0] for item in ordinary_ranked[:5]]),
            )
            augmented_candidates = _merge_candidates(
                (ordinary_candidates, [graph_candidate(item) for item in graph_result.documents])
            )
            augmented_ranked = _rank(reranker, case["query"], augmented_candidates)
            evidence = _evidence_payload(_answer_ranked(augmented_ranked))
            if graph_result.requires_temporal_abstention:
                response = _temporal_abstention(case["language"])
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                cache_hit = False
                answer_path = "safe_temporal_abstention"
            else:
                response, usage, cache_hit = _answer_with_cache(
                    client=client,
                    case_id=case["id"],
                    query=case["query"],
                    evidence=evidence,
                    suite_hash=expected_suite_sha256,
                    cache_dir=cache_dir,
                )
                answer_path = "retrieval_graph_structured_answer"
            runtime_records.append(
                {
                    **case,
                    "ordinary_candidate_pages": _candidate_pages(ordinary_candidates),
                    "ordinary_top20": ranked_signature(ordinary_ranked, limit=20),
                    "augmented_candidate_pages": _candidate_pages(augmented_candidates),
                    "augmented_top20": ranked_signature(augmented_ranked, limit=20),
                    "graph_trace": graph_result.trace.as_dict(),
                    "requires_temporal_abstention": graph_result.requires_temporal_abstention,
                    "retrieved_evidence": evidence,
                    "answer_path": answer_path,
                    "response": response,
                    "usage": usage,
                    "cache_hit": cache_hit,
                    "latency_seconds": time.perf_counter() - started,
                }
            )
            print(f"[enriched-e2e {index}/{len(inputs)}] {case['id']} graph={graph_result.trace.status.value} answer={response['status']}", flush=True)
        after = _snapshot(graph.snapshot())
    finally:
        driver.close()

    runtime_artifact = {
        "status": "frozen_before_scoring",
        "experiment_id": experiment_id,
        "runtime_gold_access": False,
        "case_count": len(runtime_records),
        "records": runtime_records,
    }
    write_json_atomic(runtime_output_path, runtime_artifact)

    gold = {case["id"]: case for case in cases}
    scored = []
    for record in runtime_records:
        case = gold[record["id"]]
        ordinary = (
            score_case(case, record["ordinary_candidate_pages"], record["ordinary_top20"])
            if case.get("relevant", True)
            else None
        )
        retrieval = (
            score_case(case, record["augmented_candidate_pages"], record["augmented_top20"])
            if case.get("relevant", True)
            else None
        )
        diagnostics = structured_diagnostics(case, record["response"], record["retrieved_evidence"])
        scored.append(
            {
                "id": record["id"],
                "language": record["language"],
                "category": record["category"],
                "relevant": case.get("relevant", True),
                "expected_behavior": case.get("expected_behavior"),
                "ordinary_retrieval": ordinary,
                "retrieval": retrieval,
                "graph_status": record["graph_trace"]["status"],
                "answer_path": record["answer_path"],
                "response": record["response"],
                "diagnostics": diagnostics,
                "automatic_answer_audit": (
                    automatic_answer_audit(case, record["response"]["answer"])
                    if case.get("relevant", True)
                    else None
                ),
                "usage": record["usage"],
                "latency_seconds": record["latency_seconds"],
            }
        )

    unchanged = before == after
    result = {
        "status": "complete_pending_semantic_review" if unchanged else "failed",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "configuration": {
            "retrieval": "structured native dense20+BM15; Arabic OCR dense5+BM5; identity-aware BGE; diverse pages",
            "graph": "read-only verified Neo4j expansion from ordinary top-five pages",
            "answer_model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "reasoning_effort": REASONING_EFFORT,
        },
        "inputs": {
            "suite_sha256": expected_suite_sha256,
            "native_manifest_sha256": NATIVE_MANIFEST_SHA256,
            "ocr_manifest_sha256": OCR_MANIFEST_SHA256,
            "runtime_output_sha256": sha256_file(runtime_output_path),
        },
        "metrics": {
            "overall": _metrics(scored),
            "fr": _metrics([r for r in scored if r["language"] == "fr"]),
            "ar": _metrics([r for r in scored if r["language"] == "ar"]),
            "graph_status_counts": dict(sorted(Counter(r["graph_status"] for r in scored).items())),
        },
        "usage": {
            field: sum(r["usage"][field] for r in scored)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "latency_seconds": {
            "total": sum(r["latency_seconds"] for r in scored),
            "mean": statistics.mean(r["latency_seconds"] for r in scored),
            "median": statistics.median(r["latency_seconds"] for r in scored),
        },
        "persistent_graph": {"before": before, "after": after, "unchanged": unchanged},
        "records": scored,
        "limitations": [
            "Semantic correctness and substantive grounding remain pending case-level review.",
            f"This run contains {len(cases)} frozen enriched-evaluation cases.",
            "This uses the integrated local retrieval configuration, not Voyage configuration K.",
            "The cases are independent single-turn questions; conversation-memory quality is evaluated separately.",
        ],
    }
    write_json_atomic(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--ocr-manifest", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:17687")
    parser.add_argument("--expected-suite-sha256", default=SUITE_SHA256)
    parser.add_argument("--expected-case-count", type=int, default=EXPECTED_CASE_COUNT)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--confirm-public-documents", action="store_true")
    args = parser.parse_args()
    result = run(
        suite_path=args.suite,
        native_manifest_path=args.native_manifest,
        ocr_manifest_path=args.ocr_manifest,
        dotenv_path=args.dotenv,
        cache_dir=args.cache_dir,
        runtime_output_path=args.runtime_output,
        result_path=args.result,
        neo4j_uri=args.neo4j_uri,
        confirm_public_documents=args.confirm_public_documents,
        expected_suite_sha256=args.expected_suite_sha256,
        expected_case_count=args.expected_case_count,
        experiment_id=args.experiment_id,
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
