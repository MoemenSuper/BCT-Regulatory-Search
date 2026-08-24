"""Run the answer policy on retrieved development context without gold evidence."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.gold_evidence_answer_experiment import automatic_answer_audit
from experiments.provisional_validation_retrieval import (
    NATIVE_MANIFEST_SHA256,
    OCR_BM25_K,
    OCR_DENSE_K,
    OCR_MANIFEST_SHA256,
    diversify_ranked_pages,
    is_arabic_query,
)
from experiments.query_state_experiment import _answer, derive_status
from experiments.retrieval_ablations import (
    BM25_K,
    DENSE_K,
    _load_search_representation,
    _merge_candidates,
    _ranks,
    _retrieve,
)
from experiments.structured_answer_experiment import (
    MODEL,
    _SCHEMA_V2,
    _SYSTEM_V5,
    _aggregate,
    _usage,
    parse_structured_answer,
)
from reranker import create_reranker, score_documents


PROMPT_VERSION = "bct-retrieved-context-answer-v1"


def _query_states(
    suite: dict[str, Any], negative_result: dict[str, Any], relevant_result: dict[str, Any]
) -> dict[str, dict[str, str]]:
    records = [*negative_result["records"], *relevant_result["records"]]
    states = {record["id"]: record["query_state"] for record in records}
    if len(states) != len(records):
        raise ValueError("Query-state inputs contain duplicate IDs")
    expected_ids = {case["id"] for case in suite["cases"]}
    if set(states) != expected_ids:
        raise ValueError("Query-state inputs must exactly cover the answer suite")
    return states


def _evidence_payload(
    ranked: list[tuple[dict[str, Any], float]], limit: int = 5
) -> list[dict[str, Any]]:
    evidence = []
    for index, (candidate, score) in enumerate(ranked[:limit], start=1):
        document = candidate["document"]
        metadata = document.metadata
        evidence.append(
            {
                "evidence_id": f"E{index}",
                "source": Path(str(metadata.get("source", ""))).name,
                "page": int(metadata.get("page", -1)),
                "page_end": int(metadata.get("page_end", metadata.get("page", -1))),
                "text": document.page_content,
                "reranker_score": score,
                "representations": sorted(candidate["representations"]),
            }
        )
    return evidence


def retrieved_structured_diagnostics(
    case: dict[str, Any], response: dict[str, Any], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    claim_links_valid = bool(response["claims"]) and all(
        bool(claim["evidence_ids"])
        and set(claim["evidence_ids"]).issubset(evidence_by_id)
        for claim in response["claims"]
    )
    citations_valid = all(
        citation["evidence_id"] in evidence_by_id
        and Path(citation["source"]).name.casefold()
        == Path(evidence_by_id[citation["evidence_id"]]["source"]).name.casefold()
        and int(citation["page"])
        == int(evidence_by_id[citation["evidence_id"]]["page"])
        for citation in response["citations"]
    )
    if case["relevant"]:
        exact_expected_citation = any(
            Path(citation["source"]).name.casefold()
            == Path(case["expected_source"]).name.casefold()
            and int(citation["page"]) == int(case["expected_page"])
            for citation in response["citations"]
        )
        expected_status = response["status"] == "answered"
    else:
        exact_expected_citation = not response["citations"]
        expected_status = response["status"] == {
            "clarify": "clarification_needed",
            "reject_out_of_scope": "out_of_scope",
            "abstain": "insufficient_evidence",
        }[case["expected_behavior"]]
    return {
        "status_expected": expected_status,
        "exact_structured_citation": exact_expected_citation,
        "exact_expected_citation": exact_expected_citation,
        "citations_match_retrieved_evidence": citations_valid,
        "claim_evidence_links_valid": claim_links_valid if response["status"] == "answered" else not response["claims"],
        "claim_count": len(response["claims"]),
        "citation_count": len(response["citations"]),
    }


def apply_post_retrieval_query_state(
    response: dict[str, Any], state: dict[str, str], language: str
) -> tuple[dict[str, Any], str]:
    if response["status"] == "answered":
        return response, "retrieved_top5_answer"
    status = derive_status(state)
    return (
        {
            "status": status,
            "answer": _answer(status, state, language),
            "claims": [],
            "citations": [],
        },
        "retrieved_top5_abstention_then_query_state",
    )


def run_retrieved_context_answer_experiment(
    *,
    suite_path: Path,
    native_manifest_path: Path,
    ocr_manifest_path: Path,
    negative_query_state_path: Path,
    relevant_query_state_path: Path,
    dotenv_path: Path,
    output_dir: Path,
    confirm_public_documents: bool,
) -> dict[str, Any]:
    if not confirm_public_documents:
        raise ValueError("Public-document confirmation is required for hosted answer calls")
    if sha256_file(native_manifest_path) != NATIVE_MANIFEST_SHA256:
        raise ValueError("Native representation manifest differs from the selected control")
    if sha256_file(ocr_manifest_path) != OCR_MANIFEST_SHA256:
        raise ValueError("OCR representation manifest differs from the selected control")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured")

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_hash = sha256_file(suite_path)
    negative_query_state = json.loads(negative_query_state_path.read_text(encoding="utf-8"))
    relevant_query_state = json.loads(relevant_query_state_path.read_text(encoding="utf-8"))
    states = _query_states(suite, negative_query_state, relevant_query_state)
    native = _load_search_representation(
        json.loads(native_manifest_path.read_text(encoding="utf-8"))
    )
    ocr = _load_search_representation(
        json.loads(ocr_manifest_path.read_text(encoding="utf-8"))
    )
    reranker = create_reranker()
    client = Groq(max_retries=3, timeout=90.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for index, case in enumerate(suite["cases"], start=1):
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
        evidence = _evidence_payload(diverse)
        source_rank = exact_page_rank = None
        if case["relevant"]:
            source_rank, exact_page_rank = _ranks(
                diverse, case["expected_source"], int(case["expected_page"])
            )

        user_payload = json.dumps(
            {
                "question": case["query"],
                "evidence": [
                    {
                        key: item[key]
                        for key in ("evidence_id", "source", "page", "text")
                    }
                    for item in evidence
                ],
            },
            ensure_ascii=False,
        )
        cache_path = cache_dir / f"{case['id']}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("model") != MODEL
                or cached.get("prompt_version") != PROMPT_VERSION
                or cached.get("suite_sha256") != suite_hash
                or cached.get("user_payload") != user_payload
            ):
                raise ValueError(f"Retrieved-answer cache differs for {case['id']}")
            generated = parse_structured_answer(
                json.dumps(cached["response"], ensure_ascii=False),
                require_nonempty_answer=True,
            )
        else:
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM_V5},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "bct_retrieved_context_answer",
                            "strict": True,
                            "schema": _SCHEMA_V2,
                        },
                    },
                    reasoning_effort="medium",
                    temperature=0,
                    seed=20260824,
                    max_completion_tokens=2048,
                )
            except RateLimitError as error:
                checkpoint = {
                    "status": "rate_limited",
                    "completed": len(records),
                    "total": len(suite["cases"]),
                    "retry_after": (
                        error.response.headers.get("retry-after", "unknown")
                        if error.response is not None
                        else "unknown"
                    ),
                }
                write_json_atomic(output_dir / "checkpoint.json", checkpoint)
                return checkpoint
            generated = parse_structured_answer(
                completion.choices[0].message.content or "",
                require_nonempty_answer=True,
            )
            cached = {
                "id": case["id"],
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "suite_sha256": suite_hash,
                "user_payload": user_payload,
                "response_id": completion.id,
                "usage": _usage(completion.usage),
                "response": generated,
            }
            write_json_atomic(cache_path, cached)

        response, answer_path = apply_post_retrieval_query_state(
            generated, states[case["id"]], case["language"]
        )
        record = {
            "id": case["id"],
            "language": case["language"],
            "answer_suite_role": case["answer_suite_role"],
            "relevant": bool(case["relevant"]),
            "expected_behavior": case.get("expected_behavior"),
            "result": {
                "source_rank": source_rank,
                "exact_page_rank": exact_page_rank,
            },
            "retrieved_evidence": evidence,
            "generated_status_before_query_state": generated["status"],
            "answer_path": answer_path,
            "response": response,
            "usage": cached["usage"],
            "latency_seconds": time.perf_counter() - started,
            "automatic_audit": automatic_answer_audit(case, response["answer"]),
            "structured_diagnostics": retrieved_structured_diagnostics(
                case, response, evidence
            ),
        }
        records.append(record)
        print(
            f"[retrieved-answer {index}/{len(suite['cases'])}] {case['id']} "
            f"page={exact_page_rank} status={response['status']}",
            flush=True,
        )

    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "retrieval-first-answer-development-v1",
        "configuration": {
            "retrieval": "frozen native candidate union plus additive Arabic OCR, BGE reranker, diverse top-5 pages",
            "answer_model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "post_retrieval_status": "query state is consulted only after the generator declines to answer",
        },
        "inputs": {
            "answer_suite_sha256": suite_hash,
            "native_manifest_sha256": sha256_file(native_manifest_path),
            "ocr_manifest_sha256": sha256_file(ocr_manifest_path),
            "negative_query_state_sha256": sha256_file(negative_query_state_path),
            "relevant_query_state_sha256": sha256_file(relevant_query_state_path),
        },
        "automatic_metrics": {
            "overall": _aggregate(records),
            "fr": _aggregate([record for record in records if record["language"] == "fr"]),
            "ar": _aggregate([record for record in records if record["language"] == "ar"]),
        },
        "usage": {
            field: sum(record["usage"][field] for record in records)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "latency_seconds": {
            "mean": statistics.mean(record["latency_seconds"] for record in records),
            "median": statistics.median(record["latency_seconds"] for record in records),
        },
        "limitations": [
            "Development-only end-to-end retrieved-context experiment.",
            "Semantic answer correctness and grounding require case-level evidence review.",
            "The first pass has no page-expansion retry and is expected to expose evidence-sufficiency failures.",
        ],
        "records": records,
    }
    write_json_atomic(output_dir / "result.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--ocr-manifest", type=Path, required=True)
    parser.add_argument("--negative-query-state", type=Path, required=True)
    parser.add_argument("--relevant-query-state", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-public-documents", action="store_true")
    args = parser.parse_args()
    run_retrieved_context_answer_experiment(
        suite_path=args.suite,
        native_manifest_path=args.native_manifest,
        ocr_manifest_path=args.ocr_manifest,
        negative_query_state_path=args.negative_query_state,
        relevant_query_state_path=args.relevant_query_state,
        dotenv_path=args.dotenv,
        output_dir=args.output_dir,
        confirm_public_documents=args.confirm_public_documents,
    )


if __name__ == "__main__":
    main()
