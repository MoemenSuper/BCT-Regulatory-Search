"""Re-run only answer cases changed by the frozen document-identity ranking."""

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
from experiments.document_identity_candidate_experiment import (
    NATIVE_CANDIDATES_SHA256,
    build_identity_reranker_documents,
    parse_query_identity,
    ranked_signature,
    _rank,
)
from experiments.gold_evidence_answer_experiment import automatic_answer_audit
from experiments.ocr_fusion_retrieval import (
    OCR_BM25_K,
    OCR_DENSE_K,
    _deserialize_candidates,
    _load_search_representation,
    _merge_candidates,
    _retrieve,
    is_arabic_query,
)
from experiments.provisional_validation_retrieval import OCR_MANIFEST_SHA256
from experiments.retrieved_context_answer_experiment import (
    MAX_COMPLETION_TOKENS,
    PROMPT_VERSION,
    REASONING_EFFORT,
    _evidence_payload,
    _query_states,
    apply_post_retrieval_query_state,
    retrieved_structured_diagnostics,
)
from experiments.structured_answer_experiment import (
    MODEL,
    _SCHEMA_V2,
    _SYSTEM_V5,
    _aggregate,
    _usage,
    parse_structured_answer,
)
from reranker import create_reranker


ANSWER_SUITE_SHA256 = (
    "5E3D840B6DF8FDF3046AA2A5A67B84DF8CC0E0E61D4EC810E5D51446C89917A1"
)
RAW_RETRIEVED_RESULT_SHA256 = (
    "5E6BE4C8D05A144248821156A03F5789D91D63F2D6928EC599E997905E2DE450"
)
SEED = 20260824
EXPERIMENT_ID = "document-identity-answer-recomposition-development-v1"


def _safe_abstention(language: str) -> dict[str, Any]:
    answer = (
        "الأدلة المسترجعة غير كافية أو تعذر التحقق من بنية الإجابة، لذلك لا يمكنني تقديم إجابة موثوقة."
        if language == "ar"
        else "Les preuves récupérées sont insuffisantes ou la réponse n'a pas pu être validée; je ne peux donc pas répondre de façon fiable."
    )
    return {
        "status": "insufficient_evidence",
        "answer": answer,
        "claims": [],
        "citations": [],
    }


def parse_or_fail_closed(
    content: str, language: str
) -> tuple[dict[str, Any], str, str | None]:
    try:
        return (
            parse_structured_answer(content, require_nonempty_answer=True),
            "valid",
            None,
        )
    except (TypeError, ValueError) as error:
        return _safe_abstention(language), "fail_closed_structured_validation", str(error)


def derive_changed_answer_case_ids(
    suite: dict[str, Any], raw: dict[str, Any], identity_result: dict[str, Any]
) -> list[str]:
    suite_ids = [case["id"] for case in suite["cases"]]
    raw_ids = [record["id"] for record in raw["records"]]
    if len(set(suite_ids)) != len(suite_ids) or len(set(raw_ids)) != len(raw_ids):
        raise ValueError("Suite and raw answer records must contain unique IDs")
    if set(suite_ids) != set(raw_ids):
        raise ValueError("Raw answer records must exactly cover the answer suite")
    changed = set(identity_result["changed_top5_ids"])
    return [case_id for case_id in raw_ids if case_id in changed]


def validate_top5_signature(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    if len(actual) != len(expected):
        raise ValueError("Identity top-five replay differs from the frozen receipt")
    for actual_item, expected_item in zip(actual, expected):
        same_source = (
            Path(actual_item["source"]).name.casefold()
            == Path(expected_item["source"]).name.casefold()
        )
        same_page = int(actual_item["page"]) == int(expected_item["page"])
        same_score = abs(float(actual_item["score"]) - float(expected_item["score"])) <= 1e-9
        if not (same_source and same_page and same_score):
            raise ValueError("Identity top-five replay differs from the frozen receipt")


def recompose_records(
    raw_records: list[dict[str, Any]],
    replacements: dict[str, dict[str, Any]],
    changed_ids: list[str],
) -> list[dict[str, Any]]:
    if set(replacements) != set(changed_ids):
        raise ValueError("Replacement IDs must exactly match frozen changed answer IDs")
    return [replacements.get(record["id"], record) for record in raw_records]


def answer_cache_binding(
    *,
    suite_sha256: str,
    raw_result_sha256: str,
    identity_result_sha256: str,
    routing_receipt_sha256: str,
    user_payload: str,
) -> dict[str, Any]:
    return {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "seed": SEED,
        "suite_sha256": suite_sha256,
        "raw_result_sha256": raw_result_sha256,
        "identity_result_sha256": identity_result_sha256,
        "routing_receipt_sha256": routing_receipt_sha256,
        "user_payload": user_payload,
    }


def _answer_payload(question: str, evidence: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "question": question,
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


def run_identity_answer_recomposition(
    *,
    suite_path: Path,
    raw_result_path: Path,
    identity_result_path: Path,
    routing_receipt_path: Path,
    native_candidates_path: Path,
    ocr_manifest_path: Path,
    negative_query_state_path: Path,
    relevant_query_state_path: Path,
    dotenv_path: Path,
    output_dir: Path,
    confirm_public_documents: bool,
    prior_uncached_hosted_attempts: int = 0,
) -> dict[str, Any]:
    if not confirm_public_documents:
        raise ValueError("Hosted answer execution requires public-document confirmation")

    input_hashes = {
        "answer_suite_sha256": sha256_file(suite_path),
        "raw_retrieved_result_sha256": sha256_file(raw_result_path),
        "identity_result_sha256": sha256_file(identity_result_path),
        "routing_receipt_sha256": sha256_file(routing_receipt_path),
        "native_candidates_sha256": sha256_file(native_candidates_path),
        "ocr_manifest_sha256": sha256_file(ocr_manifest_path),
        "negative_query_state_sha256": sha256_file(negative_query_state_path),
        "relevant_query_state_sha256": sha256_file(relevant_query_state_path),
    }
    expected_frozen = {
        "answer_suite_sha256": ANSWER_SUITE_SHA256,
        "raw_retrieved_result_sha256": RAW_RETRIEVED_RESULT_SHA256,
        "native_candidates_sha256": NATIVE_CANDIDATES_SHA256,
        "ocr_manifest_sha256": OCR_MANIFEST_SHA256,
    }
    mismatches = [
        key for key, expected in expected_frozen.items() if input_hashes[key] != expected
    ]
    if mismatches:
        raise ValueError(f"Frozen answer-stage inputs differ: {mismatches}")

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_result_path.read_text(encoding="utf-8"))
    identity_result = json.loads(identity_result_path.read_text(encoding="utf-8"))
    routing = json.loads(routing_receipt_path.read_text(encoding="utf-8"))
    if identity_result.get("decision") != "KEEP" or not all(
        identity_result.get("gate", {}).values()
    ):
        raise ValueError("Identity retrieval result did not pass its complete gate")
    if identity_result.get("inputs", {}).get("routing_receipt_sha256") != input_hashes[
        "routing_receipt_sha256"
    ]:
        raise ValueError("Identity result is bound to a different routing receipt")

    changed_ids = derive_changed_answer_case_ids(suite, raw, identity_result)
    if not changed_ids:
        raise ValueError("No answer-suite evidence changed")
    cases = {case["id"]: case for case in suite["cases"]}
    raw_records = {record["id"]: record for record in raw["records"]}
    identity_records = {record["id"]: record for record in identity_result["records"]}
    routing_records = {record["id"]: record for record in routing["records"]}
    if not set(changed_ids).issubset(identity_records) or not set(changed_ids).issubset(
        routing_records
    ):
        raise ValueError("Frozen identity artifacts do not cover changed answer cases")

    negative_query_state = json.loads(
        negative_query_state_path.read_text(encoding="utf-8")
    )
    relevant_query_state = json.loads(
        relevant_query_state_path.read_text(encoding="utf-8")
    )
    states = _query_states(suite, negative_query_state, relevant_query_state)
    native_cache = json.loads(native_candidates_path.read_text(encoding="utf-8"))
    ocr = _load_search_representation(
        json.loads(ocr_manifest_path.read_text(encoding="utf-8"))
    )
    reranker = create_reranker()

    load_dotenv(dotenv_path=dotenv_path, override=False)
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured")
    client = Groq(max_retries=3, timeout=90.0)
    available = {model.id for model in client.models.list().data}
    if MODEL not in available:
        raise ValueError(f"Required answer model is not currently available: {MODEL}")

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    replacements: dict[str, dict[str, Any]] = {}
    hosted_requests = cache_hits = 0
    usage_total = {
        field: 0 for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    latencies: list[float] = []

    for case_id in changed_ids:
        case = cases[case_id]
        frozen_route = routing_records[case_id]
        query_identity = parse_query_identity(case["query"])
        if query_identity is None or query_identity != frozen_route["query_identity"]:
            raise ValueError(f"Runtime identity route drift for {case_id}")
        base = _deserialize_candidates(native_cache[case_id])
        ocr_candidates = (
            _retrieve(ocr, case["query"], OCR_DENSE_K, OCR_BM25_K)
            if is_arabic_query(case["query"])
            else []
        )
        candidates = _merge_candidates((base, ocr_candidates))
        _undiversified, diverse = _rank(
            reranker,
            case["query"],
            candidates,
            build_identity_reranker_documents(candidates, query_identity),
        )
        replayed_top5 = ranked_signature(diverse, limit=5)
        validate_top5_signature(replayed_top5, frozen_route["candidate_ranked"][:5])
        evidence = _evidence_payload(diverse)
        user_payload = _answer_payload(case["query"], evidence)
        binding = answer_cache_binding(
            suite_sha256=input_hashes["answer_suite_sha256"],
            raw_result_sha256=input_hashes["raw_retrieved_result_sha256"],
            identity_result_sha256=input_hashes["identity_result_sha256"],
            routing_receipt_sha256=input_hashes["routing_receipt_sha256"],
            user_payload=user_payload,
        )
        cache_path = cache_dir / f"{case_id}.json"
        cache_hit = cache_path.exists()
        if cache_hit:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_mismatches = [
                key for key, expected in binding.items() if cached.get(key) != expected
            ]
            if cache_mismatches:
                raise ValueError(
                    f"Identity-answer cache differs for {case_id}: {cache_mismatches}"
                )
            generated = parse_structured_answer(
                json.dumps(cached["response"], ensure_ascii=False),
                require_nonempty_answer=True,
            )
            validation_status = cached["structured_validation_status"]
            validation_error = cached.get("structured_validation_error")
            cache_hits += 1
        else:
            started = time.perf_counter()
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
                    reasoning_effort=REASONING_EFFORT,
                    temperature=0,
                    seed=SEED,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )
            except RateLimitError as error:
                checkpoint = {
                    "status": "rate_limited",
                    "completed_changed_cases": len(replacements),
                    "total_changed_cases": len(changed_ids),
                    "retry_after": (
                        error.response.headers.get("retry-after", "unknown")
                        if error.response is not None
                        else "unknown"
                    ),
                    "cache_is_resumable": True,
                }
                write_json_atomic(output_dir / "checkpoint.json", checkpoint)
                return checkpoint
            latency = time.perf_counter() - started
            raw_response_content = completion.choices[0].message.content or ""
            generated, validation_status, validation_error = parse_or_fail_closed(
                raw_response_content, case["language"]
            )
            cached = {
                "id": case_id,
                **binding,
                "response_id": completion.id,
                "usage": _usage(completion.usage),
                "latency_seconds": latency,
                "raw_response_content": raw_response_content,
                "structured_validation_status": validation_status,
                "structured_validation_error": validation_error,
                "response": generated,
            }
            write_json_atomic(cache_path, cached)
            hosted_requests += 1

        usage = cached["usage"]
        latency = float(cached.get("latency_seconds", 0.0))
        for field in usage_total:
            usage_total[field] += int(usage[field])
        if latency:
            latencies.append(latency)
        response, answer_path = apply_post_retrieval_query_state(
            generated, states[case_id], case["language"]
        )
        identity_record = identity_records[case_id]
        raw_record = raw_records[case_id]
        replacements[case_id] = {
            **raw_record,
            "result": {
                "source_rank": identity_record["candidate_source_rank"],
                "exact_page_rank": identity_record["candidate_page_rank"],
            },
            "retrieved_evidence": evidence,
            "generated_status_before_query_state": generated["status"],
            "answer_path": answer_path,
            "response": response,
            "usage": usage,
            "latency_seconds": latency,
            "automatic_audit": automatic_answer_audit(case, response["answer"]),
            "structured_diagnostics": retrieved_structured_diagnostics(
                case, response, evidence
            ),
            "document_identity": {
                "routed": True,
                "route_reason": query_identity["route_reason"],
                "query_identity": query_identity,
                "control_top5": identity_record["control_top5"],
                "candidate_top5": identity_record["candidate_top5"],
                "response_changed": response != raw_record["response"],
                "original_response": raw_record["response"],
                "cache_hit": cache_hit,
                "structured_validation_status": validation_status,
                "structured_validation_error": validation_error,
            },
        }
        print(
            f"[identity-answer {case_id}] page={identity_record['candidate_page_rank']} "
            f"status={response['status']} changed={response != raw_record['response']} "
            f"cache_hit={cache_hit}",
            flush=True,
        )

    records = recompose_records(raw["records"], replacements, changed_ids)
    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "decision": "pending_manual_review",
        "configuration": {
            "changed_variable": "identity_prefixed_reranking_evidence_only",
            "answer_model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "reasoning_effort": REASONING_EFFORT,
            "seed": SEED,
            "untouched_cases": "copied byte-for-value from frozen raw records",
            "post_retrieval_status": "query state consulted only after generator abstention",
        },
        "inputs": input_hashes,
        "counts": {
            "all_cases": len(records),
            "changed_answer_cases": len(changed_ids),
            "untouched_cases": len(records) - len(changed_ids),
            "hosted_answer_requests_this_run": hosted_requests,
            "prior_uncached_hosted_attempts": prior_uncached_hosted_attempts,
            "total_known_hosted_attempts": hosted_requests
            + prior_uncached_hosted_attempts,
            "answer_cache_hits_this_run": cache_hits,
        },
        "changed_answer_case_ids": changed_ids,
        "automatic_metrics": {
            "overall": _aggregate(records),
            "fr": _aggregate([record for record in records if record["language"] == "fr"]),
            "ar": _aggregate([record for record in records if record["language"] == "ar"]),
        },
        "new_answer_usage": usage_total,
        "usage_completeness": (
            "incomplete_one_or_more_prior_attempts_failed_before_cache"
            if prior_uncached_hosted_attempts
            else "complete_for_known_hosted_attempts"
        ),
        "new_answer_latency_seconds": {
            "total": sum(latencies),
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
        },
        "cost_status": "not_estimated_without_verified_applicable_account_rate",
        "validation_metrics": {
            "status": "not_run",
            "reason": "Provisional answer validation remains closed.",
        },
        "records": records,
    }
    write_json_atomic(output_dir / "result.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--identity-result", type=Path, required=True)
    parser.add_argument("--routing-receipt", type=Path, required=True)
    parser.add_argument("--native-candidates", type=Path, required=True)
    parser.add_argument("--ocr-manifest", type=Path, required=True)
    parser.add_argument("--negative-query-state", type=Path, required=True)
    parser.add_argument("--relevant-query-state", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-public-documents", action="store_true")
    parser.add_argument("--prior-uncached-hosted-attempts", type=int, default=0)
    args = parser.parse_args()
    run_identity_answer_recomposition(
        suite_path=args.suite,
        raw_result_path=args.raw_result,
        identity_result_path=args.identity_result,
        routing_receipt_path=args.routing_receipt,
        native_candidates_path=args.native_candidates,
        ocr_manifest_path=args.ocr_manifest,
        negative_query_state_path=args.negative_query_state,
        relevant_query_state_path=args.relevant_query_state,
        dotenv_path=args.dotenv,
        output_dir=args.output_dir,
        confirm_public_documents=args.confirm_public_documents,
        prior_uncached_hosted_attempts=args.prior_uncached_hosted_attempts,
    )


if __name__ == "__main__":
    main()
