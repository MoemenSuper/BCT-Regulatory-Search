"""Re-run only frozen routed answers with cache-bound visual evidence."""

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

from experiments.arabic_visual_fallback import (
    validate_visual_cache_binding,
    visual_evidence,
    visual_payload_is_usable,
)
from experiments.artifacts import sha256_file, write_json_atomic
from experiments.gold_evidence_answer_experiment import automatic_answer_audit
from experiments.retrieved_context_answer_experiment import (
    MAX_COMPLETION_TOKENS,
    REASONING_EFFORT,
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


PROMPT_VERSION = "bct-retrieved-context-answer-visual-v1-low-reasoning"
SEED = 20260825


def _page_key(source: str, page: int) -> tuple[str, int]:
    return Path(source).name.casefold(), int(page)


def _safe_abstention(language: str) -> dict[str, Any]:
    answer = (
        "الأدلة المرئية المتاحة غير مكتملة أو غير مؤكدة، لذلك لا يمكنني تقديم إجابة موثوقة."
        if language == "ar"
        else "Les preuves visuelles disponibles sont incomplètes ou incertaines; je ne peux donc pas répondre de façon fiable."
    )
    return {
        "status": "insufficient_evidence",
        "answer": answer,
        "claims": [],
        "citations": [],
    }


def prepare_routed_evidence(
    *,
    record: dict[str, Any],
    route: dict[str, Any],
    visual_pages: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    evidence = [dict(item) for item in record["retrieved_evidence"]]
    evidence_by_id = {item["evidence_id"]: item for item in evidence}
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for page in route["pages"]:
        native = evidence_by_id.get(page["evidence_id"])
        if native is None or _page_key(native["source"], native["page"]) != _page_key(
            page["source"], page["page"]
        ):
            raise ValueError(f"Frozen route differs from raw evidence for {record['id']}")
        cached = visual_pages.get(_page_key(page["source"], page["page"]))
        if (
            cached is None
            or cached.get("validation_status") != "valid"
            or not visual_payload_is_usable(cached["response"])
        ):
            return evidence, "fail_closed_visual_unavailable_invalid_or_uncertain"
        validate_visual_cache_binding(
            cached,
            source_pdf_sha256=cached["source_pdf_sha256"],
            page=cached["page"],
            image_sha256=cached["image_sha256"],
        )
        selected.append((native, cached))

    replacements: dict[str, dict[str, Any]] = {}
    for native, cached in selected:
        replacement = visual_evidence(native, cached)
        if replacement is None:
            return evidence, "fail_closed_visual_unavailable_invalid_or_uncertain"
        replacements[native["evidence_id"]] = replacement
    return [replacements.get(item["evidence_id"], item) for item in evidence], "generate"


def _answer_payload(question: str, evidence: list[dict[str, Any]]) -> str:
    values = []
    for item in evidence:
        value = {
            key: item[key] for key in ("evidence_id", "source", "page", "text")
        }
        if "visual_verification" in item:
            value["evidence_origin"] = "cache_bound_visual_transcription"
            value["visual_verification"] = item["visual_verification"]
        values.append(value)
    return json.dumps({"question": question, "evidence": values}, ensure_ascii=False)


def run_routed_visual_answers(
    *,
    suite_path: Path,
    raw_result_path: Path,
    routing_receipt_path: Path,
    visual_result_path: Path,
    dotenv_path: Path,
    output_dir: Path,
    confirm_public_documents: bool,
) -> dict[str, Any]:
    if not confirm_public_documents:
        raise ValueError("Hosted answer execution requires public-document confirmation")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_result_path.read_text(encoding="utf-8"))
    routing = json.loads(routing_receipt_path.read_text(encoding="utf-8"))
    visual = json.loads(visual_result_path.read_text(encoding="utf-8"))
    if visual.get("status") != "complete":
        raise ValueError("Visual result is incomplete")
    if visual.get("inputs", {}).get("routing_receipt_sha256") != sha256_file(
        routing_receipt_path
    ):
        raise ValueError("Visual result is bound to a different routing receipt")

    cases = {case["id"]: case for case in suite["cases"]}
    raw_records = {record["id"]: record for record in raw["records"]}
    routes = {route["id"]: route for route in routing["routes"]}
    if not set(routes).issubset(raw_records) or set(cases) != set(raw_records):
        raise ValueError("Suite, raw result, and routes do not align")
    visual_pages = {
        _page_key(page["source"], page["page"]): page for page in visual["pages"]
    }

    load_dotenv(dotenv_path=dotenv_path, override=False)
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured")
    client = Groq(max_retries=3, timeout=90.0)
    available = {model.id for model in client.models.list().data}
    if MODEL not in available:
        raise ValueError(f"Required answer model is not currently available: {MODEL}")
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suite_hash = sha256_file(suite_path)
    raw_hash = sha256_file(raw_result_path)
    visual_hash = sha256_file(visual_result_path)

    records = []
    new_usage = {field: 0 for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
    new_latencies: list[float] = []
    hosted_requests = 0
    cache_hits = 0
    for raw_record in raw["records"]:
        case_id = raw_record["id"]
        if case_id not in routes:
            records.append(raw_record)
            continue
        case = cases[case_id]
        evidence, action = prepare_routed_evidence(
            record=raw_record,
            route=routes[case_id],
            visual_pages=visual_pages,
        )
        original_response = raw_record["response"]
        if action != "generate":
            generated = _safe_abstention(case["language"])
            usage = {field: 0 for field in new_usage}
            latency = 0.0
            cache_hit = False
        else:
            user_payload = _answer_payload(case["query"], evidence)
            cache_path = cache_dir / f"{case_id}.json"
            cache_hit = cache_path.exists()
            if cache_hit:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                expected = {
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "suite_sha256": suite_hash,
                    "raw_result_sha256": raw_hash,
                    "visual_result_sha256": visual_hash,
                    "user_payload": user_payload,
                }
                mismatches = [key for key, value in expected.items() if cached.get(key) != value]
                if mismatches:
                    raise ValueError(f"Routed-answer cache differs for {case_id}: {mismatches}")
                generated = parse_structured_answer(
                    json.dumps(cached["response"], ensure_ascii=False),
                    require_nonempty_answer=True,
                )
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
                                "name": "bct_routed_visual_answer",
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
                        "completed_records": len(records),
                        "total_records": len(raw["records"]),
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
                generated = parse_structured_answer(
                    completion.choices[0].message.content or "",
                    require_nonempty_answer=True,
                )
                cached = {
                    "id": case_id,
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "suite_sha256": suite_hash,
                    "raw_result_sha256": raw_hash,
                    "visual_result_sha256": visual_hash,
                    "user_payload": user_payload,
                    "response_id": completion.id,
                    "usage": _usage(completion.usage),
                    "latency_seconds": latency,
                    "response": generated,
                }
                write_json_atomic(cache_path, cached)
            usage = cached["usage"]
            latency = float(cached.get("latency_seconds", 0.0))
            if cache_hit:
                cache_hits += 1
            else:
                hosted_requests += 1
        for field in new_usage:
            new_usage[field] += int(usage[field])
        if latency:
            new_latencies.append(latency)
        record = {
            **raw_record,
            "retrieved_evidence": evidence,
            "generated_status_before_query_state": generated["status"],
            "answer_path": (
                "routed_visual_answer" if action == "generate" else action
            ),
            "response": generated,
            "usage": usage,
            "latency_seconds": latency,
            "automatic_audit": automatic_answer_audit(case, generated["answer"]),
            "structured_diagnostics": retrieved_structured_diagnostics(
                case, generated, evidence
            ),
            "visual_fallback": {
                "routed": True,
                "action": action,
                "cache_hit": cache_hit,
                "response_changed": generated != original_response,
                "original_response": original_response,
            },
        }
        records.append(record)
        print(
            f"[visual-answer {case_id}] action={action} status={generated['status']} "
            f"changed={generated != original_response} cache_hit={cache_hit}",
            flush=True,
        )

    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "arabic-query-time-visual-fallback-development-v1",
        "configuration": {
            "answer_model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "reasoning_effort": REASONING_EFFORT,
            "untouched_cases": "copied byte-for-value from frozen raw records",
            "invalid_uncertain_or_unavailable_visual": "fail_closed_without_answer_call",
        },
        "inputs": {
            "answer_suite_sha256": suite_hash,
            "raw_retrieved_result_sha256": raw_hash,
            "routing_receipt_sha256": sha256_file(routing_receipt_path),
            "visual_result_sha256": visual_hash,
        },
        "counts": {
            "all_cases": len(records),
            "routed_cases": len(routes),
            "untouched_cases": len(records) - len(routes),
            "hosted_answer_requests_this_run": hosted_requests,
            "answer_cache_hits_this_run": cache_hits,
        },
        "automatic_metrics": {
            "overall": _aggregate(records),
            "fr": _aggregate([record for record in records if record["language"] == "fr"]),
            "ar": _aggregate([record for record in records if record["language"] == "ar"]),
        },
        "new_answer_usage": new_usage,
        "new_answer_latency_seconds": {
            "total": sum(new_latencies),
            "mean": statistics.mean(new_latencies) if new_latencies else 0.0,
            "median": statistics.median(new_latencies) if new_latencies else 0.0,
        },
        "cost_status": "not_calculated_provider_pricing_not_snapshotted",
        "records": records,
    }
    write_json_atomic(output_dir / "arabic_visual_answer_v1.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--routing-receipt", type=Path, required=True)
    parser.add_argument("--visual-result", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-public-documents", action="store_true")
    args = parser.parse_args()
    run_routed_visual_answers(
        suite_path=args.suite,
        raw_result_path=args.raw_result,
        routing_receipt_path=args.routing_receipt,
        visual_result_path=args.visual_result,
        dotenv_path=args.dotenv,
        output_dir=args.output_dir,
        confirm_public_documents=args.confirm_public_documents,
    )


if __name__ == "__main__":
    main()
