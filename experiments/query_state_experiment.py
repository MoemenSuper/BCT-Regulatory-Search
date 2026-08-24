"""Evaluate explicit query-state classification with deterministic answer statuses."""

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
from experiments.structured_answer_experiment import structured_diagnostics


MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "bct-query-state-v1"
_SCOPES = {"bct_regulatory_or_financial", "clearly_unrelated"}
_TEMPORAL_STATES = {"current_or_future", "not_current_or_future"}
_AMBIGUITY_STATES = {"missing_discriminating_detail", "sufficiently_specific"}
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scope", "temporal_state", "ambiguity", "missing_detail"],
    "properties": {
        "scope": {"type": "string", "enum": sorted(_SCOPES)},
        "temporal_state": {
            "type": "string",
            "enum": sorted(_TEMPORAL_STATES),
        },
        "ambiguity": {"type": "string", "enum": sorted(_AMBIGUITY_STATES)},
        "missing_detail": {"type": "string"},
    },
}
_SYSTEM = """Classify a user query for a high-trust Tunisian regulatory search system.

Return only the required JSON object. Do not answer the query.

scope:
- bct_regulatory_or_financial when the subject concerns BCT regulations, banking, transfers, allowances, exchange rates, imports, currency, or financial rules;
- clearly_unrelated only for subjects such as recipes, weather, restaurants, or sports.

temporal_state:
- current_or_future when the requested fact explicitly depends on today, current applicability, tomorrow, a future date, prediction, or a future rule;
- not_current_or_future otherwise.

ambiguity:
- missing_discriminating_detail when multiple regulatory searches are plausible because a type, category, document, transaction, allowance, or similar detail is missing;
- sufficiently_specific otherwise.

missing_detail must name the one missing detail when ambiguity is missing_discriminating_detail, otherwise it must be an empty string.

Examples:
- A request for tomorrow's exchange rate is financial, current_or_future, and sufficiently_specific.
- A request asking what travel allowance is available without naming the allowance type is financial, not_current_or_future, and missing_discriminating_detail.
- A request for required transfer documents without naming the transfer type is financial, not_current_or_future, and missing_discriminating_detail.
- A recipe or weather request is clearly_unrelated."""


def parse_query_state(content: str) -> dict[str, str]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Query state must be valid JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "scope",
        "temporal_state",
        "ambiguity",
        "missing_detail",
    }:
        raise ValueError("Query state has unexpected fields")
    if value["scope"] not in _SCOPES:
        raise ValueError("Query state scope is invalid")
    if value["temporal_state"] not in _TEMPORAL_STATES:
        raise ValueError("Query state temporal_state is invalid")
    if value["ambiguity"] not in _AMBIGUITY_STATES:
        raise ValueError("Query state ambiguity is invalid")
    if not isinstance(value["missing_detail"], str):
        raise ValueError("Query state missing_detail is invalid")
    missing = value["missing_detail"].strip()
    if value["ambiguity"] == "missing_discriminating_detail" and not missing:
        raise ValueError("Ambiguous query state requires a missing detail")
    if value["ambiguity"] == "sufficiently_specific" and missing:
        raise ValueError("Specific query state must not contain a missing detail")
    return {**value, "missing_detail": missing}


def derive_status(state: dict[str, str]) -> str:
    if state["scope"] == "clearly_unrelated":
        return "out_of_scope"
    if state["temporal_state"] == "current_or_future":
        return "insufficient_evidence"
    if state["ambiguity"] == "missing_discriminating_detail":
        return "clarification_needed"
    return "insufficient_evidence"


def _answer(status: str, state: dict[str, str], language: str) -> str:
    if status == "clarification_needed":
        if language == "ar":
            return f"يرجى توضيح التفصيل التالي: {state['missing_detail']}"
        return f"Veuillez préciser le détail suivant : {state['missing_detail']}"
    if status == "out_of_scope":
        if language == "ar":
            return "عذرًا، هذا الطلب خارج نطاق الوثائق التنظيمية والمالية للبنك المركزي التونسي."
        return "Désolé, cette demande sort du périmètre documentaire réglementaire et financier de la BCT."
    if language == "ar":
        return "لا تتوفر أدلة كافية وموثوقة للإجابة عن هذا الطلب."
    return "Les éléments disponibles ne permettent pas de répondre de manière fiable à cette demande."


def _usage(value: Any) -> dict[str, int]:
    return {
        field: int(getattr(value, field, 0) or 0)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def run_query_state_experiment(
    *,
    suite_path: Path,
    dotenv_path: Path,
    output_dir: Path,
    confirm_public_documents: bool,
) -> dict[str, Any]:
    if not confirm_public_documents:
        raise ValueError("Public-document confirmation is required for hosted calls")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if any(case.get("relevant") for case in suite["cases"]):
        raise ValueError("Query-state gate accepts only negative or ambiguous cases")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured")
    suite_hash = sha256_file(suite_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = Groq(max_retries=3, timeout=90.0)
    records = []
    for index, case in enumerate(suite["cases"], start=1):
        cache_path = cache_dir / f"{case['id']}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("model") != MODEL
                or cached.get("prompt_version") != PROMPT_VERSION
                or cached.get("suite_sha256") != suite_hash
            ):
                raise ValueError(f"Query-state cache differs for {case['id']}")
            state = parse_query_state(
                json.dumps(cached["query_state"], ensure_ascii=False)
            )
        else:
            started = time.perf_counter()
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": case["query"]},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "bct_query_state",
                            "strict": True,
                            "schema": _SCHEMA,
                        },
                    },
                    reasoning_effort="medium",
                    temperature=0,
                    seed=20260824,
                    max_completion_tokens=1024,
                )
            except RateLimitError as error:
                retry_after = (
                    error.response.headers.get("retry-after", "unknown")
                    if error.response is not None
                    else "unknown"
                )
                checkpoint = {
                    "status": "rate_limited",
                    "completed": len(records),
                    "total": len(suite["cases"]),
                    "retry_after": retry_after,
                }
                write_json_atomic(output_dir / "checkpoint.json", checkpoint)
                return checkpoint
            state = parse_query_state(completion.choices[0].message.content or "")
            cached = {
                "id": case["id"],
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "suite_sha256": suite_hash,
                "response_id": completion.id,
                "latency_seconds": time.perf_counter() - started,
                "usage": _usage(completion.usage),
                "query_state": state,
            }
            write_json_atomic(cache_path, cached)
        status = derive_status(state)
        response = {
            "status": status,
            "answer": _answer(status, state, case["language"]),
            "claims": [],
            "citations": [],
        }
        records.append(
            {
                "id": case["id"],
                "language": case["language"],
                "answer_suite_role": case["answer_suite_role"],
                "relevant": False,
                "expected_behavior": case["expected_behavior"],
                "query_state": state,
                "response": response,
                "latency_seconds": cached["latency_seconds"],
                "usage": cached["usage"],
                "automatic_audit": automatic_answer_audit(case, response["answer"]),
                "structured_diagnostics": structured_diagnostics(case, response),
            }
        )
        print(f"[query-state {index}/{len(suite['cases'])}] {case['id']} {status}", flush=True)
    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "explicit-query-state-development-v1",
        "configuration": {
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "classification": "scope plus temporal state plus ambiguity",
            "status_mapping": "deterministic precedence: unrelated, temporal, ambiguity, fallback",
            "user_facing_text": "deterministic bilingual templates",
        },
        "inputs": {"answer_suite_sha256": suite_hash},
        "automatic_metrics": {
            "negative_expected_status_rate": sum(
                record["structured_diagnostics"]["status_expected"] for record in records
            )
            / len(records)
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
            "Development-only query-state gate over eight inspected cases.",
            "The classifier still uses a hosted language model and is not proof of deterministic semantic correctness.",
            "No retrieved evidence is supplied, so this does not solve evidence-sufficiency detection.",
        ],
        "records": records,
    }
    write_json_atomic(output_dir / "result.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-public-documents", action="store_true")
    args = parser.parse_args()
    run_query_state_experiment(
        suite_path=args.suite,
        dotenv_path=args.dotenv,
        output_dir=args.output_dir,
        confirm_public_documents=args.confirm_public_documents,
    )


if __name__ == "__main__":
    main()
