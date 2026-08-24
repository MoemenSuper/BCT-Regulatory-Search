"""Test a strict claim-linked answer contract on the frozen gold-evidence suite."""

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


MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "bct-claim-linked-answer-v1"
PROMPT_VERSION_V2 = "bct-claim-linked-answer-v2"
PROMPT_VERSION_V3 = "bct-claim-linked-answer-v3"
PROMPT_VERSION_V4 = "bct-claim-linked-answer-v4"
_STATUSES = {"answered", "insufficient_evidence", "clarification_needed", "out_of_scope"}
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "answer", "claims", "citations"],
    "properties": {
        "status": {"type": "string", "enum": sorted(_STATUSES)},
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "source", "page"],
                "properties": {
                    "evidence_id": {"type": "string"},
                    "source": {"type": "string"},
                    "page": {"type": "integer"},
                },
            },
        },
    },
}
_SYSTEM = """You are a high-trust Tunisian regulatory evidence assistant.

Return only the required JSON object.

For a supported regulatory question:
- status must be answered;
- answer in the question's language, directly and concisely;
- use only the supplied evidence;
- do not add inferred durations, future applicability, synonyms, explanations, or legal effects absent from evidence;
- preserve regulatory operation names and conditions verbatim when paraphrasing could change meaning;
- copy every number, date, percentage, code, and identifier you use exactly from evidence;
- break the answer into atomic claims and attach the supporting evidence ID to every claim;
- cite only evidence actually used.

When no evidence is supplied:
- if the request is outside BCT regulatory-document scope, status is out_of_scope and politely refuse;
- if a missing document, transfer type, allowance type, year, version, or other detail prevents a unique search, status is clarification_needed and ask one specific clarifying question;
- otherwise status is insufficient_evidence and explicitly abstain;
- claims and citations must be empty.

Never claim current applicability from an old document without explicit current-status evidence. Never invent a citation."""
_SCHEMA_V2 = json.loads(json.dumps(_SCHEMA))
_SCHEMA_V2["properties"]["answer"]["minLength"] = 1
_SYSTEM_V2 = _SYSTEM + """

For a question with no supplied evidence, apply this decision order exactly:
1. Use out_of_scope only when the subject is clearly unrelated to BCT regulatory documents, such as cooking or weather.
2. Use clarification_needed when the regulatory request has multiple plausible meanings and one missing discriminating detail, such as the allowance type or transfer type, prevents a unique search. Ask exactly one specific question.
3. Use insufficient_evidence for a requested current applicability/status or a future value, date, rate, document, or rule. Explicitly say that the supplied evidence cannot establish the current or future answer; do not ask for a document or year when time is the only missing fact.
4. Use insufficient_evidence for any other in-scope factual request that has no supporting evidence.

The answer field must always contain a concise user-facing explanation or clarifying question, including for abstentions and refusals."""
_SYSTEM_V3 = _SYSTEM_V2 + """

A request for an unknown future exchange rate, financial value, or BCT-related rule remains in scope even when it cannot be answered. With no supporting evidence, classify it as insufficient_evidence, never out_of_scope. Reserve out_of_scope for clearly unrelated subjects such as recipes, weather, restaurants, or sports."""
_SYSTEM_V4 = _SYSTEM_V3 + """

For a supported answer, preserve the evidence's exact literal fidelity:
- when answering an amount, rate, date, or time question, include the exact value and unit shown in evidence;
- when confirming that a named entity appears in a list or table, include the entity's row identifier when one is present;
- do not normalize, translate, or silently omit those supporting literals."""


def _contract(suite: dict[str, Any]) -> tuple[str, str, dict[str, Any], bool]:
    requested = suite.get("answer_experiment", {}).get("prompt_version", PROMPT_VERSION)
    if requested == PROMPT_VERSION:
        return PROMPT_VERSION, _SYSTEM, _SCHEMA, False
    if requested == PROMPT_VERSION_V2:
        return PROMPT_VERSION_V2, _SYSTEM_V2, _SCHEMA_V2, True
    if requested == PROMPT_VERSION_V3:
        return PROMPT_VERSION_V3, _SYSTEM_V3, _SCHEMA_V2, True
    if requested == PROMPT_VERSION_V4:
        return PROMPT_VERSION_V4, _SYSTEM_V4, _SCHEMA_V2, True
    raise ValueError(f"Unsupported structured answer prompt version: {requested}")


def parse_structured_answer(
    content: str, *, require_nonempty_answer: bool = False
) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Structured answer must be valid JSON") from error
    if not isinstance(value, dict) or set(value) != {"status", "answer", "claims", "citations"}:
        raise ValueError("Structured answer has unexpected top-level fields")
    if value["status"] not in _STATUSES or not isinstance(value["answer"], str):
        raise ValueError("Structured answer status or answer is invalid")
    if require_nonempty_answer and not value["answer"].strip():
        raise ValueError("Structured answer requires a non-empty answer")
    for claim in value["claims"]:
        if (
            not isinstance(claim, dict)
            or set(claim) != {"text", "evidence_ids"}
            or not isinstance(claim["text"], str)
            or not isinstance(claim["evidence_ids"], list)
            or any(not isinstance(item, str) for item in claim["evidence_ids"])
        ):
            raise ValueError("Structured answer claim is invalid")
    for citation in value["citations"]:
        if (
            not isinstance(citation, dict)
            or set(citation) != {"evidence_id", "source", "page"}
            or not isinstance(citation["evidence_id"], str)
            or not isinstance(citation["source"], str)
            or not isinstance(citation["page"], int)
        ):
            raise ValueError("Structured answer citation is invalid")
    if value["status"] != "answered" and (value["claims"] or value["citations"]):
        raise ValueError("Non-answered response must not contain claims or citations")
    return value


def structured_diagnostics(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    evidence_ids = {"E1"} if case.get("relevant") else set()
    claim_links_valid = all(
        bool(claim["evidence_ids"])
        and set(claim["evidence_ids"]).issubset(evidence_ids)
        for claim in response["claims"]
    )
    if case.get("relevant"):
        source = Path(str(case["expected_source"])).name.casefold()
        page = int(case["expected_page"])
        exact_citation = any(
            citation["evidence_id"] == "E1"
            and Path(citation["source"]).name.casefold() == source
            and citation["page"] == page
            for citation in response["citations"]
        )
        status_expected = response["status"] == "answered"
    else:
        exact_citation = not response["citations"]
        expected = case.get("expected_behavior")
        expected_status = {
            "clarify": "clarification_needed",
            "reject_out_of_scope": "out_of_scope",
            "abstain": "insufficient_evidence",
        }[expected]
        status_expected = response["status"] == expected_status
    return {
        "status_expected": status_expected,
        "exact_structured_citation": exact_citation,
        "claim_evidence_links_valid": claim_links_valid,
        "claim_count": len(response["claims"]),
        "citation_count": len(response["citations"]),
    }


def _usage(value: Any) -> dict[str, int]:
    return {
        field: int(getattr(value, field, 0) or 0)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [record for record in records if record["relevant"]]
    negative = [record for record in records if not record["relevant"]]
    return {
        "case_count": len(records),
        "relevant_count": len(relevant),
        "answered_status_rate": (
            sum(record["response"]["status"] == "answered" for record in relevant)
            / len(relevant)
            if relevant
            else None
        ),
        "exact_structured_citation_rate": (
            sum(record["structured_diagnostics"]["exact_structured_citation"] for record in relevant)
            / len(relevant)
            if relevant
            else None
        ),
        "claim_link_valid_rate": (
            sum(record["structured_diagnostics"]["claim_evidence_links_valid"] for record in relevant)
            / len(relevant)
            if relevant
            else None
        ),
        "negative_count": len(negative),
        "negative_expected_status_rate": (
            sum(record["structured_diagnostics"]["status_expected"] for record in negative)
            / len(negative)
            if negative
            else None
        ),
    }


def run_structured_answer_experiment(
    *,
    suite_path: Path,
    dotenv_path: Path,
    output_dir: Path,
    confirm_public_documents: bool,
) -> dict[str, Any]:
    if not confirm_public_documents:
        raise ValueError("Public-document confirmation is required for hosted answer calls")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not configured")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_hash = sha256_file(suite_path)
    prompt_version, system_prompt, response_schema, require_nonempty = _contract(suite)
    experiment = suite.get("answer_experiment", {})
    experiment_id = experiment.get(
        "experiment_id", "claim-linked-structured-answer-development-v1"
    )
    candidate_and_evidence = experiment.get(
        "candidate_and_evidence", "identical frozen gold-evidence suite"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = Groq(max_retries=3, timeout=90.0)
    records = []
    total = len(suite["cases"])
    for index, case in enumerate(suite["cases"], start=1):
        cache_path = cache_dir / f"{case['id']}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("model") != MODEL
                or cached.get("prompt_version") != prompt_version
                or cached.get("suite_sha256") != suite_hash
            ):
                raise ValueError(f"Structured answer cache differs for {case['id']}")
            response = parse_structured_answer(
                json.dumps(cached["response"], ensure_ascii=False),
                require_nonempty_answer=require_nonempty,
            )
        else:
            evidence = []
            if case.get("relevant"):
                evidence = [
                    {
                        "evidence_id": "E1",
                        "source": Path(str(case["expected_source"])).name,
                        "page": int(case["expected_page"]),
                        "text": case["evidence_quote"],
                    }
                ]
            user_payload = json.dumps(
                {"question": case["query"], "evidence": evidence},
                ensure_ascii=False,
            )
            started = time.perf_counter()
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "bct_claim_linked_answer",
                            "strict": True,
                            "schema": response_schema,
                        },
                    },
                    reasoning_effort="medium",
                    temperature=0,
                    seed=20260824,
                    max_completion_tokens=2048,
                )
            except RateLimitError as error:
                retry_after = (
                    error.response.headers.get("retry-after", "unknown")
                    if error.response is not None
                    else "unknown"
                )
                checkpoint = {
                    "status": "rate_limited",
                    "model": MODEL,
                    "completed": len(records),
                    "total": total,
                    "retry_after": retry_after,
                }
                write_json_atomic(output_dir / "checkpoint.json", checkpoint)
                print(
                    f"[structured-answer rate-limited] completed={len(records)}/{total}; "
                    f"retry_after={retry_after}",
                    flush=True,
                )
                return checkpoint
            response = parse_structured_answer(
                completion.choices[0].message.content or "",
                require_nonempty_answer=require_nonempty,
            )
            cached = {
                "id": case["id"],
                "model": MODEL,
                "prompt_version": prompt_version,
                "suite_sha256": suite_hash,
                "response_id": completion.id,
                "latency_seconds": time.perf_counter() - started,
                "usage": _usage(completion.usage),
                "response": response,
            }
            write_json_atomic(cache_path, cached)
        records.append(
            {
                "id": case["id"],
                "language": case["language"],
                "answer_suite_role": case["answer_suite_role"],
                "relevant": bool(case["relevant"]),
                "expected_behavior": case.get("expected_behavior"),
                "response": response,
                "latency_seconds": cached["latency_seconds"],
                "usage": cached["usage"],
                "automatic_audit": automatic_answer_audit(case, response["answer"]),
                "structured_diagnostics": structured_diagnostics(case, response),
            }
        )
        print(f"[structured-answer {index}/{total}] {case['id']}", flush=True)

    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "configuration": {
            "model": MODEL,
            "prompt_version": prompt_version,
            "response_format": "strict JSON schema",
            "candidate_and_evidence": candidate_and_evidence,
        },
        "inputs": {"answer_suite_sha256": suite_hash},
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
        "label_status": "strict-structure diagnostics only; semantic labels require evidence review",
        "limitations": [
            "Development-only gold-evidence prompt ablation.",
            "Valid claim links show declared provenance, not semantic entailment.",
            "Independent human confirmation is still required before release.",
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
    run_structured_answer_experiment(
        suite_path=args.suite,
        dotenv_path=args.dotenv,
        output_dir=args.output_dir,
        confirm_public_documents=args.confirm_public_documents,
    )


if __name__ == "__main__":
    main()
