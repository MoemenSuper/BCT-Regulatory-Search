"""Final bounded qualification: Voyage Context-4, Voyage rerank, safe answers.

This runner deliberately leaves Neo4j expansion off. The graph remains useful
for explicit lineage UI features, but it produced no retrieval repairs in the
807-case control and is therefore not part of the default answer path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any

from dotenv import load_dotenv

from experiments.arabic_visual_fallback import build_routing_receipt
from experiments.arabic_visual_answer_experiment import (
    _page_key,
    _safe_abstention,
    enforce_routed_numeric_authority,
    prepare_routed_evidence,
    validate_visual_result_for_composition,
)
from experiments.artifacts import sha256_file, write_json_atomic
from experiments.candidate_diversity_ablation import diversify_ranked_pages
from experiments.enriched_graphrag_end_to_end import (
    _groq_client_pool_from_environment,
    structured_diagnostics,
)
from experiments.graph_temporal_benchmark import score_case
from experiments.gold_evidence_answer_experiment import automatic_answer_audit
from experiments.post_retrieval_answer_safety import apply_answer_safety
from experiments.provider_retrieval_matrix import Matrix, _prefix_texts
from experiments.retrieved_context_answer_experiment import _evidence_payload
from experiments.structured_answer_experiment import (
    MODEL,
    _SCHEMA_V2,
    _SYSTEM_V5,
    _usage,
    parse_structured_answer,
)


EXPERIMENT_ID = "final-voyage-enriched-qualification-v1"
EMBEDDING_MODEL = "voyage-context-4"
RERANKER_MODEL = "rerank-2.5"
PROMPT_VERSION = "bct-final-voyage-answer-v1"
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 1536

FINAL_SYSTEM_PROMPT = _SYSTEM_V5 + """

Additional release-safety rules:
- If the question does not identify which instrument or operation is meant and
  the evidence supports materially different interpretations, return
  clarification_needed. Do not silently choose one.
- A claim about what is currently in force requires explicit evidence of
  current legal status. Otherwise return insufficient_evidence.
- Never combine provisions from different instruments as though they came from
  one instrument. Label each instrument and version explicitly.
- Never repair or infer a number or date from a filename. Every asserted
  literal must appear in the claim-linked evidence.
- Return insufficient_evidence when the answer-bearing provision is incomplete.
"""


def runtime_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove all gold fields before retrieval or generation."""
    return [
        {
            key: case.get(key)
            for key in ("id", "query", "language", "category")
        }
        for case in cases
    ]


def score_retrieval_case(
    case: dict[str, Any],
    candidate_pages: list[dict[str, Any]],
    ranked_pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score old single-page and enriched multi-page gold after runtime ends."""
    scoring_case = case
    if not case.get("expected_sources"):
        scoring_case = {
            **case,
            "expected_sources": [
                {"source": case["expected_source"], "pages": [case["expected_page"]]}
            ],
        }
    return score_case(scoring_case, candidate_pages, ranked_pages)


def _load_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    cases = value if isinstance(value, list) else value.get("cases")
    if not isinstance(cases, list) or not all(isinstance(item, dict) for item in cases):
        raise ValueError("Suite must be a case list or an object with cases")
    ids = [str(case.get("id", "")) for case in cases]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Suite IDs must be non-empty and unique")
    return cases, sha256_file(path)


def build_matrix(
    *,
    provider_root: Path,
    cases: list[dict[str, Any]],
    native_chunks: Path,
    ocr_chunks: Path,
) -> Matrix:
    """Use the proven provider client without the old 697-case gold contract."""
    return Matrix.for_runtime_cases(
        root=provider_root,
        cases=runtime_cases(cases),
        native_chunks=native_chunks,
        ocr_chunks=ocr_chunks,
    )


def _retrieve(
    *,
    matrix: Matrix,
    cases: list[dict[str, Any]],
    suite_hash: str,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    native_hash = sha256_file(matrix.native_chunks)
    ocr_hash = sha256_file(matrix.ocr_chunks)
    name = (
        f"voyage-context-4-enriched-{suite_hash[:12].lower()}-"
        f"{native_hash[:8].lower()}-{ocr_hash[:8].lower()}"
    )
    binding = {
        "suite_sha256": suite_hash,
        "native_chunks_sha256": native_hash,
        "ocr_chunks_sha256": ocr_hash,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "candidate_pool_name": name,
    }
    previous = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    if previous and previous.get("binding") != binding:
        raise ValueError("Retrieval checkpoint binding differs")
    pools = matrix.candidates(name, "voyage", EMBEDDING_MODEL, contextual=True)
    records = previous.get("records", [])
    completed = {record["id"] for record in records}
    if len(completed) != len(records):
        raise ValueError("Retrieval checkpoint contains duplicate IDs")
    for index, case in enumerate(runtime_cases(cases), start=1):
        if case["id"] in completed:
            continue
        pool = pools[case["id"]]
        texts = _prefix_texts(pool, case["query"])
        started = time.perf_counter()
        scores = matrix.client.rerank("voyage", RERANKER_MODEL, case["query"], texts)
        ranked = diversify_ranked_pages(
            sorted(zip(pool, scores), key=lambda item: item[1], reverse=True)
        )
        records.append(
            {
                **case,
                "retrieved_evidence": _evidence_payload(ranked),
                "candidate_pages": [
                    {
                        "source": Path(str(item["document"].metadata.get("source", ""))).name,
                        "page": int(item["document"].metadata.get("page", -1)),
                    }
                    for item in pool
                ],
                "top20": [
                    {
                        "source": Path(str(item[0]["document"].metadata.get("source", ""))).name,
                        "page": int(item[0]["document"].metadata.get("page", -1)),
                        "score": float(item[1]),
                    }
                    for item in ranked[:20]
                ],
                "retrieval_latency_seconds": time.perf_counter() - started,
            }
        )
        if len(records) % 10 == 0:
            write_json_atomic(
                checkpoint_path,
                {"status": "running", "binding": binding, "records": records},
            )
        print(f"[voyage-retrieval {index}/{len(cases)}] {case['id']}", flush=True)
    write_json_atomic(
        checkpoint_path,
        {"status": "complete", "binding": binding, "records": records},
    )
    return records


def _answer_payload(query: str, evidence: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "question": query,
            "evidence": [
                {key: item[key] for key in ("evidence_id", "source", "page", "text")}
                for item in evidence
            ],
        },
        ensure_ascii=False,
    )


def _answer(
    *,
    client_pool: Any,
    case: dict[str, Any],
    evidence: list[dict[str, Any]],
    suite_hash: str,
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, int], bool, str]:
    payload = _answer_payload(case["query"], evidence)
    cache_path = cache_dir / f"{case['id']}.json"
    binding = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "suite_sha256": suite_hash,
        "user_payload": payload,
    }
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if any(cached.get(key) != value for key, value in binding.items()):
            raise ValueError(f"Answer cache binding differs for {case['id']}")
        response = parse_structured_answer(
            json.dumps(cached["response"], ensure_ascii=False),
            require_nonempty_answer=True,
        )
        return response, cached["usage"], True, cached.get("credential_slot", "GROQ_API_KEY")
    completion, slot = client_pool.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "bct_final_answer", "strict": True, "schema": _SCHEMA_V2},
        },
        reasoning_effort=REASONING_EFFORT,
        temperature=0,
        seed=20260902,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    response = parse_structured_answer(
        completion.choices[0].message.content or "", require_nonempty_answer=True
    )
    usage = _usage(completion.usage)
    write_json_atomic(
        cache_path,
        {
            **binding,
            "id": case["id"],
            "response_id": completion.id,
            "credential_slot": slot,
            "usage": usage,
            "response": response,
        },
    )
    return response, usage, False, slot


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_public_documents:
        raise ValueError("Public-document confirmation is required")
    load_dotenv(args.dotenv, override=False)
    cases, suite_hash = _load_cases(args.suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = args.output_dir / "voyage_retrieval.json"
    matrix = build_matrix(
        provider_root=args.provider_root,
        cases=cases,
        native_chunks=args.native_chunks,
        ocr_chunks=args.ocr_chunks,
    )
    retrieval = _retrieve(
        matrix=matrix,
        cases=cases,
        suite_hash=suite_hash,
        checkpoint_path=retrieval_path,
    )
    by_id = {record["id"]: record for record in retrieval}
    client_pool = _groq_client_pool_from_environment()
    cache_dir = args.output_dir / "answer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, case in enumerate(cases, start=1):
        retrieved = by_id[case["id"]]
        raw, usage, cache_hit, slot = _answer(
            client_pool=client_pool,
            case=case,
            evidence=retrieved["retrieved_evidence"],
            suite_hash=suite_hash,
            cache_dir=cache_dir,
        )
        response, safety = apply_answer_safety(
            question=case["query"],
            language=case["language"],
            response=raw,
            evidence=retrieved["retrieved_evidence"],
            query_state={"ambiguity": "none", "missing_detail": ""},
        )
        diagnostics = structured_diagnostics(case, response, retrieved["retrieved_evidence"])
        retrieval_score = (
            score_retrieval_case(case, retrieved["candidate_pages"], retrieved["top20"])
            if case.get("relevant", True)
            else None
        )
        records.append(
            {
                "id": case["id"],
                "query": case["query"],
                "language": case["language"],
                "category": case.get("category"),
                "retrieved_evidence": retrieved["retrieved_evidence"],
                "retrieval": retrieval_score,
                "response_before_safety": raw,
                "response": response,
                "safety": safety,
                "diagnostics": diagnostics,
                "automatic_answer_audit": (
                    automatic_answer_audit(case, response["answer"])
                    if case.get("relevant", True)
                    else None
                ),
                "usage": usage,
                "cache_hit": cache_hit,
                "credential_slot": slot,
            }
        )
        print(f"[voyage-answer {index}/{len(cases)}] {case['id']} status={response['status']} safety={safety['action']}", flush=True)
    raw_result = {
        "status": "complete_pending_visual_routing_and_semantic_review",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "configuration": {
            "retrieval": "Voyage Context-4 dense + BM25 + additive Arabic OCR; identity prefix; Voyage rerank-2.5; diverse top-5",
            "graph_default": "disabled_no_measured_retrieval_repairs",
            "answer_model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "post_retrieval_safety": True,
        },
        "inputs": {"suite_sha256": suite_hash},
        "records": records,
    }
    result_path = args.output_dir / "final_voyage_raw_answers.json"
    write_json_atomic(result_path, raw_result)
    risk = json.loads(args.digit_risk_result.read_text(encoding="utf-8"))
    suite_value = json.loads(args.suite.read_text(encoding="utf-8"))
    if isinstance(suite_value, list):
        suite_value = {"cases": suite_value}
    receipt = build_routing_receipt(
        suite=suite_value,
        retrieved_result=raw_result,
        risk_result=risk,
        input_hashes={
            "suite_sha256": suite_hash,
            "raw_result_sha256": sha256_file(result_path),
            "digit_risk_result_sha256": sha256_file(args.digit_risk_result),
        },
    )
    write_json_atomic(args.output_dir / "visual_routing_receipt.json", receipt)
    relevant = [record for record, case in zip(records, cases) if case.get("relevant", True)]
    negatives = [record for record, case in zip(records, cases) if not case.get("relevant", True)]
    metrics = {
        "relevant_count": len(relevant),
        "page_at_5": sum(r["retrieval"]["complete_required_page_pairs_at_5"] for r in relevant) / len(relevant),
        "structural_strict": sum(
            r["diagnostics"]["answered"]
            and r["diagnostics"]["complete_required_page_citations"]
            and r["diagnostics"]["citations_match_evidence"]
            and r["diagnostics"]["claim_evidence_links_valid"]
            for r in relevant
        ) / len(relevant),
        "negative_expected_status": sum(r["diagnostics"]["status_expected"] for r in negatives) / len(negatives),
        "visual_routed_cases": receipt["counts"]["routed_cases"],
        "usage": {
            field: sum(record["usage"][field] for record in records)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "semantic_correctness": "pending_manual_review",
    }
    final = {**raw_result, "status": "complete_pending_visual_and_semantic_review", "metrics": metrics}
    write_json_atomic(args.output_dir / "final_voyage_qualification.json", final)
    return final


def compose_visual(args: argparse.Namespace) -> dict[str, Any]:
    """Compose only Gemini-routed cases, then reapply deterministic safety."""
    load_dotenv(args.dotenv, override=False)
    cases, suite_hash = _load_cases(args.suite)
    by_case = {case["id"]: case for case in cases}
    raw_path = args.output_dir / "final_voyage_raw_answers.json"
    receipt_path = args.output_dir / "visual_routing_receipt.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    visual = json.loads(args.visual_result.read_text(encoding="utf-8"))
    validate_visual_result_for_composition(visual)
    if visual.get("inputs", {}).get("routing_receipt_sha256") != sha256_file(receipt_path):
        raise ValueError("Gemini visual result is bound to a different routing receipt")
    routes = {route["id"]: route for route in receipt["routes"]}
    visual_pages = {
        _page_key(page["source"], page["page"]): page for page in visual["pages"]
    }
    from experiments.gemini_visual_transcription_experiment import validate_cached_page

    client_pool = _groq_client_pool_from_environment()
    cache_dir = args.output_dir / "visual_answer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for record in raw["records"]:
        case = by_case[record["id"]]
        route = routes.get(record["id"])
        if route is None:
            records.append(record)
            continue
        evidence, action = prepare_routed_evidence(
            record=record,
            route=route,
            visual_pages=visual_pages,
            cache_validator=validate_cached_page,
        )
        if action == "generate":
            generated, usage, cache_hit, slot = _answer(
                client_pool=client_pool,
                case=case,
                evidence=evidence,
                suite_hash=suite_hash,
                cache_dir=cache_dir,
            )
            generated, authority = enforce_routed_numeric_authority(
                response=generated,
                evidence=evidence,
                routed_evidence_ids={page["evidence_id"] for page in route["pages"]},
                language=case["language"],
            )
            if authority != "generate":
                action = authority
        else:
            generated = _safe_abstention(case["language"])
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            cache_hit, slot = False, None
        response, safety = apply_answer_safety(
            question=case["query"],
            language=case["language"],
            response=generated,
            evidence=evidence,
            query_state={"ambiguity": "none", "missing_detail": ""},
        )
        records.append(
            {
                **record,
                "retrieved_evidence": evidence,
                "response_before_safety": generated,
                "response": response,
                "safety": safety,
                "diagnostics": structured_diagnostics(case, response, evidence),
                "automatic_answer_audit": (
                    automatic_answer_audit(case, response["answer"])
                    if case.get("relevant", True)
                    else None
                ),
                "usage": usage,
                "cache_hit": cache_hit,
                "credential_slot": slot,
                "visual_fallback": {"routed": True, "action": action},
            }
        )
    relevant = [record for record in records if by_case[record["id"]].get("relevant", True)]
    negatives = [record for record in records if not by_case[record["id"]].get("relevant", True)]
    result = {
        "status": "complete_pending_semantic_review",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "configuration": {
            **raw["configuration"],
            "visual_provider": "Google Gemini API",
            "visual_model": "gemini-3.7-flash",
            "visual_conflicts": "fail_closed",
        },
        "inputs": {
            **raw["inputs"],
            "raw_result_sha256": sha256_file(raw_path),
            "routing_receipt_sha256": sha256_file(receipt_path),
            "visual_result_sha256": sha256_file(args.visual_result),
        },
        "metrics": {
            "relevant_count": len(relevant),
            "structural_strict": sum(
                r["diagnostics"]["answered"]
                and r["diagnostics"]["complete_required_page_citations"]
                and r["diagnostics"]["citations_match_evidence"]
                and r["diagnostics"]["claim_evidence_links_valid"]
                for r in relevant
            ) / len(relevant),
            "negative_expected_status": sum(
                r["diagnostics"]["status_expected"] for r in negatives
            ) / len(negatives),
            "visual_routed_cases": len(routes),
            "visual_fail_closed_cases": sum(
                r.get("visual_fallback", {}).get("action", "generate") != "generate"
                for r in records
            ),
            "semantic_correctness": "pending_manual_review",
        },
        "records": records,
    }
    write_json_atomic(args.output_dir / "final_voyage_visual_qualified.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--native-chunks", type=Path, required=True)
    parser.add_argument("--ocr-chunks", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--digit-risk-result", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-result", type=Path)
    parser.add_argument("--confirm-public-documents", action="store_true")
    args = parser.parse_args()
    result = compose_visual(args) if args.visual_result else run(args)
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
