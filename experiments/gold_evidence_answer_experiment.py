"""Run the current answer generator on a frozen gold-evidence development suite."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.documents import Document

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.numeric_fidelity_stress import critical_identifiers, critical_numbers
from llm import create_llm, generate_answer


MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "current-llm-py-gold-evidence-v1"
_PDF_CITATION = re.compile(r"\b[^\s,;()\[\]]+\.pdf\b", re.IGNORECASE)
_REFUSAL_MARKERS = (
    "not found",
    "information was not found",
    "cannot answer",
    "can't answer",
    "je ne peux",
    "pas trouv",
    "n'a pas été trouv",
    "n’a pas été trouv",
    "n'est pas disponible",
    "n’est pas disponible",
    "ne figurent pas",
    "ne permet pas",
    "لم يتم العثور",
    "لا أستطيع",
    "لا يمكن",
    "غير متوفر",
    "غير موجود",
    "لا تتوفر",
)
_CLARIFY_MARKERS = ("précis", "précisez", "quel ", "which ", "وضح", "تحديد", "أي ")


def _page_cited(answer: str, page: int) -> bool:
    escaped = re.escape(str(page))
    patterns = (
        rf"\bpage\s*[:#.-]?\s*{escaped}\b",
        rf"\bp\.?\s*{escaped}\b",
        rf"(?:الصفحة|صفحة|ص)\s*(?:رقم\s*)?[:#.-]?\s*{escaped}\b",
    )
    return any(re.search(pattern, answer, re.IGNORECASE) for pattern in patterns)


def _literal_numbers(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"(?<=\d)[\s\u00a0\u202f](?=\d{3}\b)", "", normalized)
    generic = set()
    for value in re.findall(r"\d+(?:[.,]\d+)?", normalized):
        characters = []
        for character in value:
            if character.isdigit():
                characters.append(str(unicodedata.digit(character)))
            elif character == ",":
                characters.append(".")
            else:
                characters.append(character)
        generic.add("".join(characters))
    return critical_numbers(normalized) | generic


def automatic_answer_audit(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Compute literal/citation checks; never promote them to semantic correctness."""
    normalized_answer = answer.casefold()
    cited_filenames = sorted(set(_PDF_CITATION.findall(answer)))
    if not case.get("relevant"):
        expected_behavior = case.get("expected_behavior")
        refusal = any(marker in normalized_answer for marker in _REFUSAL_MARKERS)
        clarification = "?" in answer or any(
            marker in normalized_answer for marker in _CLARIFY_MARKERS
        )
        preliminary_safe = not cited_filenames and (
            clarification if expected_behavior == "clarify" else refusal
        )
        return {
            "expected_behavior": expected_behavior,
            "cited_filenames": cited_filenames,
            "preliminary_refusal_marker": refusal,
            "preliminary_clarification_marker": clarification,
            "preliminary_safe_response": preliminary_safe,
            "manual_review_required": True,
        }

    expected_answer = str(case.get("expected_answer") or "")
    expected_numbers = _literal_numbers(expected_answer)
    expected_identifiers = {
        value.strip("./-") for value in critical_identifiers(expected_answer)
    }
    answer_numbers = _literal_numbers(answer)
    answer_identifiers = {value.strip("./-") for value in critical_identifiers(answer)}
    allowed_numbers = _literal_numbers(
        "\n".join(
            (
                str(case.get("query") or ""),
                str(case.get("evidence_quote") or ""),
                str(case.get("expected_source") or ""),
                str(case.get("expected_page") or ""),
            )
        )
    )
    source = Path(str(case["expected_source"])).name
    return {
        "expected_numbers": sorted(expected_numbers),
        "matched_expected_numbers": sorted(expected_numbers & answer_numbers),
        "number_recall": (
            len(expected_numbers & answer_numbers) / len(expected_numbers)
            if expected_numbers
            else None
        ),
        "unmatched_number_literals": sorted(answer_numbers - allowed_numbers),
        "expected_identifiers": sorted(expected_identifiers),
        "matched_expected_identifiers": sorted(expected_identifiers & answer_identifiers),
        "identifier_recall": (
            len(expected_identifiers & answer_identifiers) / len(expected_identifiers)
            if expected_identifiers
            else None
        ),
        "expected_source_present": source.casefold() in normalized_answer,
        "expected_page_cited": _page_cited(answer, int(case["expected_page"])),
        "cited_filenames": cited_filenames,
        "manual_review_required": True,
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [record for record in records if record["relevant"]]
    negative = [record for record in records if not record["relevant"]]
    numeric = [
        record["automatic_audit"]["number_recall"]
        for record in relevant
        if record["automatic_audit"]["number_recall"] is not None
    ]
    identifiers = [
        record["automatic_audit"]["identifier_recall"]
        for record in relevant
        if record["automatic_audit"]["identifier_recall"] is not None
    ]
    return {
        "case_count": len(records),
        "relevant_count": len(relevant),
        "negative_count": len(negative),
        "expected_source_literal_rate": (
            sum(record["automatic_audit"]["expected_source_present"] for record in relevant)
            / len(relevant)
            if relevant
            else None
        ),
        "expected_page_label_rate": (
            sum(record["automatic_audit"]["expected_page_cited"] for record in relevant)
            / len(relevant)
            if relevant
            else None
        ),
        "mean_expected_number_recall": statistics.mean(numeric) if numeric else None,
        "full_expected_number_recall_rate": (
            sum(value == 1.0 for value in numeric) / len(numeric) if numeric else None
        ),
        "unmatched_number_case_count": sum(
            bool(record["automatic_audit"]["unmatched_number_literals"])
            for record in relevant
        ),
        "mean_expected_identifier_recall": (
            statistics.mean(identifiers) if identifiers else None
        ),
        "preliminary_safe_negative_rate": (
            sum(record["automatic_audit"]["preliminary_safe_response"] for record in negative)
            / len(negative)
            if negative
            else None
        ),
    }


def run_gold_evidence_answers(
    *,
    suite_path: Path,
    dotenv_path: Path,
    output_dir: Path,
    confirm_public_documents: bool,
) -> dict[str, Any]:
    if not confirm_public_documents:
        raise ValueError("Public-document confirmation is required for hosted answer calls")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    llm = create_llm()
    records = []
    total = len(suite["cases"])
    for index, case in enumerate(suite["cases"], start=1):
        cache_path = cache_dir / f"{case['id']}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("model") != MODEL
                or cached.get("prompt_version") != PROMPT_VERSION
                or cached.get("suite_sha256") != sha256_file(suite_path)
            ):
                raise ValueError(f"Answer cache configuration differs for {case['id']}")
        else:
            documents = []
            if case.get("relevant"):
                documents = [
                    Document(
                        page_content=str(case["evidence_quote"]),
                        metadata={
                            "source": Path(str(case["expected_source"])).name,
                            "page_label": int(case["expected_page"]),
                        },
                    )
                ]
            started = time.perf_counter()
            try:
                answer = str(generate_answer(llm, case["query"], documents))
            except RateLimitError as error:
                retry_after = error.response.headers.get("retry-after", "unknown")
                checkpoint = {
                    "status": "rate_limited",
                    "model": MODEL,
                    "completed": len(records),
                    "total": total,
                    "retry_after": retry_after,
                }
                write_json_atomic(output_dir / "checkpoint.json", checkpoint)
                print(
                    f"[answer rate-limited] completed={len(records)}/{total}; "
                    f"retry_after={retry_after}",
                    flush=True,
                )
                return checkpoint
            cached = {
                "id": case["id"],
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "suite_sha256": sha256_file(suite_path),
                "latency_seconds": time.perf_counter() - started,
                "answer": answer,
            }
            write_json_atomic(cache_path, cached)
        record = {
            "id": case["id"],
            "language": case["language"],
            "answer_suite_role": case["answer_suite_role"],
            "relevant": bool(case["relevant"]),
            "expected_behavior": case.get("expected_behavior"),
            "answer": cached["answer"],
            "latency_seconds": cached["latency_seconds"],
            "automatic_audit": automatic_answer_audit(case, cached["answer"]),
        }
        records.append(record)
        print(f"[answer {index}/{total}] {case['id']}", flush=True)

    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "current-generator-gold-evidence-development-v1",
        "configuration": {
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "context": "verified evidence snippet and exact source/page only",
            "negative_context": "empty",
        },
        "inputs": {"answer_suite_sha256": sha256_file(suite_path)},
        "metrics": {
            "overall": _aggregate(records),
            "fr": _aggregate([record for record in records if record["language"] == "fr"]),
            "ar": _aggregate([record for record in records if record["language"] == "ar"]),
        },
        "latency_seconds": {
            "mean": statistics.mean(record["latency_seconds"] for record in records),
            "median": statistics.median(record["latency_seconds"] for record in records),
        },
        "label_status": (
            "automatic literal and citation-format diagnostics only; semantic answer correctness, "
            "claim support, citation entailment, and safe abstention require independent human review"
        ),
        "limitations": [
            "Development-only gold-evidence test; no generalization claim.",
            "Literal filename/page checks do not prove citation entailment.",
            "Number/identifier recall does not prove complete answer correctness.",
            "Unmatched number literals are review flags, not automatic hallucination labels; presentation ordinals may be benign.",
            "Refusal markers are preliminary heuristics and are not human safety labels.",
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
    run_gold_evidence_answers(
        suite_path=args.suite,
        dotenv_path=args.dotenv,
        output_dir=args.output_dir,
        confirm_public_documents=args.confirm_public_documents,
    )


if __name__ == "__main__":
    main()
