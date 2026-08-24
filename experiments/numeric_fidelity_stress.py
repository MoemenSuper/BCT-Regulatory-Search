"""Freeze numeric and identifier fidelity checks for triggered Arabic OCR pages."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


_DIGIT_SEQUENCE = re.compile(r"\d+(?:[.,]\d+)?", re.UNICODE)
_IDENTIFIER = re.compile(
    r"(?<![a-z0-9])(?=[a-z0-9./-]*[a-z])(?=[a-z0-9./-]*\d)[a-z0-9./-]+"
)


def _normalized(text: str) -> str:
    characters = []
    for character in unicodedata.normalize("NFKC", text).casefold():
        if character.isdigit():
            characters.append(str(unicodedata.digit(character)))
        elif character in {"\u066b", ","}:
            characters.append(".")
        elif character == "\u066c":
            characters.append("")
        else:
            characters.append(character)
    return "".join(characters)


def critical_numbers(text: str) -> set[str]:
    normalized = _normalized(text)
    without_identifiers = _IDENTIFIER.sub(" ", normalized)
    return set(_DIGIT_SEQUENCE.findall(without_identifiers))


def critical_identifiers(text: str) -> set[str]:
    return set(_IDENTIFIER.findall(_normalized(text)))


def _fidelity(
    expected_numbers: set[str],
    expected_identifiers: set[str],
    actual: str,
) -> dict[str, Any]:
    actual_numbers = critical_numbers(actual)
    actual_identifiers = critical_identifiers(actual)
    return {
        "number_recall": (
            len(expected_numbers & actual_numbers) / len(expected_numbers)
            if expected_numbers
            else None
        ),
        "identifier_recall": (
            len(expected_identifiers & actual_identifiers) / len(expected_identifiers)
            if expected_identifiers
            else None
        ),
        "matched_numbers": sorted(expected_numbers & actual_numbers),
        "missing_numbers": sorted(expected_numbers - actual_numbers),
        "matched_identifiers": sorted(expected_identifiers & actual_identifiers),
        "missing_identifiers": sorted(expected_identifiers - actual_identifiers),
    }


def evaluate_numeric_fidelity(expected: str, native: str, ocr: str) -> dict[str, Any]:
    expected_numbers = critical_numbers(expected)
    expected_identifiers = critical_identifiers(expected)
    return {
        "expected_numbers": sorted(expected_numbers),
        "expected_identifiers": sorted(expected_identifiers),
        "native": _fidelity(expected_numbers, expected_identifiers, native),
        "ocr": _fidelity(expected_numbers, expected_identifiers, ocr),
        "union": _fidelity(
            expected_numbers,
            expected_identifiers,
            f"{native}\n{ocr}",
        ),
    }


def _aggregate(cases: list[dict[str, Any]], method: str) -> dict[str, Any]:
    numeric = [case[method]["number_recall"] for case in cases]
    identifiers = [
        case[method]["identifier_recall"]
        for case in cases
        if case[method]["identifier_recall"] is not None
    ]
    return {
        "case_count": len(cases),
        "identifier_case_count": len(identifiers),
        "mean_number_recall": statistics.mean(numeric) if numeric else 0.0,
        "full_number_recall_rate": sum(value == 1.0 for value in numeric) / len(numeric)
        if numeric
        else 0.0,
        "mean_identifier_recall": statistics.mean(identifiers) if identifiers else None,
        "full_identifier_recall_rate": (
            sum(value == 1.0 for value in identifiers) / len(identifiers)
            if identifiers
            else None
        ),
    }


def build_numeric_stress_suite(
    *,
    evaluation_path: Path,
    structured_manifest_path: Path,
    ocr_representation_manifest_path: Path,
    ocr_cache_dir: Path,
) -> dict[str, Any]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    structured_manifest = json.loads(
        structured_manifest_path.read_text(encoding="utf-8")
    )
    ocr_manifest = json.loads(
        ocr_representation_manifest_path.read_text(encoding="utf-8")
    )
    structured_by_source = {
        record["source"].casefold(): record
        for record in structured_manifest["records"].values()
    }
    ocr_by_page = {
        (record["source"].casefold(), int(record["page"])): record
        for record in ocr_manifest["ocr_page_records"]
    }
    document_cache: dict[str, dict[str, Any]] = {}
    cases = []
    for case in evaluation:
        if not case.get("relevant") or not critical_numbers(case.get("evidence_quote", "")):
            continue
        source = str(case["expected_source"]).replace("\\", "/").rsplit("/", 1)[-1]
        page_number = int(case["expected_page"])
        key = (source.casefold(), page_number)
        ocr_record = ocr_by_page.get(key)
        if ocr_record is None:
            continue
        structured_record = structured_by_source.get(source.casefold())
        if structured_record is None:
            raise ValueError(f"Structured manifest is missing {source}")
        if structured_record["sha256"] != ocr_record["source_pdf_sha256"]:
            raise ValueError(f"OCR and structured source hashes differ for {source}")
        if source.casefold() not in document_cache:
            document_cache[source.casefold()] = json.loads(
                Path(structured_record["artifact"]).read_text(encoding="utf-8")
            )
        page = next(
            (
                value
                for value in document_cache[source.casefold()]["pages"]
                if int(value["page_number"]) == page_number
            ),
            None,
        )
        if page is None:
            raise ValueError(f"Structured artifact is missing {source} page {page_number}")
        ocr_path = ocr_cache_dir / (
            f"{structured_record['sha256'].casefold()}-p{page_number}-arabic_rapidocr.json"
        )
        if sha256_file(ocr_path) != ocr_record["ocr_cache_sha256"]:
            raise ValueError(f"OCR cache hash differs for {source} page {page_number}")
        ocr_payload = json.loads(ocr_path.read_text(encoding="utf-8"))
        fidelity = evaluate_numeric_fidelity(
            case["evidence_quote"],
            page.get("metadata", {}).get("native_raw_text", page.get("raw_text", "")),
            ocr_payload["text"],
        )
        cases.append(
            {
                "id": case["id"],
                "language": case["language"],
                "category": case.get("category"),
                "evidence_method": case.get("evidence_method"),
                "expected_source": source,
                "expected_page": page_number,
                **fidelity,
            }
        )

    improvements = [
        case["id"]
        for case in cases
        if case["ocr"]["number_recall"] > case["native"]["number_recall"]
    ]
    regressions = [
        case["id"]
        for case in cases
        if case["ocr"]["number_recall"] < case["native"]["number_recall"]
    ]
    return {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "suite_type": "development_triggered_arabic_numeric_identifier_fidelity",
        "selection": (
            "All relevant development cases with verified numeric evidence whose expected "
            "page was selected by the previously frozen Arabic fallback gate"
        ),
        "inputs": {
            "evaluation_sha256": sha256_file(evaluation_path),
            "structured_manifest_sha256": sha256_file(structured_manifest_path),
            "ocr_representation_manifest_sha256": sha256_file(
                ocr_representation_manifest_path
            ),
        },
        "metrics": {
            "native": _aggregate(cases, "native"),
            "ocr": _aggregate(cases, "ocr"),
            "union": _aggregate(cases, "union"),
            "ocr_number_improvement_count": len(improvements),
            "ocr_number_regression_count": len(regressions),
        },
        "ocr_number_improvements": improvements,
        "ocr_number_regressions": regressions,
        "limitations": [
            "Development evidence snippets only; this suite is frozen for future extractor comparisons, not generalization claims.",
            "Recall measures answer-bearing verified literals, not every number or identifier on the page.",
            "Numbers present elsewhere on a page are not labeled hallucinations or false positives.",
            "Native-plus-OCR union is a diagnostic upper bound; answer generation must resolve disagreements rather than concatenate blindly.",
        ],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--ocr-representation-manifest", type=Path, required=True)
    parser.add_argument("--ocr-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_numeric_stress_suite(
        evaluation_path=args.evaluation,
        structured_manifest_path=args.structured_manifest,
        ocr_representation_manifest_path=args.ocr_representation_manifest,
        ocr_cache_dir=args.ocr_cache_dir,
    )
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
