"""Build deterministic frozen development stress suites from benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic

_TOKEN = re.compile(r"\w+", re.UNICODE)
_ARABIC_TOKEN = re.compile(r"^[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+$")


def _normalized_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return set(_TOKEN.findall(normalized))


def _coverage(expected: str, actual: str) -> float:
    tokens = _normalized_tokens(expected)
    return len(tokens & _normalized_tokens(actual)) / len(tokens) if tokens else 0.0


def _diagnostics(case: dict[str, Any], page: dict[str, Any]) -> dict[str, Any]:
    text = str(page.get("raw_text", ""))
    visible = [character for character in text if not character.isspace()]
    arabic_tokens = [token for token in _TOKEN.findall(text) if _ARABIC_TOKEN.match(token)]
    source = str(case.get("expected_source", ""))
    year_match = re.search(r"(?:19|20)\d{2}", source)
    expected_year = year_match.group(0) if year_match else None
    total_visible = max(len(visible), 1)
    return {
        "extraction_method": page.get("extraction_method"),
        "quality_score": page.get("quality_score"),
        "quality_flags": page.get("quality_flags", []),
        "character_count": len(text),
        "arabic_character_ratio": sum(
            "\u0600" <= character <= "\u06ff"
            or "\u0750" <= character <= "\u077f"
            or "\u08a0" <= character <= "\u08ff"
            for character in visible
        )
        / total_visible,
        "latin_character_ratio": sum(
            "a" <= character.casefold() <= "z" or "\u00c0" <= character <= "\u024f"
            for character in visible
        )
        / total_visible,
        "single_arabic_token_ratio": (
            sum(len(token) == 1 for token in arabic_tokens) / len(arabic_tokens)
            if arabic_tokens
            else 0.0
        ),
        "western_digit_count": sum(character.isascii() and character.isdigit() for character in text),
        "arabic_indic_digit_count": sum("\u0660" <= character <= "\u0669" for character in text),
        "expected_year": expected_year,
        "expected_year_literal_present": expected_year in text if expected_year else None,
        "expected_year_reversed_present": expected_year[::-1] in text if expected_year else None,
        "evidence_token_coverage": _coverage(str(case.get("evidence_quote", "")), text),
    }


def _selection_key(failure_id: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{failure_id}|{candidate_id}".encode()).hexdigest()


def build_extraction_stress_suite(
    evaluation: list[dict[str, Any]],
    result: dict[str, Any],
    pages_by_source: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Pair all extraction failures with deterministic language/category controls."""
    records_by_id = {record["id"]: record for record in result["records"]}
    failures = [
        case
        for case in evaluation
        if case.get("relevant")
        and "evidence_missing_because_of_extraction"
        in records_by_id[case["id"]].get("failure_categories", [])
    ]
    controls = [
        case
        for case in evaluation
        if case.get("relevant")
        and records_by_id[case["id"]]["result"].get("exact_page_rank") is not None
        and records_by_id[case["id"]]["result"]["exact_page_rank"] <= 5
        and "evidence_missing_because_of_extraction"
        not in records_by_id[case["id"]].get("failure_categories", [])
    ]

    selected_controls: list[dict[str, Any]] = []
    used_control_ids: set[str] = set()
    unmatched = 0
    for failure in sorted(failures, key=lambda case: case["id"]):
        same_language = [
            candidate
            for candidate in controls
            if candidate["id"] not in used_control_ids
            and candidate.get("language") == failure.get("language")
        ]
        same_category = [
            candidate
            for candidate in same_language
            if candidate.get("category") == failure.get("category")
        ]
        pool = same_category or same_language
        if not pool:
            unmatched += 1
            continue
        chosen = min(pool, key=lambda candidate: _selection_key(failure["id"], candidate["id"]))
        used_control_ids.add(chosen["id"])
        selected_controls.append(chosen)

    entries = []
    for role, selected in (("failure", failures), ("control", selected_controls)):
        for case in selected:
            source = str(case["expected_source"]).replace("\\", "/").rsplit("/", 1)[-1]
            page_number = int(case["expected_page"])
            page = pages_by_source.get((source.casefold(), page_number))
            if page is None:
                raise ValueError(f"Structured page is missing for {case['id']}: {source} page {page_number}")
            entries.append(
                {
                    "id": case["id"],
                    "role": role,
                    "language": case.get("language"),
                    "category": case.get("category"),
                    "expected_source": source,
                    "expected_page": page_number,
                    "diagnostics": _diagnostics(case, page),
                }
            )
    entries.sort(key=lambda case: (case["language"], case["role"], case["id"]))
    counts = {
        language: {
            role: sum(case["language"] == language and case["role"] == role for case in entries)
            for role in ("failure", "control")
        }
        for language in ("ar", "fr")
    }
    return {
        "suite_type": "development_extraction_quality",
        "counts": counts,
        "unmatched_control_count": unmatched,
        "selection": "all extraction failures plus unique deterministic same-language, same-category Top-5 controls where available",
        "cases": entries,
    }


def _load_pages(manifest_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = {}
    for record in manifest["records"].values():
        document = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        for page in document["pages"]:
            pages[(document["filename"].casefold(), int(page["page_number"]))] = page
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    suite = build_extraction_stress_suite(
        evaluation,
        result,
        _load_pages(args.structured_manifest),
    )
    artifact = {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "evaluation_sha256": sha256_file(args.evaluation),
            "result_sha256": sha256_file(args.result),
            "structured_manifest_sha256": sha256_file(args.structured_manifest),
        },
        **suite,
    }
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
