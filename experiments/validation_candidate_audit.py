"""Audit a newly curated validation candidate before any retrieval access."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.evaluation_protocol import _load_dataset


_STOPWORDS = {
    "avec", "dans", "des", "est", "quel", "quelle", "quels", "quelles",
    "les", "par", "pour", "sont", "sur", "une", "إلى", "التي", "على",
    "هذا", "في", "ما", "من", "هل", "هي", "هو",
}


def _source(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return [
        token
        for token in re.findall(r"[\w@.%]+", normalized, re.UNICODE)
        if len(token) > 2 and token not in _STOPWORDS
    ]


def _normalized_text(value: str) -> str:
    return " ".join(_tokens(value))


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def audit_validation_candidate(
    *,
    development_path: Path,
    validation_path: Path,
    structured_manifest_path: Path,
    minimum_evidence_token_coverage: float = 0.75,
    query_similarity_review_threshold: float = 0.5,
    evidence_similarity_review_threshold: float = 0.7,
) -> dict[str, Any]:
    development = _load_dataset(development_path)
    validation = _load_dataset(validation_path)
    manifest = json.loads(structured_manifest_path.read_text(encoding="utf-8"))
    development_pages = {
        (_source(case["expected_source"]).casefold(), int(case["expected_page"]))
        for case in development
        if case.get("relevant")
    }
    records_by_source = {
        _source(record["source"]).casefold(): record
        for record in manifest["records"].values()
    }
    page_cache: dict[str, dict[str, Any]] = {}
    records = []
    errors = []
    similarity_records = []
    for case in validation:
        query_matches = [
            (_jaccard(case["query"], development_case["query"]), development_case["id"])
            for development_case in development
        ]
        best_query = max(query_matches, default=(0.0, None))
        if any(
            _normalized_text(case["query"]) == _normalized_text(development_case["query"])
            for development_case in development
        ):
            errors.append(f"{case['id']}: normalized query duplicates development")
        best_evidence = (0.0, None)
        if case["relevant"]:
            relevant_development = [value for value in development if value.get("relevant")]
            evidence_matches = [
                (
                    _jaccard(case["evidence_quote"], development_case["evidence_quote"]),
                    development_case["id"],
                )
                for development_case in relevant_development
            ]
            best_evidence = max(evidence_matches, default=(0.0, None))
            if any(
                _normalized_text(case["evidence_quote"])
                == _normalized_text(development_case["evidence_quote"])
                for development_case in relevant_development
            ):
                errors.append(f"{case['id']}: normalized evidence duplicates development")
        similarity_records.append(
            {
                "id": case["id"],
                "max_query_jaccard": best_query[0],
                "nearest_query_development_id": best_query[1],
                "max_evidence_jaccard": best_evidence[0] if case["relevant"] else None,
                "nearest_evidence_development_id": best_evidence[1] if case["relevant"] else None,
            }
        )
        if not case["relevant"]:
            continue
        case_id = case["id"]
        key = (_source(case["expected_source"]).casefold(), int(case["expected_page"]))
        if key in development_pages:
            errors.append(f"{case_id}: exact source/page occurs in development")
        if case.get("verification_status") != (
            "agent_curated_pending_independent_human_verification"
        ):
            errors.append(f"{case_id}: verification status is not the pending-human marker")
        manifest_record = records_by_source.get(key[0])
        if manifest_record is None:
            errors.append(f"{case_id}: source is absent from the structured manifest")
            continue
        artifact_path = Path(manifest_record["artifact"])
        document = page_cache.setdefault(
            key[0], json.loads(artifact_path.read_text(encoding="utf-8"))
        )
        page = next(
            (value for value in document["pages"] if int(value["page_number"]) == key[1]),
            None,
        )
        if page is None:
            errors.append(f"{case_id}: page is absent from the StructuredDocument")
            continue
        evidence_tokens = _tokens(str(case["evidence_quote"]))
        page_tokens = set(_tokens(str(page.get("raw_text", ""))))
        coverage = (
            sum(token in page_tokens for token in evidence_tokens) / len(evidence_tokens)
            if evidence_tokens
            else 1.0
        )
        if coverage < minimum_evidence_token_coverage:
            errors.append(
                f"{case_id}: evidence token coverage {coverage:.3f} is below "
                f"{minimum_evidence_token_coverage:.3f}"
            )
        records.append(
            {
                "id": case_id,
                "source": _source(case["expected_source"]),
                "page": int(case["expected_page"]),
                "pdf_sha256": str(manifest_record["sha256"]).upper(),
                "structured_artifact_sha256": sha256_file(artifact_path),
                "evidence_token_coverage": coverage,
                "evidence_method": case.get("evidence_method"),
            }
        )
    if errors:
        raise ValueError("Validation candidate audit failed:\n" + "\n".join(errors))
    return {
        "status": "candidate_frozen_pending_independent_human_verification",
        "retrieval_access": "not_run",
        "inputs": {
            "development_sha256": sha256_file(development_path),
            "validation_candidate_sha256": sha256_file(validation_path),
            "structured_manifest_sha256": sha256_file(structured_manifest_path),
        },
        "counts": {
            "total": len(validation),
            "relevant": sum(case["relevant"] for case in validation),
            "negative_or_ambiguous": sum(not case["relevant"] for case in validation),
            "by_language": dict(sorted(Counter(case["language"] for case in validation).items())),
            "by_category": dict(sorted(Counter(case["category"] for case in validation).items())),
        },
        "leakage_audit": {
            "exact_development_source_page_overlap": 0,
            "note": "Sources and conservative document families may overlap because all current corpus sources are legacy development; exact labeled pages do not overlap.",
        },
        "near_duplicate_audit": {
            "normalized_query_duplicates": 0,
            "normalized_evidence_duplicates": 0,
            "query_jaccard_review_threshold": query_similarity_review_threshold,
            "evidence_jaccard_review_threshold": evidence_similarity_review_threshold,
            "maximum_query_jaccard": max(
                value["max_query_jaccard"] for value in similarity_records
            ),
            "maximum_evidence_jaccard": max(
                value["max_evidence_jaccard"] or 0.0 for value in similarity_records
            ),
            "flagged_pairs": [
                value
                for value in similarity_records
                if value["max_query_jaccard"] >= query_similarity_review_threshold
                or (value["max_evidence_jaccard"] or 0.0)
                >= evidence_similarity_review_threshold
            ],
            "records": similarity_records,
            "note": "Token Jaccard is a review screen, not proof that unflagged cases are semantically independent.",
        },
        "evidence_audit": {
            "minimum_required_token_coverage": minimum_evidence_token_coverage,
            "minimum_observed_token_coverage": min(
                record["evidence_token_coverage"] for record in records
            ),
            "visual_review_case_ids": [
                record["id"]
                for record in records
                if record["evidence_method"] == "visual_review"
            ],
        },
        "final_holdout": {
            "status": "prospective_not_available",
            "reason": "All 439 current sources are legacy development; a family-disjoint final holdout requires future BCT document families.",
        },
        "limitations": [
            "Labels were curated and source-checked by the agent, not independently adjudicated by a human.",
            "No retrieval, ranking, or answer model has been run on this candidate file.",
            "Page disjointness reduces exact-evidence reuse but does not remove source/family familiarity.",
        ],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = audit_validation_candidate(
        development_path=args.development,
        validation_path=args.validation,
        structured_manifest_path=args.structured_manifest,
    )
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
