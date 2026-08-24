"""Freeze and audit development, validation, and final-holdout datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


_DOCUMENT_ID = re.compile(
    r"(?i)^(?P<kind>cir|note|cb|nb|ci)[_ -]?(?P<year>\d{4})[_ -]?(?P<number>\d+).*?[_ -]?(?P<language>fr|ar)\.pdf$"
)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Evaluation dataset must be a JSON array: {path}")
    if not value:
        raise ValueError(f"Evaluation dataset must not be empty: {path}")
    if any(not isinstance(case, dict) for case in value):
        raise ValueError(f"Every evaluation case must be a JSON object: {path}")
    ids = [case.get("id") for case in value]
    if any(not case_id for case_id in ids):
        raise ValueError(f"Every evaluation case needs a non-empty id: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Evaluation case IDs must be unique: {path}")
    for case in value:
        missing = [field for field in ("query", "language", "relevant") if field not in case]
        if missing:
            raise ValueError(f"Case {case['id']} is missing required fields: {', '.join(missing)}")
        if case["language"] not in {"fr", "ar"}:
            raise ValueError(f"Case {case['id']} language must be 'fr' or 'ar'")
        if not isinstance(case["relevant"], bool):
            raise ValueError(f"Case {case['id']} relevant must be Boolean")
        if case["relevant"]:
            missing = [
                field
                for field in ("expected_source", "expected_page", "evidence_quote")
                if case.get(field) in (None, "")
            ]
            if missing:
                raise ValueError(
                    f"Relevant case {case['id']} is missing evidence fields: {', '.join(missing)}"
                )
    relevant_languages = {case["language"] for case in value if case["relevant"]}
    if relevant_languages != {"fr", "ar"}:
        raise ValueError(
            f"Evaluation dataset must contain relevant French and Arabic cases: {path}"
        )
    return value


def _source(case: dict[str, Any]) -> str | None:
    source = case.get("expected_source")
    return str(source).replace("\\", "/").rsplit("/", 1)[-1] if source else None


def _family(
    source: str,
    explicit_family: str | None,
    *,
    require_identity: bool,
    role: str,
) -> tuple[str, bool]:
    normalized_explicit = (
        explicit_family.strip().casefold() if explicit_family is not None else None
    )
    if normalized_explicit == "":
        raise ValueError("document_family must be non-empty")
    match = _DOCUMENT_ID.match(source)
    if match:
        values = match.groupdict()
        inferred = f"{values['kind'].casefold()}:{int(values['number'])}"
        if normalized_explicit is not None and normalized_explicit != inferred:
            raise ValueError(
                f"Recognized source {source} has fixed family {inferred}; "
                "document_family cannot override it"
            )
        return inferred, True
    if normalized_explicit is not None:
        return normalized_explicit, True
    if require_identity:
        raise ValueError(
            f"{role} source {source} has no recognized document identity; "
            "add an explicit document_family field"
        )
    return f"unparsed:{source.casefold()}", False


def _summary(
    path: Path,
    cases: list[dict[str, Any]],
    exposure: str,
    *,
    role: str,
    require_family_identity: bool,
) -> dict[str, Any]:
    relevant = [case for case in cases if case["relevant"]]
    sources = sorted({_source(case) for case in relevant if _source(case)}, key=str.casefold)
    families = set()
    unparsed_sources = set()
    for case in relevant:
        source = _source(case)
        if not source:
            continue
        family, parsed = _family(
            source,
            case.get("document_family"),
            require_identity=require_family_identity,
            role=role,
        )
        families.add(family)
        if not parsed:
            unparsed_sources.add(source)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "exposure": exposure,
        "case_count": len(cases),
        "relevant_count": len(relevant),
        "negative_or_ambiguous_count": len(cases) - len(relevant),
        "languages": dict(sorted(Counter(case.get("language", "unknown") for case in cases).items())),
        "categories": dict(sorted(Counter(case.get("category", "unknown") for case in cases).items())),
        "source_count": len(sources),
        "family_count": len(families),
        "sources": sources,
        "families": sorted(families),
        "unparsed_sources": sorted(unparsed_sources, key=str.casefold),
    }


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "exact_sources": sorted(set(left["sources"]) & set(right["sources"]), key=str.casefold),
        "families": sorted(set(left["families"]) & set(right["families"])),
    }


def freeze_protocol(
    *,
    development: Path,
    validation: Path,
    holdout: Path | None,
    output: Path,
    frozen_at: str | None = None,
    corpus_manifest: Path | None = None,
    validation_review_receipt: Path | None = None,
    allow_legacy_development_holdout_overlap: bool = False,
    prospective_holdout_reason: str | None = None,
) -> dict[str, Any]:
    """Validate and hash a leakage-audited three-way evaluation protocol."""
    if holdout is None:
        if not prospective_holdout_reason or not prospective_holdout_reason.strip():
            raise ValueError("A missing holdout requires a prospective holdout reason")
    elif prospective_holdout_reason is not None:
        raise ValueError("Do not provide a prospective holdout reason with a holdout file")
    datasets = {
        "development": _load_dataset(development),
        "validation": _load_dataset(validation),
    }
    if holdout is not None:
        datasets["final_holdout"] = _load_dataset(holdout)
    all_ids: dict[str, str] = {}
    for role, cases in datasets.items():
        for case in cases:
            prior = all_ids.get(case["id"])
            if prior:
                raise ValueError(f"Evaluation case {case['id']} occurs in both {prior} and {role}")
            all_ids[case["id"]] = role

    paths = {
        "development": development,
        "validation": validation,
    }
    if holdout is not None:
        paths["final_holdout"] = holdout
    exposures = {
        "development": "inspected_development",
        "validation": "periodic_aggregate_and_failure_review",
        "final_holdout": "final_aggregate_only",
    }
    summaries = {
        role: _summary(
            paths[role],
            datasets[role],
            exposures[role],
            role=role,
            require_family_identity=role != "development",
        )
        for role in datasets
    }
    if holdout is None:
        summaries["final_holdout"] = {
            "status": "prospective_not_available",
            "reason": prospective_holdout_reason.strip(),
            "path": None,
            "sha256": None,
            "exposure": exposures["final_holdout"],
            "case_count": 0,
            "relevant_count": 0,
            "negative_or_ambiguous_count": 0,
            "languages": {},
            "categories": {},
            "source_count": 0,
            "family_count": 0,
            "sources": [],
            "families": [],
            "unparsed_sources": [],
        }
    leakage = {
        "development_validation": _overlap(summaries["development"], summaries["validation"]),
        "development_holdout": _overlap(summaries["development"], summaries["final_holdout"]),
        "validation_holdout": _overlap(summaries["validation"], summaries["final_holdout"]),
    }
    if leakage["validation_holdout"]["families"]:
        raise ValueError("Validation and final holdout must not share document families")
    if (
        leakage["development_holdout"]["families"]
        and not allow_legacy_development_holdout_overlap
    ):
        raise ValueError(
            "Development and final holdout must not share document families; "
            "use a prospective holdout or explicitly allow legacy overlap"
        )

    protocol = {
        "protocol_version": 1,
        "frozen_at": frozen_at or datetime.now(timezone.utc).isoformat(),
        "sets": summaries,
        "leakage_audit": leakage,
        "rules": {
            "development": "Individual failures may be inspected and used for tuning.",
            "validation": "Use periodically for architecture selection; record every access and avoid repeated case-specific tuning.",
            "final_holdout": "Run only for final generalization checks; expose aggregate metrics only and never tune from individual failures.",
            "legacy_exposure": "The original 697-case benchmark was previously inspected and is development data regardless of later partitioning.",
            "family_policy": "Final holdout is disjoint from development and validation by conservative type-number family across years and languages unless legacy development overlap is explicitly allowed and recorded.",
            "legacy_development_holdout_overlap_allowed": allow_legacy_development_holdout_overlap,
            "prospective_holdout": holdout is None,
        },
    }
    if corpus_manifest:
        protocol["corpus_manifest"] = {
            "path": str(corpus_manifest.resolve()),
            "sha256": sha256_file(corpus_manifest),
        }
    if validation_review_receipt:
        review_receipt = json.loads(
            validation_review_receipt.read_text(encoding="utf-8")
        )
        if review_receipt.get("candidate_sha256") != summaries["validation"]["sha256"]:
            raise ValueError("Validation review receipt and dataset hashes differ")
        protocol["validation_review_receipt"] = {
            "path": str(validation_review_receipt.resolve()),
            "sha256": sha256_file(validation_review_receipt),
            "status": review_receipt.get("status"),
            "independent_human_approval": review_receipt.get(
                "independent_human_approval"
            ),
        }
    write_json_atomic(output, protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    holdout = parser.add_mutually_exclusive_group(required=True)
    holdout.add_argument("--holdout", type=Path)
    holdout.add_argument("--prospective-holdout-reason")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--validation-review-receipt", type=Path)
    parser.add_argument("--allow-legacy-development-holdout-overlap", action="store_true")
    args = parser.parse_args()
    freeze_protocol(
        development=args.development,
        validation=args.validation,
        holdout=args.holdout,
        output=args.output,
        corpus_manifest=args.corpus_manifest,
        validation_review_receipt=args.validation_review_receipt,
        allow_legacy_development_holdout_overlap=args.allow_legacy_development_holdout_overlap,
        prospective_holdout_reason=args.prospective_holdout_reason,
    )


if __name__ == "__main__":
    main()
