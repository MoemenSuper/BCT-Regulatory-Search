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
    ids = [case.get("id") for case in value]
    if any(not case_id for case_id in ids):
        raise ValueError(f"Every evaluation case needs a non-empty id: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Evaluation case IDs must be unique: {path}")
    for case in value:
        missing = [field for field in ("query", "language", "relevant") if field not in case]
        if missing:
            raise ValueError(f"Case {case['id']} is missing required fields: {', '.join(missing)}")
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
    match = _DOCUMENT_ID.match(source)
    if match:
        values = match.groupdict()
        inferred = f"{values['kind'].casefold()}:{int(values['number'])}"
        if explicit_family and explicit_family.strip().casefold() != inferred:
            raise ValueError(
                f"Recognized source {source} has fixed family {inferred}; "
                "document_family cannot override it"
            )
        return inferred, True
    if explicit_family:
        return explicit_family.strip().casefold(), True
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
    holdout: Path,
    output: Path,
    frozen_at: str | None = None,
    corpus_manifest: Path | None = None,
    allow_legacy_development_holdout_overlap: bool = False,
) -> dict[str, Any]:
    """Validate and hash a leakage-audited three-way evaluation protocol."""
    datasets = {
        "development": _load_dataset(development),
        "validation": _load_dataset(validation),
        "final_holdout": _load_dataset(holdout),
    }
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
        "final_holdout": holdout,
    }
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
        },
    }
    if corpus_manifest:
        protocol["corpus_manifest"] = {
            "path": str(corpus_manifest.resolve()),
            "sha256": sha256_file(corpus_manifest),
        }
    write_json_atomic(output, protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--allow-legacy-development-holdout-overlap", action="store_true")
    args = parser.parse_args()
    freeze_protocol(
        development=args.development,
        validation=args.validation,
        holdout=args.holdout,
        output=args.output,
        corpus_manifest=args.corpus_manifest,
        allow_legacy_development_holdout_overlap=args.allow_legacy_development_holdout_overlap,
    )


if __name__ == "__main__":
    main()
