"""Build a targeted full-page context suite from frozen StructuredDocuments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


EXPERIMENT_ID = "structured-full-page-context-development-v1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().upper()


def render_structured_page(page: dict[str, Any]) -> str:
    parts = []
    for block in page.get("blocks", []):
        text = str(block.get("text", "")).strip()
        if text:
            parts.append(f"[{str(block.get('type', 'block')).upper()}]\n{text}")
    if not parts:
        raise ValueError(f"Structured page {page.get('page_number')} has no text blocks")
    return "\n\n".join(parts)


def evidence_alphanumeric_identifiers(text: str) -> list[str]:
    """Return stable letter-and-digit tokens from a verified evidence excerpt."""
    return sorted(
        {
            token
            for token in re.findall(r"[A-Za-z0-9]+", text)
            if any(character.isalpha() for character in token)
            and any(character.isdigit() for character in token)
        },
        key=str.casefold,
    )


def build_context_expansion_suite(
    *,
    base_suite: dict[str, Any],
    manifest: dict[str, Any],
    selected_ids: set[str],
    base_suite_sha256: str,
    manifest_sha256: str,
    include_verified_excerpt: bool = False,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    base_by_id = {case["id"]: case for case in base_suite["cases"]}
    if len(base_by_id) != len(base_suite["cases"]):
        raise ValueError("Base answer suite contains duplicate case IDs")
    missing = selected_ids - set(base_by_id)
    if missing:
        raise ValueError(f"Selected IDs are absent from the base suite: {sorted(missing)}")

    records_by_source: dict[str, list[dict[str, Any]]] = {}
    for record in manifest["records"].values():
        records_by_source.setdefault(Path(record["source"]).name.casefold(), []).append(record)

    cases = []
    document_hashes = {}
    for case in base_suite["cases"]:
        if case["id"] not in selected_ids:
            continue
        if not case.get("relevant"):
            raise ValueError(f"Context expansion requires a relevant case: {case['id']}")
        matches = records_by_source.get(Path(case["expected_source"]).name.casefold(), [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one StructuredDocument for {case['expected_source']}; "
                f"found {len(matches)}"
            )
        record = matches[0]
        artifact_path = Path(record["artifact"])
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
        pages = [
            page
            for page in document["pages"]
            if int(page["page_number"]) == int(case["expected_page"])
        ]
        if len(pages) != 1:
            raise ValueError(
                f"Expected exactly one page {case['expected_page']} in {case['expected_source']}"
            )
        expanded = deepcopy(case)
        original_quote = str(expanded["evidence_quote"])
        page_context = render_structured_page(pages[0])
        if include_verified_excerpt:
            expanded["evidence_quote"] = (
                f"[VERIFIED EXCERPT]\n{original_quote}\n\n"
                f"[FULL LABELED PAGE]\n{page_context}"
            )
            expanded["evidence_method"] = (
                "verified_excerpt_plus_structured_full_labeled_page"
            )
        else:
            expanded["evidence_quote"] = page_context
            expanded["evidence_method"] = "structured_full_labeled_page"
        expanded["context_expansion"] = {
            "selection": "all non-empty blocks from the exact labeled StructuredDocument page, in source order",
            "baseline_evidence_quote_sha256": _sha256_text(original_quote),
            "structured_document_pdf_sha256": str(record["sha256"]).upper(),
            "structured_document_artifact_sha256": sha256_file(artifact_path),
            "block_count": len(pages[0].get("blocks", [])),
            "verified_excerpt_included": include_verified_excerpt,
        }
        if prompt_version == "bct-claim-linked-answer-v5":
            expanded["required_answer_literals"] = evidence_alphanumeric_identifiers(
                original_quote
            )
        document_hashes[Path(case["expected_source"]).name] = {
            "pdf_sha256": str(record["sha256"]).upper(),
            "artifact_sha256": sha256_file(artifact_path),
        }
        cases.append(expanded)

    all_relevant_ids = {
        case["id"] for case in base_suite["cases"] if case.get("relevant")
    }
    full_relevant_suite = selected_ids == all_relevant_ids
    return {
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "base_answer_suite_sha256": base_suite_sha256,
            "structured_manifest_sha256": manifest_sha256,
            "structured_documents": document_hashes,
        },
        "suite_type": (
            "full_relevant_gold_evidence_context_development"
            if full_relevant_suite
            else "targeted_gold_evidence_context_sufficiency_development"
        ),
        "selection": (
            "All relevant cases from the frozen answer-safety suite."
            if full_relevant_suite
            else "The selected claim-linked relevant failures whose frozen snippets omitted "
            "a query condition or governing table heading."
        ),
        "evaluation_protocol": (
            "Keep question, source, page, expected answer, model, and prompt fixed; expand evidence "
            "with all ordered blocks from the exact labeled page."
        ),
        "answer_experiment": {
            "experiment_id": (
                "claim-linked-full-page-relevant-development-v1"
                if full_relevant_suite
                else EXPERIMENT_ID
            ),
            "candidate_and_evidence": (
                "verified excerpt plus full exact-page StructuredDocument context"
                if include_verified_excerpt
                else "targeted full exact-page StructuredDocument context"
            ),
            **({"prompt_version": prompt_version} if prompt_version else {}),
        },
        "counts": {
            "total": len(cases),
            "relevant": len(cases),
            "by_language": {
                language: sum(case["language"] == language for case in cases)
                for language in ("ar", "fr")
            },
        },
        "limitations": [
            "Development-only gold-evidence context suite.",
            "Exact labeled pages isolate context sufficiency and do not measure whether retrieval supplies that context.",
            "Required v5 identifiers, when present, are derived from the verified evidence excerpt rather than the expected answer.",
            "A passing result cannot by itself justify production page expansion.",
        ],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-suite", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id", action="append")
    selection.add_argument("--all-relevant", action="store_true")
    parser.add_argument("--include-verified-excerpt", action="store_true")
    parser.add_argument("--prompt-version")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_suite = json.loads(args.base_suite.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_ids = (
        {case["id"] for case in base_suite["cases"] if case.get("relevant")}
        if args.all_relevant
        else set(args.case_id)
    )
    artifact = build_context_expansion_suite(
        base_suite=base_suite,
        manifest=manifest,
        selected_ids=selected_ids,
        base_suite_sha256=sha256_file(args.base_suite),
        manifest_sha256=sha256_file(args.manifest),
        include_verified_excerpt=args.include_verified_excerpt,
        prompt_version=args.prompt_version,
    )
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
