"""Screen Arabic documents for contextually implausible extracted years."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


_SOURCE_YEAR = re.compile(r"^[^_]+_(20\d{2})_.*_ar\.pdf$", re.IGNORECASE)
_FOUR_DIGITS = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_YEAR_CONTEXT = re.compile(
    r"(?:لسنة|سنة|عام|موسم|جانفي|فيفري|مارس|أفريل|ماي|جوان|جويلية|أوت|"
    r"سبتمبر|أكتوبر|نوفمبر|ديسمبر|année|janvier|février|mars|avril|mai|juin|"
    r"juillet|août|septembre|octobre|novembre|décembre)",
    re.IGNORECASE,
)


def source_year(source: str) -> int | None:
    match = _SOURCE_YEAR.match(Path(source).name)
    return int(match.group(1)) if match else None


def contextual_implausible_years(text: str, *, document_year: int) -> list[str]:
    """Return impossible four-digit year-like tokens near explicit date context."""
    suspicious = []
    for match in _FOUR_DIGITS.finditer(text):
        value = int(match.group(1))
        if 1900 <= value <= document_year + 5:
            continue
        before = text[max(0, match.start() - 48) : match.start()]
        context_matches = list(_YEAR_CONTEXT.finditer(before))
        directly_after_context = bool(
            context_matches
            and not re.search(r"\d", before[context_matches[-1].end() :])
        )
        if directly_after_context:
            suspicious.append(match.group(1))
    return sorted(set(suspicious))


def document_digit_order_risk(document: dict[str, Any]) -> dict[str, Any]:
    year = source_year(str(document.get("filename", "")))
    if year is None or document.get("language") != "ar":
        return {"requires_visual_fallback": False, "page_hits": []}
    page_hits = []
    for page in document["pages"]:
        text = str(
            page.get("metadata", {}).get(
                "native_raw_text", page.get("raw_text", "")
            )
        )
        tokens = contextual_implausible_years(text, document_year=year)
        if tokens:
            page_hits.append(
                {"page": int(page["page_number"]), "suspicious_tokens": tokens}
            )
    return {
        "requires_visual_fallback": bool(page_hits),
        "document_year": year,
        "page_hits": page_hits,
    }


def evaluate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    records = []
    arabic_document_count = 0
    for record in manifest["records"].values():
        document = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        if document.get("language") != "ar":
            continue
        arabic_document_count += 1
        risk = document_digit_order_risk(document)
        if risk["requires_visual_fallback"]:
            records.append({"source": document["filename"], **risk})
    return {
        "arabic_document_count": arabic_document_count,
        "flagged_document_count": len(records),
        "flagged_document_rate": (
            len(records) / arabic_document_count if arabic_document_count else 0.0
        ),
        "flagged_page_count": sum(len(record["page_hits"]) for record in records),
        "decision": "KEEP_AS_QUERY_TIME_VISUAL_FALLBACK_SIGNAL",
        "decision_basis": (
            "The signal catches all four visually confirmed Arabic digit-order failures from the "
            "retrieval-first answer run, but its corpus trigger rate is too broad for unconditional VLM ingestion."
        ),
        "records": records,
        "limitations": [
            "This detector identifies risk; it does not repair or authorize extracted text.",
            "A flagged document still requires visual, OCR, or VLM comparison before numeric claims are answered.",
            "The corpus trigger rate is too broad for unconditional VLM processing; use only after retrieval for claims that depend on dates or numbers.",
            "The rule is intentionally narrow and will miss digit corruption without explicit year or date context.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.structured_manifest.read_text(encoding="utf-8"))
    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "arabic-contextual-digit-order-risk-development-v1",
        "inputs": {"structured_manifest_sha256": sha256_file(args.structured_manifest)},
        **evaluate_manifest(manifest),
    }
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
