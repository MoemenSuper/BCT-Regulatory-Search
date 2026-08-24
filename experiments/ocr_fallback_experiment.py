"""Cached development comparison of native, auto-OCR, and Arabic OCR text."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrAutoOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from experiments.arabic_quality_experiment import _proposed_gate
from experiments.artifacts import sha256_file, write_json_atomic
from ingestion.docling_pdf_loader import _extract_blocks


_TOKEN = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*", re.UNICODE)


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(_normalized_text(text)))


def _numbers(text: str) -> set[str]:
    translated = "".join(
        str(unicodedata.digit(character)) if character.isdigit() else character
        for character in _normalized_text(text)
    )
    return set(_NUMBER.findall(translated))


def _recall(expected: set[str], actual: set[str]) -> float | None:
    return len(expected & actual) / len(expected) if expected else None


def _text_diagnostics(expected: str, actual: str) -> dict[str, Any]:
    visible = [character for character in actual if not character.isspace()]
    total_visible = max(len(visible), 1)
    return {
        "evidence_token_coverage": _recall(_tokens(expected), _tokens(actual)) or 0.0,
        "critical_number_recall": _recall(_numbers(expected), _numbers(actual)),
        "expected_critical_numbers": sorted(_numbers(expected)),
        "extracted_critical_numbers": sorted(_numbers(expected) & _numbers(actual)),
        "character_count": len(actual),
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
    }


def select_triggered_arabic_cases(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the already frozen development gate without changing its thresholds."""
    return [
        case
        for case in suite["cases"]
        if case.get("language") == "ar" and _proposed_gate(case["diagnostics"])
    ]


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [
        row["critical_number_recall"]
        for row in rows
        if row["critical_number_recall"] is not None
    ]
    return {
        "case_count": len(rows),
        "numeric_case_count": len(numeric),
        "mean_evidence_token_coverage": _mean(
            [row["evidence_token_coverage"] for row in rows]
        ),
        "median_evidence_token_coverage": (
            statistics.median(row["evidence_token_coverage"] for row in rows)
            if rows
            else 0.0
        ),
        "mean_critical_number_recall": _mean(numeric),
        "empty_page_count": sum(row["character_count"] == 0 for row in rows),
        "mean_elapsed_seconds": _mean([row["elapsed_seconds"] for row in rows]),
    }


def evaluate_variants(
    cases: list[dict[str, Any]],
    variants: dict[tuple[str, int], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Score cached extraction variants against verified development evidence snippets."""
    rows = []
    for case in cases:
        key = (case["expected_source"], int(case["expected_page"]))
        if key not in variants:
            raise ValueError(f"Missing extraction variants for {key[0]} page {key[1]}")
        for method, payload in variants[key].items():
            rows.append(
                {
                    "id": case["id"],
                    "role": case["role"],
                    "source": key[0],
                    "page": key[1],
                    "method": method,
                    "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
                    **_text_diagnostics(case["evidence_quote"], payload.get("text", "")),
                }
            )
    methods = sorted({row["method"] for row in rows})
    aggregates = {
        role: {
            method: _aggregate(
                [row for row in rows if row["role"] == role and row["method"] == method]
            )
            for method in methods
        }
        for role in ("failure", "control")
    }

    def coverage(role: str, method: str) -> float:
        return aggregates[role][method]["mean_evidence_token_coverage"]

    def numeric(role: str, method: str) -> float:
        return aggregates[role][method]["mean_critical_number_recall"]

    required = {"native", "auto_ocr", "arabic_rapidocr"}
    if not required.issubset(methods):
        raise ValueError(f"Extraction variants must include {sorted(required)}")
    comparisons = {
        "arabic_vs_native_failure_coverage_delta": coverage(
            "failure", "arabic_rapidocr"
        )
        - coverage("failure", "native"),
        "arabic_vs_auto_failure_coverage_delta": coverage("failure", "arabic_rapidocr")
        - coverage("failure", "auto_ocr"),
        "arabic_vs_native_failure_numeric_delta": numeric("failure", "arabic_rapidocr")
        - numeric("failure", "native"),
        "arabic_vs_native_control_coverage_delta": coverage("control", "arabic_rapidocr")
        - coverage("control", "native"),
    }
    keep = (
        comparisons["arabic_vs_native_failure_coverage_delta"] >= 0.15
        and comparisons["arabic_vs_auto_failure_coverage_delta"] >= 0.10
        and comparisons["arabic_vs_native_failure_numeric_delta"] >= 0.10
        and comparisons["arabic_vs_native_control_coverage_delta"] >= -0.05
    )
    return {
        "decision_gate": {
            "minimum_arabic_vs_native_failure_coverage_delta": 0.15,
            "minimum_arabic_vs_auto_failure_coverage_delta": 0.10,
            "minimum_arabic_vs_native_failure_numeric_delta": 0.10,
            "minimum_arabic_vs_native_control_coverage_delta": -0.05,
        },
        "aggregates": aggregates,
        "comparisons": comparisons,
        "decision": "KEEP_FOR_CACHED_RETRIEVAL_ABLATION" if keep else "REJECT",
        "case_scores": rows,
        "limitations": [
            "Development stress cases only; thresholds and trigger were selected on development data.",
            "Evidence snippets verify key answer-bearing text, not complete page transcription or hallucination rate.",
            "Only gate-triggered pages are measured, so the control sample may be small.",
            "Passing this screen permits only a cached retrieval ablation, not ingestion deployment.",
        ],
    }


def _converter(ocr_options: OcrAutoOptions | RapidOcrOptions) -> DocumentConverter:
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.ocr_options = ocr_options
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _extract_single_page(
    pdf_path: Path,
    page_number: int,
    converter: DocumentConverter,
    method: str,
    cache_path: Path,
) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix="bct-ocr-comparison-") as temp_dir:
        with fitz.open(pdf_path) as source:
            single_page = fitz.open()
            single_page.insert_pdf(
                source, from_page=page_number - 1, to_page=page_number - 1
            )
            single_page_path = Path(temp_dir) / "page.pdf"
            single_page.save(single_page_path)
            single_page.close()
        started = time.perf_counter()
        result = converter.convert(single_page_path)
        elapsed = time.perf_counter() - started
        blocks = _extract_blocks(result.document, method).get(1, [])
    payload = {
        "method": method,
        "text": "\n".join(block.text for block in blocks),
        "block_count": len(blocks),
        "elapsed_seconds": elapsed,
    }
    write_json_atomic(cache_path, payload)
    return payload


def run_experiment(
    *,
    stress_suite_path: Path,
    evaluation_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    suite = json.loads(stress_suite_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation_by_id = {case["id"]: case for case in evaluation}
    selected = select_triggered_arabic_cases(suite)
    cases = []
    for case in selected:
        if case["id"] not in evaluation_by_id:
            raise ValueError(f"Evaluation data is missing selected case {case['id']}")
        cases.append({**case, "evidence_quote": evaluation_by_id[case["id"]]["evidence_quote"]})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records_by_source = {
        record["source"].casefold(): (relative, record)
        for relative, record in manifest["records"].items()
    }
    documents_dir = Path(manifest["documents_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    converters = {
        "auto_ocr": _converter(OcrAutoOptions(force_full_page_ocr=True)),
        "arabic_rapidocr": _converter(
            RapidOcrOptions(
                lang=["arabic"],
                backend="onnxruntime",
                force_full_page_ocr=True,
            )
        ),
    }

    variants: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for source, page_number in sorted(
        {(case["expected_source"], int(case["expected_page"])) for case in cases},
        key=lambda item: (item[0].casefold(), item[1]),
    ):
        relative, record = records_by_source[source.casefold()]
        pdf_path = documents_dir / Path(relative)
        if sha256_file(pdf_path) != record["sha256"]:
            raise ValueError(f"Source PDF hash drifted: {pdf_path}")
        document = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        page = next(
            (item for item in document["pages"] if int(item["page_number"]) == page_number),
            None,
        )
        if page is None:
            raise ValueError(f"Structured artifact lacks {source} page {page_number}")
        variants[(source, page_number)] = {
            "native": {
                "text": page.get("metadata", {}).get("native_raw_text", page["raw_text"]),
                "elapsed_seconds": 0.0,
            }
        }
        for method, converter in converters.items():
            cache_path = cache_dir / f"{record['sha256'].casefold()}-p{page_number}-{method}.json"
            variants[(source, page_number)][method] = _extract_single_page(
                pdf_path, page_number, converter, method, cache_path
            )

    evaluation_result = evaluate_variants(cases, variants)
    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_scope": "development_gate_triggered_arabic_pages",
        "inputs": {
            "stress_suite_sha256": sha256_file(stress_suite_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "structured_manifest_sha256": sha256_file(manifest_path),
        },
        "configuration": {
            "native": "cached Docling native_raw_text",
            "auto_ocr": "OcrAutoOptions(force_full_page_ocr=True)",
            "arabic_rapidocr": "RapidOcrOptions(lang=['arabic'], backend='onnxruntime', force_full_page_ocr=True)",
            "docling_version": version("docling"),
            "rapidocr_version": version("rapidocr"),
            "selected_case_count": len(cases),
            "selected_unique_page_count": len(variants),
        },
        **evaluation_result,
    }
    write_json_atomic(output_dir / "ocr_fallback_result.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-suite", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(
        stress_suite_path=args.stress_suite,
        evaluation_path=args.evaluation,
        manifest_path=args.structured_manifest,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
