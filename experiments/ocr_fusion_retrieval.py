"""Corpus-wide cached retrieval ablation for additive Arabic OCR evidence."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from bm25 import create_bm25
from embedding import create_embedding_model
from experiments.arabic_quality_experiment import _proposed_gate
from experiments.artifacts import sha256_file, write_json_atomic
from experiments.ocr_fallback_experiment import _converter, _extract_single_page
from experiments.retrieval_ablations import (
    SearchRepresentation,
    _candidate_item,
    _classification,
    _load_search_representation,
    _mcnemar,
    _merge_candidates,
    _metrics,
    _pages_by_source,
    _retrieve,
    _ranks,
    _serialize_chunks,
    _split_text,
)
from reranker import create_reranker, score_documents
from docling.datamodel.pipeline_options import RapidOcrOptions


EVALUATION_SHA256 = "00964BA335B759D01BA42CED75FC6AE10F082AE4D45499426CEA69F3F1DF3CA1"
CURRENT_RESULT_SHA256 = "A785228676583BE0961F3163E56B3B2D9D33C8B0E9D5F8D5EAB0866A31F9562F"
CURRENT_CANDIDATES_SHA256 = "A061D88BEE3F4C221EFC8612AC13272C83AA1C8A2EEA475409C2DC6E0CF65B33"
STRUCTURED_MANIFEST_SHA256 = "09ABE18E9DCAD9043650EC1B204967642741CE6C2F1EE93F042B83375B59D5BA"
OCR_DENSE_K = 5
OCR_BM25_K = 5
_TOKEN = re.compile(r"\w+", re.UNICODE)
_ARABIC_TOKEN = re.compile(r"^[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+$")


def page_risk_diagnostics(text: str, quality_score: float) -> dict[str, Any]:
    """Compute exactly the three features used by the frozen Arabic gate."""
    visible = [character for character in text if not character.isspace()]
    arabic_tokens = [token for token in _TOKEN.findall(text) if _ARABIC_TOKEN.match(token)]
    diagnostics = {
        "quality_score": float(quality_score),
        "latin_character_ratio": sum(
            "a" <= character.casefold() <= "z" or "\u00c0" <= character <= "\u024f"
            for character in visible
        )
        / max(len(visible), 1),
        "single_arabic_token_ratio": (
            sum(len(token) == 1 for token in arabic_tokens) / len(arabic_tokens)
            if arabic_tokens
            else 0.0
        ),
    }
    diagnostics["requires_fallback"] = _proposed_gate(diagnostics)
    return diagnostics


def is_arabic_query(query: str) -> bool:
    return any(
        "\u0600" <= character <= "\u06ff"
        or "\u0750" <= character <= "\u077f"
        or "\u08a0" <= character <= "\u08ff"
        for character in query
    )


def secondary_ocr_chunks(
    *,
    source: str,
    page_number: int,
    text: str,
    max_chars: int = 1000,
    overlap: int = 200,
) -> list[Document]:
    metadata = {
        "source": source,
        "page": page_number,
        "page_end": page_number,
        "pages": [page_number],
        "heading_path": "",
        "block_types": "flat_text",
        "extraction_methods": "ocr",
        "language": "ar",
        "representation": "arabic_ocr_secondary",
    }
    return [
        Document(
            page_content=piece,
            metadata={
                **metadata,
                "flat_part": part,
                "full_hierarchy_text": piece,
                "article_body_text": piece,
                "body_text": piece,
            },
        )
        for part, piece in enumerate(_split_text(text, max_chars=max_chars, overlap=overlap))
    ]


def _structured_pages(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    documents_dir = Path(manifest["documents_dir"])
    for relative, record in manifest["records"].items():
        document = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        if document["language"] != "ar":
            continue
        for page in document["pages"]:
            native_text = page.get("metadata", {}).get("native_raw_text", page.get("raw_text", ""))
            quality_score = page.get("metadata", {}).get(
                "native_quality_score", page.get("quality_score", 0.0)
            )
            diagnostics = page_risk_diagnostics(native_text, float(quality_score))
            if diagnostics["requires_fallback"]:
                selected.append(
                    {
                        "source": record["source"],
                        "relative_path": relative,
                        "pdf_path": documents_dir / Path(relative),
                        "pdf_sha256": record["sha256"],
                        "page_number": int(page["page_number"]),
                        "diagnostics": diagnostics,
                    }
                )
    return selected


def build_ocr_representation(
    *,
    structured_manifest_path: Path,
    output_dir: Path,
    ocr_cache_dir: Path,
) -> dict[str, Any]:
    representation_dir = output_dir / "representation"
    representation_manifest = representation_dir / "manifest.json"
    if representation_manifest.exists():
        return json.loads(representation_manifest.read_text(encoding="utf-8"))
    if representation_dir.exists():
        raise ValueError(
            f"Incomplete OCR representation exists without a manifest: {representation_dir}"
        )
    if sha256_file(structured_manifest_path) != STRUCTURED_MANIFEST_SHA256:
        raise ValueError("Structured ingestion manifest differs from the frozen control")
    manifest = json.loads(structured_manifest_path.read_text(encoding="utf-8"))
    selected = _structured_pages(manifest)
    ocr_cache_dir.mkdir(parents=True, exist_ok=True)
    converter = _converter(
        RapidOcrOptions(
            lang=["arabic"],
            backend="onnxruntime",
            force_full_page_ocr=True,
        )
    )
    chunks: list[Document] = []
    page_records = []
    for index, page in enumerate(selected, start=1):
        pdf_path = page["pdf_path"]
        if sha256_file(pdf_path) != page["pdf_sha256"]:
            raise ValueError(f"Source PDF hash drifted: {pdf_path}")
        cache_path = ocr_cache_dir / (
            f"{page['pdf_sha256'].casefold()}-p{page['page_number']}-arabic_rapidocr.json"
        )
        payload = _extract_single_page(
            pdf_path,
            page["page_number"],
            converter,
            "arabic_rapidocr",
            cache_path,
        )
        page_chunks = secondary_ocr_chunks(
            source=page["source"],
            page_number=page["page_number"],
            text=payload["text"],
        )
        chunks.extend(page_chunks)
        page_records.append(
            {
                "source": page["source"],
                "page": page["page_number"],
                "source_pdf_sha256": page["pdf_sha256"],
                "ocr_cache_sha256": sha256_file(cache_path),
                "ocr_block_count": payload["block_count"],
                "ocr_character_count": len(payload["text"]),
                "ocr_seconds": payload["elapsed_seconds"],
                "chunk_count": len(page_chunks),
                "trigger_diagnostics": page["diagnostics"],
            }
        )
        print(f"[ocr {index}/{len(selected)}] {page['source']} page {page['page_number']}", flush=True)

    representation_dir.mkdir(parents=True, exist_ok=False)
    chunks_path = representation_dir / "chunks.jsonl"
    _serialize_chunks(chunks_path, chunks)
    chroma_documents = [
        Document(
            page_content=chunk.page_content,
            metadata={
                key: ",".join(str(value) for value in value)
                if isinstance(value, list)
                else value
                for key, value in chunk.metadata.items()
            },
        )
        for chunk in chunks
    ]
    started = time.perf_counter()
    vector_store = Chroma.from_documents(
        documents=chroma_documents,
        embedding=create_embedding_model(),
        collection_name="bct_arabic_ocr_secondary_v1",
        persist_directory=str(representation_dir / "chroma_db"),
    )
    result = {
        "name": "arabic_ocr_secondary",
        "collection": "bct_arabic_ocr_secondary_v1",
        "chunks_path": str(chunks_path.resolve()),
        "index_dir": str((representation_dir / "chroma_db").resolve()),
        "structured_manifest_sha256": sha256_file(structured_manifest_path),
        "triggered_page_count": len(selected),
        "chunk_count": len(chunks),
        "chroma_count": vector_store._collection.count(),
        "embedding_model": "intfloat/multilingual-e5-small",
        "build_seconds": time.perf_counter() - started,
        "ocr_page_records": page_records,
    }
    write_json_atomic(representation_manifest, result)
    return result


def _deserialize_candidates(raw: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "document": Document(**item["document"]),
            "representations": set(item["representations"]),
            "ranks": item["ranks"],
        }
        for item in raw["candidates"]
    ]


def _language_metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return {
        language: _metrics(
            [record for record in records if record.get("language") == language], field
        )
        for language in ("fr", "ar")
    }


def run_retrieval(
    *,
    evaluation_path: Path,
    current_result_path: Path,
    current_candidates_path: Path,
    structured_cache_dir: Path,
    representation_manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    for path, expected, label in (
        (evaluation_path, EVALUATION_SHA256, "evaluation"),
        (current_result_path, CURRENT_RESULT_SHA256, "current winner result"),
        (current_candidates_path, CURRENT_CANDIDATES_SHA256, "current candidate cache"),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen {label} hash differs: {actual}")
    cases = json.loads(evaluation_path.read_text(encoding="utf-8"))
    current = json.loads(current_result_path.read_text(encoding="utf-8"))
    current_by_id = {record["id"]: record for record in current["records"]}
    base_candidates = json.loads(current_candidates_path.read_text(encoding="utf-8"))
    if set(current_by_id) != {case["id"] for case in cases} or set(base_candidates) != set(current_by_id):
        raise ValueError("Current result and candidate cache must exactly cover evaluation IDs")

    output_path = output_dir / "results" / "additive_arabic_ocr_5_5.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
        records = existing.get("records", [])
    else:
        records = []
    completed_ids = {record["id"] for record in records}
    representation: SearchRepresentation = _load_search_representation(
        representation_manifest
    )
    reranker = create_reranker()
    pages_by_source = _pages_by_source(structured_cache_dir)
    for chunk in representation.documents:
        source = str(chunk.metadata["source"])
        page = int(chunk.metadata["page"])
        current_text = pages_by_source.setdefault(source, {}).get(page, "")
        if chunk.page_content not in current_text:
            pages_by_source[source][page] = f"{current_text}\n{chunk.page_content}".strip()

    for number, case in enumerate(cases, start=1):
        if case["id"] in completed_ids:
            continue
        current_record = current_by_id[case["id"]]
        baseline_result = current_record["result"]
        if not is_arabic_query(case["query"]):
            records.append(
                {
                    **current_record,
                    "baseline": baseline_result,
                    "result": baseline_result,
                    "candidate_count": len(base_candidates[case["id"]]["candidates"]),
                    "base_candidate_count": len(base_candidates[case["id"]]["candidates"]),
                    "ocr_candidate_count": 0,
                    "repaired": False,
                    "regressed": False,
                    "latency_seconds": current_record["latency_seconds"],
                    "observed_explanation": "French query bypassed the Arabic OCR channel.",
                }
            )
            continue

        started = time.perf_counter()
        base = _deserialize_candidates(base_candidates[case["id"]])
        ocr = _retrieve(representation, case["query"], OCR_DENSE_K, OCR_BM25_K)
        candidates = _merge_candidates((base, ocr))
        ocr_retrieval_seconds = time.perf_counter() - started
        reranker_started = time.perf_counter()
        scored = score_documents(
            reranker,
            case["query"],
            [candidate["document"] for candidate in candidates],
        )
        ranked = sorted(
            zip(candidates, (float(score) for _document, score in scored)),
            key=lambda item: item[1],
            reverse=True,
        )
        latency = (
            float(base_candidates[case["id"]].get("retrieval_seconds", 0.0))
            + ocr_retrieval_seconds
            + time.perf_counter()
            - reranker_started
        )
        source_rank = exact_page_rank = None
        if case["relevant"]:
            source_rank, exact_page_rank = _ranks(
                ranked, case["expected_source"], int(case["expected_page"])
            )
        baseline_rank = baseline_result.get("exact_page_rank")
        repaired = bool(
            case["relevant"]
            and (baseline_rank is None or baseline_rank > 5)
            and exact_page_rank is not None
            and exact_page_rank <= 5
        )
        regressed = bool(
            case["relevant"]
            and baseline_rank is not None
            and baseline_rank <= 5
            and (exact_page_rank is None or exact_page_rank > 5)
        )
        expected_page = case.get("expected_page")
        page_text = (
            pages_by_source.get(str(case.get("expected_source")), {}).get(int(expected_page))
            if expected_page is not None
            else None
        )
        categories, diagnostics = _classification(
            case, candidates, page_text, exact_page_rank
        )
        records.append(
            {
                "id": case["id"],
                "question_number": number,
                "query": case["query"],
                "language": case.get("language"),
                "category": case.get("category"),
                "evidence_method": case.get("evidence_method"),
                "relevant": case["relevant"],
                "expected_source": case.get("expected_source"),
                "expected_page": case.get("expected_page"),
                "baseline": baseline_result,
                "result": {
                    "source_rank": source_rank,
                    "exact_page_rank": exact_page_rank,
                    "top5": [
                        _candidate_item(candidate, score, candidate["document"].page_content)
                        for candidate, score in ranked[:5]
                    ],
                },
                "candidate_count": len(candidates),
                "base_candidate_count": len(base),
                "ocr_candidate_count": len(ocr),
                "latency_seconds": latency,
                "repaired": repaired,
                "regressed": regressed,
                "failure_categories": categories,
                "primary_failure_category": categories[0] if categories else None,
                "evidence_diagnostics": diagnostics,
                "observed_explanation": (
                    f"Expected page moved from current rank {baseline_rank} to {exact_page_rank}."
                    if repaired or regressed
                    else "No Top-5 exact-page status change."
                ),
            }
        )
        print(
            f"[retrieval {number}/{len(cases)}] {case['id']} current={baseline_rank} fusion={exact_page_rank}",
            flush=True,
        )
        if number % 10 == 0:
            write_json_atomic(
                output_path,
                {"status": "running", "completed": number, "records": records},
            )

    latency = [float(record["latency_seconds"]) for record in records]
    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "name": "additive_arabic_ocr_5_5",
            "base": "frozen structured_baseline_chunking candidate union",
            "arabic_query_routing": "Unicode Arabic-script presence in runtime query text",
            "ocr_representation": "explicit Arabic RapidOCR, page-local sequential 1000/200 chunks",
            "ocr_dense_k": OCR_DENSE_K,
            "ocr_bm25_k": OCR_BM25_K,
            "candidate_merge": "exact union; native candidates retained",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "reranker_top_k": 5,
            "triggered_corpus_pages": representation_manifest["triggered_page_count"],
            "ocr_chunk_count": representation_manifest["chunk_count"],
        },
        "inputs": {
            "evaluation_sha256": sha256_file(evaluation_path),
            "current_result_sha256": sha256_file(current_result_path),
            "current_candidates_sha256": sha256_file(current_candidates_path),
            "structured_manifest_sha256": representation_manifest[
                "structured_manifest_sha256"
            ],
            "ocr_representation_manifest_sha256": sha256_file(
                output_dir / "representation" / "manifest.json"
            ),
        },
        "summary": {
            "current": _metrics(records, "baseline"),
            "fusion": _metrics(records, "result"),
            "current_by_language": _language_metrics(records, "baseline"),
            "fusion_by_language": _language_metrics(records, "result"),
            "repairs": [record["id"] for record in records if record["repaired"]],
            "regressions": [record["id"] for record in records if record["regressed"]],
            "paired_page5_vs_current": _mcnemar(records),
            "failure_category_counts": dict(
                Counter(
                    category
                    for record in records
                    for category in record.get("failure_categories", [])
                )
            ),
            "latency_seconds": {
                "mean": statistics.mean(latency),
                "median": statistics.median(latency),
            },
        },
        "limitations": [
            "The original 697-case benchmark is development data; this does not establish generalization.",
            "The Arabic gate and additive representation were selected after development-failure inspection.",
            "French queries bypass the OCR channel and preserve the frozen current result.",
            "This is retrieval-only and does not measure answer, citation, grounding, or abstention quality.",
        ],
        "records": records,
    }
    write_json_atomic(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "run", "all"))
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--current-result", type=Path, required=True)
    parser.add_argument("--current-candidates", type=Path, required=True)
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--structured-cache-dir", type=Path, required=True)
    parser.add_argument("--ocr-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representation_manifest = build_ocr_representation(
        structured_manifest_path=args.structured_manifest,
        output_dir=args.output_dir,
        ocr_cache_dir=args.ocr_cache_dir,
    )
    if args.command in {"run", "all"}:
        run_retrieval(
            evaluation_path=args.evaluation,
            current_result_path=args.current_result,
            current_candidates_path=args.current_candidates,
            structured_cache_dir=args.structured_cache_dir,
            representation_manifest=representation_manifest,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
