"""Disposable structured-ingestion corpus build and controlled retrieval benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from bm25 import create_bm25, retrieve_bm25
from embedding import create_embedding_model
from ingestion.docling_pdf_loader import DoclingPdfLoader
from ingestion.models import Block, Page, StructuredDocument
from ingestion.structured_chunker import structure_aware_chunks
from reranker import create_reranker, rank_scored_documents, score_documents


COLLECTION_NAME = "bct_structured_ingestion_experiment"
DENSE_K = 20
BM25_K = 15
RERANKER_TOP_K = 5
BASELINE_SHA256 = "DC9917CF985250FA7453C7EEF7D5C31443345F7030BE5E20EEFF924E6D8C36C3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _document_from_dict(value: dict[str, Any]) -> StructuredDocument:
    pages = []
    for raw_page in value["pages"]:
        blocks = [Block(**raw_block) for raw_block in raw_page.get("blocks", [])]
        pages.append(Page(**{**raw_page, "blocks": blocks}))
    return StructuredDocument(**{**value, "pages": pages})


def ingest_corpus(documents_dir: Path, output_dir: Path, limit: int | None = None) -> dict[str, Any]:
    pdfs = sorted(documents_dir.rglob("*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]
    structured_dir = output_dir / "structured_documents"
    structured_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "documents_dir": str(documents_dir.resolve()), "records": {}, "errors": []
    }
    loader: DoclingPdfLoader | None = None

    started = time.perf_counter()
    for index, pdf_path in enumerate(pdfs, start=1):
        relative_path = pdf_path.relative_to(documents_dir).as_posix()
        file_hash = _sha256(pdf_path)
        artifact_path = structured_dir / f"{file_hash.lower()}.json"
        prior = manifest["records"].get(relative_path)
        if prior and prior.get("sha256") == file_hash and artifact_path.exists():
            print(f"[{index}/{len(pdfs)}] cached {relative_path}", flush=True)
            continue
        if loader is None:
            loader = DoclingPdfLoader()
        print(f"[{index}/{len(pdfs)}] extracting {relative_path}", flush=True)
        document_started = time.perf_counter()
        try:
            document = loader.load(pdf_path)
            _write_json(artifact_path, document.to_dict())
            ocr_pages = [page.page_number for page in document.pages if page.extraction_method == "ocr"]
            manifest["records"][relative_path] = {
                "source": pdf_path.name,
                "sha256": file_hash,
                "artifact": str(artifact_path.resolve()),
                "pages": len(document.pages),
                "blocks": sum(len(page.blocks) for page in document.pages),
                "ocr_pages": ocr_pages,
                "seconds": time.perf_counter() - document_started,
            }
            manifest["errors"] = [error for error in manifest["errors"] if error["path"] != relative_path]
        except Exception as error:
            manifest["errors"] = [entry for entry in manifest["errors"] if entry["path"] != relative_path]
            manifest["errors"].append({
                "path": relative_path,
                "type": type(error).__name__,
                "message": str(error),
            })
            print(f"ERROR {relative_path}: {type(error).__name__}: {error}", flush=True)
        _write_json(manifest_path, manifest)

    manifest["elapsed_seconds_this_run"] = time.perf_counter() - started
    manifest["document_count"] = len(manifest["records"])
    manifest["error_count"] = len(manifest["errors"])
    _write_json(manifest_path, manifest)
    return manifest


def _iter_structured_documents(manifest: dict[str, Any]) -> Iterable[StructuredDocument]:
    for record in manifest["records"].values():
        value = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        yield _document_from_dict(value)


def build_index(output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((output_dir / "ingestion_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("errors"):
        raise RuntimeError(f"Cannot build complete index with {len(manifest['errors'])} extraction errors")
    index_dir = output_dir / "chroma_db"
    index_manifest_path = output_dir / "index_manifest.json"
    if index_dir.exists() or index_manifest_path.exists():
        raise FileExistsError("Experimental index already exists; choose a new output directory")

    chunks: list[Document] = []
    extraction_counts: Counter[str] = Counter()
    table_chunks = 0
    for document in _iter_structured_documents(manifest):
        document_chunks = structure_aware_chunks(document)
        chunks.extend(document_chunks)
        for chunk in document_chunks:
            extraction_counts.update(chunk.metadata["extraction_methods"].split(","))
            table_chunks += "table" in chunk.metadata["block_types"].split(",")

    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps({
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }, ensure_ascii=False) + "\n")

    chroma_chunks = [Document(page_content=chunk.page_content, metadata={
        key: (",".join(str(item) for item in value) if isinstance(value, list) else value)
        for key, value in chunk.metadata.items()
    }) for chunk in chunks]
    started = time.perf_counter()
    embedding = create_embedding_model()
    vector_store = Chroma.from_documents(
        documents=chroma_chunks,
        embedding=embedding,
        persist_directory=str(index_dir),
        collection_name=COLLECTION_NAME,
    )
    result = {
        "collection": COLLECTION_NAME,
        "index_dir": str(index_dir.resolve()),
        "chunks_path": str(chunks_path.resolve()),
        "chunk_count": len(chunks),
        "chroma_count": vector_store._collection.count(),
        "table_chunks": table_chunks,
        "extraction_method_chunk_counts": dict(extraction_counts),
        "embedding_model": "intfloat/multilingual-e5-small",
        "chunk_max_chars": 1000,
        "chunk_overlap": 200,
        "seconds": time.perf_counter() - started,
    }
    _write_json(index_manifest_path, result)
    return result


def _load_all_chroma_documents(vector_store: Chroma) -> list[Document]:
    raw = vector_store.get(include=["documents", "metadatas"])
    return [Document(page_content=text, metadata=metadata) for text, metadata in zip(raw["documents"], raw["metadatas"])]


def _combine_documents(dense: list[Document], sparse: list[Document]) -> list[Document]:
    combined: list[Document] = []
    seen: set[tuple[str, Any, Any]] = set()
    for document in dense + sparse:
        key = (document.page_content, document.metadata.get("source"), document.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            combined.append(document)
    return combined


def _page_matches(metadata: dict[str, Any], expected_page: int) -> bool:
    start = int(metadata.get("page", -1))
    end = int(metadata.get("page_end", start))
    return start <= expected_page <= end


def _ranks(results: list[tuple[Document, float]], expected_source: str, expected_page: int) -> tuple[int | None, int | None]:
    source_rank = exact_rank = None
    for index, (document, _score) in enumerate(results, start=1):
        if document.metadata.get("source", "").casefold() != expected_source.casefold():
            continue
        source_rank = source_rank or index
        if exact_rank is None and _page_matches(document.metadata, expected_page):
            exact_rank = index
    return source_rank, exact_rank


def _result_item(document: Document, score: float) -> dict[str, Any]:
    return {
        "source": document.metadata.get("source"),
        "page": document.metadata.get("page"),
        "page_end": document.metadata.get("page_end", document.metadata.get("page")),
        "score": float(score),
        "heading_path": document.metadata.get("heading_path", ""),
        "block_types": document.metadata.get("block_types", ""),
        "extraction_methods": document.metadata.get("extraction_methods", ""),
        "text": document.page_content,
    }


def _normalized_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return {token for token in re.findall(r"[\w\u0600-\u06ff]+", normalized) if len(token) >= 2 and not token.startswith("l00")}


def _coverage(expected: str, actual: str) -> float:
    expected_tokens = _normalized_tokens(expected)
    if not expected_tokens:
        return 0.0
    return len(expected_tokens & _normalized_tokens(actual)) / len(expected_tokens)


def _document_identity(filename: str) -> tuple[str, str, str, str] | None:
    match = re.search(r"(?i)(cir|note)[_ ]?(\d{4})[_ ]?(\d+).*?_(fr|ar)\.pdf$", filename)
    return tuple(part.casefold() for part in match.groups()) if match else None


def _wrong_version(expected_source: str, candidates: list[Document]) -> bool:
    expected = _document_identity(expected_source)
    if not expected:
        return False
    expected_type, expected_year, expected_number, expected_language = expected
    for candidate in candidates[:5]:
        identity = _document_identity(str(candidate.metadata.get("source", "")))
        if not identity:
            continue
        candidate_type, candidate_year, candidate_number, candidate_language = identity
        if (candidate_type, candidate_number, candidate_language) == (expected_type, expected_number, expected_language) and candidate_year != expected_year:
            return True
    return False


def _metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    relevant = [record for record in records if record["relevant"]]
    def hit(rank_name: str, cutoff: int) -> float:
        return sum(record[field][rank_name] is not None and record[field][rank_name] <= cutoff for record in relevant) / len(relevant)
    result = {
        "n": len(relevant),
        "source_top1": hit("source_rank", 1),
        "source_top5": hit("source_rank", 5),
        "source_top20": hit("source_rank", 20),
        "exact_page_top1": hit("exact_page_rank", 1),
        "exact_page_top5": hit("exact_page_rank", 5),
        "exact_page_top20": hit("exact_page_rank", 20),
        "mrr_source": sum(1 / record[field]["source_rank"] if record[field]["source_rank"] else 0 for record in relevant) / len(relevant),
    }
    for dimension in ("language", "category"):
        grouped: dict[str, Any] = {}
        for value in sorted({record[dimension] for record in relevant}):
            subset = [record for record in relevant if record[dimension] == value]
            grouped[value] = {
                "n": len(subset),
                "source_top5": sum(record[field]["source_rank"] is not None and record[field]["source_rank"] <= 5 for record in subset) / len(subset),
                "exact_page_top5": sum(record[field]["exact_page_rank"] is not None and record[field]["exact_page_rank"] <= 5 for record in subset) / len(subset),
            }
        result[f"by_{dimension}"] = grouped
    return result


def run_benchmark(
    evaluation_path: Path,
    baseline_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    baseline_hash = _sha256(baseline_path)
    if baseline_hash != BASELINE_SHA256:
        raise ValueError(f"Baseline artifact hash changed: {baseline_hash}")
    evaluations = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if limit is not None:
        evaluations = evaluations[:limit]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_by_id = {record["id"]: record for record in baseline["records"]}
    manifest = json.loads((output_dir / "ingestion_manifest.json").read_text(encoding="utf-8"))
    structured_by_source = {document.filename: document for document in _iter_structured_documents(manifest)}
    all_chunks = []
    chunks_by_source: dict[str, list[Document]] = defaultdict(list)
    with (output_dir / "chunks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            document = Document(**raw)
            all_chunks.append(document)
            chunks_by_source[document.metadata["source"]].append(document)

    embedding = create_embedding_model()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(output_dir / "chroma_db"),
    )
    bm25_documents = _load_all_chroma_documents(vector_store)
    bm25 = create_bm25(bm25_documents)
    reranker = create_reranker()
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    checkpoint_path = output_dir / "benchmark_results.json"

    for index, case in enumerate(evaluations, start=1):
        started = time.perf_counter()
        query = case["query"]
        dense = vector_store.similarity_search(query, k=DENSE_K)
        sparse = retrieve_bm25(query, bm25, bm25_documents, k=BM25_K)
        candidates = _combine_documents(dense, sparse)
        ranked = rank_scored_documents(score_documents(reranker, query, candidates))
        latency = time.perf_counter() - started
        latencies.append(latency)
        baseline_record = baseline_by_id[case["id"]]
        baseline_result = baseline_record["union"]

        source_rank = exact_rank = None
        if case["relevant"]:
            source_rank, exact_rank = _ranks(ranked, case["expected_source"], int(case["expected_page"]))
        result = {
            "source_rank": source_rank,
            "exact_page_rank": exact_rank,
            "top5": [_result_item(document, score) for document, score in ranked[:5]],
        }
        failure_categories: list[str] = []
        evidence = {"page_coverage": None, "best_chunk_coverage": None, "native_page_coverage": None}
        ocr_rescue = False

        if case["relevant"] and (exact_rank is None or exact_rank > RERANKER_TOP_K):
            expected_source = case["expected_source"]
            expected_page = int(case["expected_page"])
            structured = structured_by_source.get(expected_source)
            page = next((item for item in structured.pages if item.page_number == expected_page), None) if structured else None
            quote = case.get("evidence_quote", "")
            page_coverage = _coverage(quote, page.raw_text) if page else 0.0
            native_coverage = _coverage(quote, page.metadata.get("native_raw_text", "")) if page else 0.0
            page_chunks = [chunk for chunk in chunks_by_source.get(expected_source, []) if _page_matches(chunk.metadata, expected_page)]
            best_chunk_coverage = max((_coverage(quote, chunk.page_content) for chunk in page_chunks), default=0.0)
            evidence = {
                "page_coverage": page_coverage,
                "best_chunk_coverage": best_chunk_coverage,
                "native_page_coverage": native_coverage,
            }
            source_candidates = [candidate for candidate in candidates if candidate.metadata.get("source", "").casefold() == expected_source.casefold()]
            page_candidates = [candidate for candidate in source_candidates if _page_matches(candidate.metadata, expected_page)]
            if page is None or page_coverage < 0.35:
                failure_categories.append("evidence_missing_because_of_extraction")
            if not source_candidates:
                failure_categories.append("correct_document_missing_from_candidate_set")
            elif not page_candidates:
                failure_categories.append("correct_page_missing")
            elif exact_rank is None or exact_rank > RERANKER_TOP_K:
                failure_categories.append("correct_evidence_retrieved_but_ranked_too_low")
            if _wrong_version(expected_source, candidates):
                failure_categories.append("wrong_temporal_or_document_version")
            if page_coverage >= 0.55 and best_chunk_coverage < 0.45:
                failure_categories.append("chunk_boundary_or_context_problem")

        if case["relevant"]:
            structured = structured_by_source.get(case["expected_source"])
            page = next((item for item in structured.pages if item.page_number == int(case["expected_page"])), None) if structured else None
            if page and page.extraction_method == "ocr":
                native_coverage = _coverage(case.get("evidence_quote", ""), page.metadata.get("native_raw_text", ""))
                final_coverage = _coverage(case.get("evidence_quote", ""), page.raw_text)
                baseline_rank = baseline_result.get("exact_page_rank")
                ocr_rescue = native_coverage < 0.35 and final_coverage >= 0.55 and (baseline_rank is None or baseline_rank > 5) and exact_rank is not None and exact_rank <= 5

        baseline_rank = baseline_result.get("exact_page_rank")
        repaired = case["relevant"] and (baseline_rank is None or baseline_rank > 5) and exact_rank is not None and exact_rank <= 5
        regressed = case["relevant"] and baseline_rank is not None and baseline_rank <= 5 and (exact_rank is None or exact_rank > 5)
        top = result["top5"][0] if result["top5"] else {}
        if ocr_rescue:
            reason = "Selective OCR recovered evidence missing from native extraction."
        elif repaired and top.get("block_types") == "table":
            reason = "The table was preserved as retrievable Markdown."
        elif repaired and top.get("heading_path"):
            reason = f"Legal hierarchy was retained with the chunk: {top['heading_path']}."
        elif repaired:
            reason = "Structure-aware grouping kept related evidence together."
        elif regressed:
            reason = "The expected page moved below the top-five cutoff after structured chunking."
        else:
            reason = "No top-five exact-page status change."

        records.append({
            "id": case["id"], "question_number": index, "query": query,
            "language": case.get("language"), "category": case.get("category"),
            "evidence_method": case.get("evidence_method"), "relevant": case["relevant"],
            "expected_source": case.get("expected_source"), "expected_page": case.get("expected_page"),
            "baseline": {"source_rank": baseline_result.get("source_rank"), "exact_page_rank": baseline_rank, "top5": baseline_result.get("top5", [])},
            "structured_ingestion": result,
            "candidate_counts": {"dense": len(dense), "bm25": len(sparse), "union": len(candidates)},
            "latency_seconds": latency,
            "repaired": repaired, "regressed": regressed, "reason": reason,
            "failure_categories": failure_categories,
            "primary_failure_category": failure_categories[0] if failure_categories else None,
            "evidence_diagnostics": evidence,
            "ocr_rescue": ocr_rescue,
        })
        print(f"[{index}/{len(evaluations)}] {case['id']} baseline={baseline_rank} structured={exact_rank}", flush=True)
        if index % 10 == 0:
            _write_json(checkpoint_path, {"status": "running", "completed": index, "records": records})

    artifact = {
        "status": "complete",
        "controls": {
            "evaluation_sha256": _sha256(evaluation_path),
            "baseline_artifact": str(baseline_path.resolve()),
            "baseline_sha256": baseline_hash,
            "embedding_model": "intfloat/multilingual-e5-small",
            "dense_k": DENSE_K,
            "bm25_k": BM25_K,
            "bm25_tokenization": "lowercase_whitespace_split",
            "fusion": "dense_then_bm25_candidate_union_exact_deduplication",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "reranker_top_k": RERANKER_TOP_K,
            "questions_unchanged": True,
            "query_expansion": False,
            "fine_tuning": False,
            "graphrag": False,
            "answer_llm_or_prompt_changed": False,
        },
        "summary": {
            "baseline": _metrics(records, "baseline"),
            "structured_ingestion": _metrics(records, "structured_ingestion"),
            "repairs": [record["id"] for record in records if record["repaired"]],
            "regressions": [record["id"] for record in records if record["regressed"]],
            "failure_category_counts": dict(Counter(category for record in records for category in record["failure_categories"])),
            "primary_failure_category_counts": dict(Counter(record["primary_failure_category"] for record in records if record["primary_failure_category"])),
            "ocr_rescues": [record["id"] for record in records if record["ocr_rescue"]],
            "latency_seconds": {"mean": statistics.mean(latencies), "median": statistics.median(latencies)},
        },
        "records": records,
    }
    _write_json(checkpoint_path, artifact)
    _write_report(output_dir / "benchmark_report.md", artifact)
    return artifact


def _write_report(path: Path, artifact: dict[str, Any]) -> None:
    summary = artifact["summary"]
    baseline = summary["baseline"]
    experiment = summary["structured_ingestion"]
    lines = [
        "# Structured ingestion retrieval benchmark", "",
        "This is a retrieval benchmark only; it does not measure answer correctness.", "",
        "| Metric | Baseline hybrid | Structured ingestion | Delta |", "|---|---:|---:|---:|",
    ]
    for key in ("source_top1", "source_top5", "source_top20", "exact_page_top1", "exact_page_top5", "exact_page_top20", "mrr_source"):
        lines.append(f"| {key} | {baseline[key]:.2%} | {experiment[key]:.2%} | {experiment[key] - baseline[key]:+.2%} |")
    lines.extend(["", f"Repairs: {len(summary['repairs'])}", "", f"Regressions: {len(summary['regressions'])}", "", "## Failure classification", ""])
    for category, count in sorted(summary["failure_category_counts"].items()):
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## OCR rescues", ""])
    lines.extend([f"- {case_id}" for case_id in summary["ocr_rescues"]] or ["- None"])
    lines.extend(["", "## Question-by-question changes", ""])
    for record in artifact["records"]:
        if not (record["repaired"] or record["regressed"] or record["failure_categories"]):
            continue
        lines.extend([
            f"### Question {record['question_number']} — `{record['id']}`", "",
            f"Baseline exact-page rank: {record['baseline']['exact_page_rank']}", "",
            f"Structured exact-page rank: {record['structured_ingestion']['exact_page_rank']}", "",
            f"Reason: {record['reason']}", "",
            f"Failure categories: {', '.join(record['failure_categories']) or 'none'}", "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("ingest", "index", "benchmark", "all"))
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation_queries.json"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command in {"ingest", "all"}:
        ingest_corpus(args.documents_dir, args.output_dir, args.limit)
    if args.command in {"index", "all"}:
        build_index(args.output_dir)
    if args.command in {"benchmark", "all"}:
        if args.baseline is None:
            parser.error("--baseline is required for benchmark/all")
        run_benchmark(args.evaluation, args.baseline, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
