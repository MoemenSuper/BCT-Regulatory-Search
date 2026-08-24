"""Controlled retrieval ablations using cached StructuredDocument artifacts only.

This module intentionally owns its indexes beneath a caller-provided output
directory.  It never calls the production ingestion or vector-store writers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from bm25 import create_bm25, retrieve_bm25
from embedding import create_embedding_model
from ingestion.models import Block, Page, StructuredDocument
from ingestion.structured_chunker import structure_aware_chunks
from reranker import create_reranker, score_documents


DENSE_K = 20
BM25_K = 15
RERANKER_TOP_K = 5
RRF_K = 60
RRF_CANDIDATE_BUDGET = 30
EVALUATION_COUNT = 697
EVALUATION_SHA256 = "00964BA335B759D01BA42CED75FC6AE10F082AE4D45499426CEA69F3F1DF3CA1"
BASELINE_SHA256 = "DC9917CF985250FA7453C7EEF7D5C31443345F7030BE5E20EEFF924E6D8C36C3"
FAILURE_ORDER = (
    "evidence_missing_because_of_extraction",
    "wrong_temporal_or_document_version",
    "chunk_boundary_or_context_problem",
    "correct_document_missing_from_candidate_set",
    "correct_page_missing",
    "correct_evidence_retrieved_but_ranked_too_low",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _document_from_dict(value: dict[str, Any]) -> StructuredDocument:
    pages = []
    for page in value["pages"]:
        pages.append(Page(**{**page, "blocks": [Block(**block) for block in page.get("blocks", [])]}))
    return StructuredDocument(**{**value, "pages": pages})


def _iter_structured_documents(cache_dir: Path) -> Iterable[StructuredDocument]:
    manifest = json.loads((cache_dir / "ingestion_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("errors"):
        raise RuntimeError(f"Cached extraction has {len(manifest['errors'])} errors")
    for record in manifest["records"].values():
        yield _document_from_dict(json.loads(Path(record["artifact"]).read_text(encoding="utf-8")))


def _validate_reused_structured_cache(cache_dir: Path, previous_dir: Path) -> None:
    """Refuse a full-hierarchy control built from a different extracted corpus."""
    def hashes(directory: Path) -> dict[str, tuple[str, str]]:
        manifest = json.loads((directory / "ingestion_manifest.json").read_text(encoding="utf-8"))
        return {
            path: (str(record["sha256"]), _sha256(Path(record["artifact"])))
            for path, record in manifest["records"].items()
        }
    if hashes(cache_dir) != hashes(previous_dir):
        raise ValueError("Previous structured index does not use identical PDF and StructuredDocument cache artifacts")


def _split_text(text: str, max_chars: int = 1000, overlap: int = 200) -> list[str]:
    """The previous baseline's recursive-style, text-only chunk shape."""
    if not text.strip():
        return []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind(" ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            pieces.append(part)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return pieces


def _heading_text(path: str) -> tuple[str, str, str]:
    headings = [heading.strip() for heading in path.split(" > ") if heading.strip()]
    prefix = "\n".join(dict.fromkeys(headings))
    article = next((heading for heading in reversed(headings) if re.match(r"(?i)^article\b", heading)), "")
    return prefix, article, "\n".join(headings)


def _alternate_texts(content: str, heading_path: str) -> dict[str, str]:
    prefix, article, _all_headings = _heading_text(heading_path)
    body = content
    if prefix and body.startswith(prefix):
        body = body[len(prefix):].lstrip("\n")
    article_body = f"{article}\n{body}".strip() if article else body
    return {"full_hierarchy_text": content, "article_body_text": article_body, "body_text": body}


def _metadata_for_page(document: StructuredDocument, page: Page, *, representation: str) -> dict[str, Any]:
    paths = [block.heading_path for block in page.blocks if block.heading_path]
    heading_path = " > ".join(paths[-1]) if paths else ""
    return {
        "source": document.filename,
        "page": page.page_number,
        "page_end": page.page_number,
        "pages": [page.page_number],
        "heading_path": heading_path,
        "block_types": "flat_text",
        "extraction_methods": page.extraction_method,
        "language": document.language,
        "representation": representation,
    }


def structured_baseline_chunks(documents: Iterable[StructuredDocument]) -> list[Document]:
    """Flatten cached page text into the old 1000/200 text-only representation."""
    chunks: list[Document] = []
    for document in documents:
        for page in document.pages:
            metadata = _metadata_for_page(document, page, representation="structured_baseline_chunking")
            for part, text in enumerate(_split_text(page.raw_text)):
                item_metadata = {**metadata, "flat_part": part}
                item_metadata.update({"full_hierarchy_text": text, "article_body_text": text, "body_text": text})
                chunks.append(Document(page_content=text, metadata=item_metadata))
    return chunks


def structured_hierarchy_chunks(documents: Iterable[StructuredDocument], *, metadata_only: bool) -> list[Document]:
    representation = "structured_metadata_hierarchy" if metadata_only else "structured_full_hierarchy_text"
    chunks: list[Document] = []
    for document in documents:
        for chunk in structure_aware_chunks(document):
            metadata = {**chunk.metadata, "representation": representation}
            alternatives = _alternate_texts(chunk.page_content, str(metadata.get("heading_path", "")))
            metadata.update(alternatives)
            text = alternatives["article_body_text"] if metadata_only else alternatives["full_hierarchy_text"]
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


def _serialize_chunks(path: Path, chunks: list[Document]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            output.write(json.dumps({"page_content": chunk.page_content, "metadata": chunk.metadata}, ensure_ascii=False) + "\n")


def _load_chunks(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as source:
        return [Document(**json.loads(line)) for line in source if line.strip()]


def build_representation(cache_dir: Path, output_dir: Path, name: str) -> dict[str, Any]:
    """Create an isolated index once; extraction artifacts are strictly read-only."""
    representation_dir = output_dir / "representations" / name
    manifest_path = representation_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    representation_dir.mkdir(parents=True, exist_ok=False)
    documents = list(_iter_structured_documents(cache_dir))
    if name == "structured_baseline_chunking":
        chunks = structured_baseline_chunks(documents)
    elif name == "structured_metadata_hierarchy":
        chunks = structured_hierarchy_chunks(documents, metadata_only=True)
    elif name == "structured_full_hierarchy_text":
        chunks = structured_hierarchy_chunks(documents, metadata_only=False)
    else:
        raise ValueError(f"Unknown representation: {name}")
    chunks_path = representation_dir / "chunks.jsonl"
    _serialize_chunks(chunks_path, chunks)
    chroma_documents = [Document(page_content=chunk.page_content, metadata={
        key: ",".join(str(value) for value in value) if isinstance(value, list) else value
        for key, value in chunk.metadata.items()
    }) for chunk in chunks]
    started = time.perf_counter()
    collection = f"bct_ablation_{name}"
    vector_store = Chroma.from_documents(
        documents=chroma_documents,
        embedding=create_embedding_model(),
        collection_name=collection,
        persist_directory=str(representation_dir / "chroma_db"),
    )
    manifest = {
        "name": name,
        "collection": collection,
        "chunks_path": str(chunks_path.resolve()),
        "index_dir": str((representation_dir / "chroma_db").resolve()),
        "chunk_count": len(chunks),
        "chroma_count": vector_store._collection.count(),
        "embedding_model": "intfloat/multilingual-e5-small",
        "build_seconds": time.perf_counter() - started,
    }
    _write_json(manifest_path, manifest)
    return manifest


@dataclass
class SearchRepresentation:
    name: str
    vector_store: Chroma
    documents: list[Document]
    bm25: Any


def _load_search_representation(manifest: dict[str, Any]) -> SearchRepresentation:
    embedding = create_embedding_model()
    vector_store = Chroma(
        collection_name=manifest["collection"],
        embedding_function=embedding,
        persist_directory=manifest["index_dir"],
    )
    documents = _load_chunks(Path(manifest["chunks_path"]))
    return SearchRepresentation(manifest["name"], vector_store, documents, create_bm25(documents))


def _load_existing_representation(name: str, chunks_path: Path | None, index_dir: Path, collection: str) -> SearchRepresentation:
    embedding = create_embedding_model()
    vector_store = Chroma(collection_name=collection, embedding_function=embedding, persist_directory=str(index_dir))
    if chunks_path and chunks_path.exists():
        documents = _load_chunks(chunks_path)
    else:
        raw = vector_store.get(include=["documents", "metadatas"])
        documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(raw["documents"], raw["metadatas"])]
    return SearchRepresentation(name, vector_store, documents, create_bm25(documents))


def _candidate_key(document: Document) -> str:
    # Match the production candidate-union key.  Chroma serializes list metadata
    # differently from JSONL, so `pages` must not participate in equality.
    return "\x1f".join((document.page_content, str(document.metadata.get("source", "")), str(document.metadata.get("page", ""))))


def _normalized_document(document: Document, *, use_page_label: bool = False) -> Document:
    """Use the evaluation corpus's filename source identity across all indexes."""
    metadata = dict(document.metadata)
    source = str(metadata.get("source", ""))
    if source:
        metadata["source"] = Path(source).name
    if use_page_label and metadata.get("page_label") is not None:
        metadata["page"] = int(metadata["page_label"])
        metadata["page_end"] = int(metadata["page_label"])
    return Document(page_content=document.page_content, metadata=metadata)


def _retrieve(rep: SearchRepresentation, query: str, dense_k: int = DENSE_K, bm25_k: int = BM25_K) -> list[dict[str, Any]]:
    use_page_label = rep.name == "baseline"
    dense = [_normalized_document(document, use_page_label=use_page_label) for document in rep.vector_store.similarity_search(query, k=dense_k)]
    sparse = [_normalized_document(document, use_page_label=use_page_label) for document in retrieve_bm25(query, rep.bm25, rep.documents, k=bm25_k)]
    records: dict[str, dict[str, Any]] = {}
    for channel, results in (("dense", dense), ("bm25", sparse)):
        for rank, document in enumerate(results, start=1):
            key = _candidate_key(document)
            candidate = records.setdefault(key, {"document": document, "representations": set(), "ranks": {}})
            candidate["representations"].add(rep.name)
            candidate["ranks"].setdefault(rep.name, {})[channel] = rank
    return list(records.values())


def _merge_candidates(groups: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for candidate in group:
            key = _candidate_key(candidate["document"])
            if key not in merged:
                merged[key] = {"document": candidate["document"], "representations": set(), "ranks": {}}
            target = merged[key]
            target["representations"].update(candidate["representations"])
            for representation, ranks in candidate["ranks"].items():
                target["ranks"].setdefault(representation, {}).update(ranks)
    return list(merged.values())


def rrf_select(candidates: list[dict[str, Any]], budget: int = RRF_CANDIDATE_BUDGET) -> list[dict[str, Any]]:
    """Select a bounded pre-rerank pool using ranks from dense and BM25 only."""
    def score(candidate: dict[str, Any]) -> float:
        return sum(1 / (RRF_K + rank) for ranks in candidate["ranks"].values() for rank in ranks.values())
    return sorted(candidates, key=lambda candidate: (-score(candidate), _candidate_key(candidate["document"])))[:budget]


def _normalized_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return {token for token in re.findall(r"[\w\u0600-\u06ff]+", normalized) if len(token) >= 2}


def redundancy_deduplicate(candidates: list[dict[str, Any]], threshold: float = 0.92) -> tuple[list[dict[str, Any]], int]:
    """Conservative lexical dedup: only suppress near-identical chunks from one source."""
    kept: list[dict[str, Any]] = []
    token_sets: list[set[str]] = []
    for candidate in candidates:
        source = str(candidate["document"].metadata.get("source", "")).casefold()
        tokens = _normalized_tokens(candidate["document"].page_content)
        duplicate = False
        for kept_candidate, kept_tokens in zip(kept, token_sets):
            if str(kept_candidate["document"].metadata.get("source", "")).casefold() != source:
                continue
            overlap = len(tokens & kept_tokens) / max(1, len(tokens | kept_tokens))
            if overlap >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
            token_sets.append(tokens)
    return kept, len(candidates) - len(kept)


def _page_matches(metadata: dict[str, Any], expected_page: int) -> bool:
    pages = metadata.get("pages")
    if isinstance(pages, list):
        return expected_page in {int(page) for page in pages}
    if isinstance(pages, str) and pages:
        return expected_page in {int(page) for page in pages.split(",")}
    start = int(metadata.get("page", -1))
    end = int(metadata.get("page_end", start))
    return start <= expected_page <= end


def _ranks(ranked: list[tuple[dict[str, Any], float]], source: str, page: int) -> tuple[int | None, int | None]:
    source_rank = page_rank = None
    for rank, (candidate, _score) in enumerate(ranked, start=1):
        metadata = candidate["document"].metadata
        if str(metadata.get("source", "")).casefold() != source.casefold():
            continue
        source_rank = source_rank or rank
        if page_rank is None and _page_matches(metadata, page):
            page_rank = rank
    return source_rank, page_rank


def _reranker_text(candidate: dict[str, Any], style: str) -> str:
    metadata = candidate["document"].metadata
    body = str(metadata.get("body_text", candidate["document"].page_content))
    if style == "full_hierarchy":
        return str(metadata.get("full_hierarchy_text", candidate["document"].page_content))
    if style == "article_body":
        return str(metadata.get("article_body_text", candidate["document"].page_content))
    if style == "body_only":
        return body
    if style == "concise_metadata_body":
        heading = str(metadata.get("article_body_text", "")).split("\n", 1)[0]
        prefix = " | ".join(part for part in (heading if re.match(r"(?i)^article\b", heading) else "", str(metadata.get("source", ""))) if part)
        return f"{prefix}\n{body}".strip()
    raise ValueError(f"Unknown reranker style: {style}")


def _coverage(expected: str, actual: str) -> float:
    expected_tokens = _normalized_tokens(expected)
    return len(expected_tokens & _normalized_tokens(actual)) / len(expected_tokens) if expected_tokens else 0.0


def _document_identity(filename: str) -> tuple[str, str, str, str] | None:
    match = re.search(r"(?i)(cir|note)[_ ]?(\d{4})[_ ]?(\d+).*?_(fr|ar)\.pdf$", filename)
    return tuple(item.casefold() for item in match.groups()) if match else None


def _wrong_version(expected_source: str, candidates: list[dict[str, Any]]) -> bool:
    expected = _document_identity(expected_source)
    if not expected:
        return False
    for candidate in candidates:
        identity = _document_identity(str(candidate["document"].metadata.get("source", "")))
        if identity and identity[0] == expected[0] and identity[2] == expected[2] and identity[3] == expected[3] and identity[1] != expected[1]:
            return True
    return False


def _metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    relevant = [record for record in records if record["relevant"]]
    def hit(rank: str, cutoff: int) -> float:
        return sum(record[field][rank] is not None and record[field][rank] <= cutoff for record in relevant) / len(relevant)
    return {
        "n": len(relevant),
        "source_top1": hit("source_rank", 1), "source_top5": hit("source_rank", 5), "source_top20": hit("source_rank", 20),
        "exact_page_top1": hit("exact_page_rank", 1), "exact_page_top5": hit("exact_page_rank", 5), "exact_page_top20": hit("exact_page_rank", 20),
        "mrr_source": sum(1 / record[field]["source_rank"] if record[field]["source_rank"] else 0 for record in relevant) / len(relevant),
    }


def _mcnemar(records: list[dict[str, Any]]) -> dict[str, Any]:
    repaired = sum(record["repaired"] for record in records)
    regressed = sum(record["regressed"] for record in records)
    discordant = repaired + regressed
    if not discordant:
        p_value = 1.0
    else:
        smaller = min(repaired, regressed)
        p_value = min(1.0, 2 * sum(math.comb(discordant, value) for value in range(smaller + 1)) / 2**discordant)
    return {"test": "two_sided_exact_McNemar", "repaired": repaired, "regressed": regressed, "net": repaired - regressed, "p_value": p_value}


def _candidate_item(candidate: dict[str, Any], score: float | None = None, reranker_text: str | None = None) -> dict[str, Any]:
    document = candidate["document"]
    return {
        "source": document.metadata.get("source"), "page": document.metadata.get("page"), "page_end": document.metadata.get("page_end", document.metadata.get("page")),
        "representation": sorted(candidate["representations"]), "dense_bm25_ranks": candidate["ranks"], "reranker_score": float(score) if score is not None else None,
        "heading_path": document.metadata.get("heading_path", ""), "chunk_text": document.page_content, "reranker_text": reranker_text,
    }


def _load_evaluation(evaluation: Path, baseline: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _sha256(evaluation) != EVALUATION_SHA256:
        raise ValueError("Evaluation hash differs from the canonical 697-question set")
    cases = json.loads(evaluation.read_text(encoding="utf-8"))
    if len(cases) != EVALUATION_COUNT:
        raise ValueError(f"Expected {EVALUATION_COUNT} questions, got {len(cases)}")
    if _sha256(baseline) != BASELINE_SHA256:
        raise ValueError("Canonical baseline artifact hash differs")
    raw_baseline = json.loads(baseline.read_text(encoding="utf-8"))
    return cases, {record["id"]: record["union"] for record in raw_baseline["records"]}


def _classification(case: dict[str, Any], candidates: list[dict[str, Any]], page_text: str | None, result_rank: int | None) -> tuple[list[str], dict[str, float | None]]:
    if not case["relevant"] or (result_rank is not None and result_rank <= RERANKER_TOP_K):
        return [], {"page_coverage": None, "candidate_coverage": None}
    source_matches = [candidate for candidate in candidates if str(candidate["document"].metadata.get("source", "")).casefold() == str(case["expected_source"]).casefold()]
    page_matches = [candidate for candidate in source_matches if _page_matches(candidate["document"].metadata, int(case["expected_page"]))]
    quote = case.get("evidence_quote", "")
    page_coverage = _coverage(quote, page_text or "")
    candidate_coverage = max((_coverage(quote, candidate["document"].page_content) for candidate in page_matches), default=0.0)
    categories: list[str] = []
    if page_text is None or page_coverage < 0.35:
        categories.append("evidence_missing_because_of_extraction")
    if not source_matches:
        categories.append("correct_document_missing_from_candidate_set")
    elif not page_matches:
        categories.append("correct_page_missing")
    elif candidate_coverage >= 0.35:
        categories.append("correct_evidence_retrieved_but_ranked_too_low")
    if _wrong_version(str(case["expected_source"]), candidates[:5]):
        categories.append("wrong_temporal_or_document_version")
    if page_matches and (candidate_coverage < 0.35 or (page_coverage >= 0.55 and candidate_coverage < 0.45)):
        categories.append("chunk_boundary_or_context_problem")
    return categories, {"page_coverage": page_coverage, "candidate_coverage": candidate_coverage}


def run_configuration(
    *, name: str, cases: list[dict[str, Any]], baseline: dict[str, Any], representations: list[SearchRepresentation],
    output_dir: Path, pages_by_source: dict[str, dict[int, str]], reranker_style: str = "article_body",
    dense_k: int = DENSE_K, bm25_k: int = BM25_K, component_dense_k: int | None = None, component_bm25_k: int | None = None,
    select: str = "union", cache_candidates: bool = True, candidate_cache_id: str | None = None, redundancy_deduplication: bool = False,
) -> dict[str, Any]:
    result_path = output_dir / "results" / f"{name}.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
        records: list[dict[str, Any]] = existing.get("records", [])
    else:
        records = []
    reranker = create_reranker()
    latencies = [float(record["latency_seconds"]) for record in records]
    removed_counts = [int(record.get("dedup_removed", 0)) for record in records]
    candidate_cache_path = output_dir / "candidate_caches" / f"{candidate_cache_id or name}.json"
    cached = json.loads(candidate_cache_path.read_text(encoding="utf-8")) if candidate_cache_path.exists() else None
    candidate_cache: dict[str, Any] = {}
    completed_ids = {record["id"] for record in records}
    for number, case in enumerate(cases, start=1):
        if case["id"] in completed_ids:
            continue
        if cached:
            raw_candidates = []
            for item in cached[case["id"]]["candidates"]:
                raw_candidates.append({"document": Document(**item["document"]), "representations": set(item["representations"]), "ranks": item["ranks"]})
            retrieval_seconds = cached[case["id"]]["retrieval_seconds"]
        else:
            retrieval_started = time.perf_counter()
            groups = [_retrieve(rep, case["query"], component_dense_k or dense_k, component_bm25_k or bm25_k) for rep in representations]
            raw_candidates = _merge_candidates(groups)
            retrieval_seconds = time.perf_counter() - retrieval_started
            candidate_cache[case["id"]] = {"retrieval_seconds": retrieval_seconds, "candidates": [
                {"document": {"page_content": candidate["document"].page_content, "metadata": candidate["document"].metadata}, "representations": sorted(candidate["representations"]), "ranks": candidate["ranks"]}
                for candidate in raw_candidates
            ]}
        candidates = raw_candidates
        if select == "rrf":
            candidates = rrf_select(candidates)
        elif select == "union_budget":
            candidates = candidates[:RRF_CANDIDATE_BUDGET]
        if redundancy_deduplication:
            candidates, removed = redundancy_deduplicate(candidates)
        else:
            removed = 0
        reranker_started = time.perf_counter()
        inputs = [_reranker_text(candidate, reranker_style) for candidate in candidates]
        scored = score_documents(reranker, case["query"], [Document(page_content=text, metadata=candidate["document"].metadata) for candidate, text in zip(candidates, inputs)])
        ranked = sorted(zip(candidates, (float(score) for _document, score in scored), inputs), key=lambda item: item[1], reverse=True)
        latency = retrieval_seconds + time.perf_counter() - reranker_started
        latencies.append(latency)
        removed_counts.append(removed)
        source_rank = exact_page_rank = None
        if case["relevant"]:
            source_rank, exact_page_rank = _ranks([(candidate, score) for candidate, score, _text in ranked], case["expected_source"], int(case["expected_page"]))
        baseline_result = baseline[case["id"]]
        baseline_rank = baseline_result.get("exact_page_rank")
        repaired = bool(case["relevant"] and (baseline_rank is None or baseline_rank > 5) and exact_page_rank is not None and exact_page_rank <= 5)
        regressed = bool(case["relevant"] and baseline_rank is not None and baseline_rank <= 5 and (exact_page_rank is None or exact_page_rank > 5))
        expected_page = case.get("expected_page")
        page_text = pages_by_source.get(str(case.get("expected_source")), {}).get(int(expected_page)) if expected_page is not None else None
        categories, diagnostics = _classification(case, candidates, page_text, exact_page_rank)
        records.append({
            "id": case["id"], "question_number": number, "query": case["query"], "language": case.get("language"), "category": case.get("category"),
            "evidence_method": case.get("evidence_method"), "relevant": case["relevant"], "expected_source": case.get("expected_source"), "expected_page": case.get("expected_page"),
            "baseline": {"source_rank": baseline_result.get("source_rank"), "exact_page_rank": baseline_rank, "top5": baseline_result.get("top5", [])},
            "result": {"source_rank": source_rank, "exact_page_rank": exact_page_rank, "top5": [_candidate_item(candidate, score, text) for candidate, score, text in ranked[:5]]},
            "candidate_count": len(candidates), "candidate_count_before_selection": len(raw_candidates), "dedup_removed": removed, "latency_seconds": latency,
            "repaired": repaired, "regressed": regressed, "failure_categories": categories,
            "primary_failure_category": next((item for item in FAILURE_ORDER if item in categories), None), "evidence_diagnostics": diagnostics,
            "observed_explanation": (f"Observed association: expected page moved from baseline rank {baseline_rank} to {exact_page_rank}." if repaired or regressed else "No top-five exact-page status change."),
        })
        print(f"[{name} {number}/{len(cases)}] {case['id']} baseline={baseline_rank} result={exact_page_rank}", flush=True)
        if number % 10 == 0:
            _write_json(result_path, {"status": "running", "completed": number, "records": records})
    if cache_candidates and not cached and len(candidate_cache) == len(cases):
        _write_json(candidate_cache_path, candidate_cache)
    artifact = {
        "status": "complete", "configuration": {"name": name, "representations": [rep.name for rep in representations], "dense_k": dense_k, "bm25_k": bm25_k, "component_dense_k": component_dense_k,
            "component_bm25_k": component_bm25_k, "reranker": "BAAI/bge-reranker-v2-m3", "reranker_input": reranker_style, "selection": select,
            "rrf_k": RRF_K if select == "rrf" else None, "candidate_budget": RRF_CANDIDATE_BUDGET if select in {"rrf", "union_budget"} else None,
            "redundancy_deduplication": redundancy_deduplication, "dedup_method": "same_source_normalized_token_jaccard_ge_0.92" if redundancy_deduplication else None},
        "summary": {"metrics": _metrics(records, "result"), "repairs": [record["id"] for record in records if record["repaired"]], "regressions": [record["id"] for record in records if record["regressed"]],
            "paired_page5_vs_baseline": _mcnemar(records), "failure_category_counts": dict(Counter(category for record in records for category in record["failure_categories"])),
            "primary_failure_category_counts": dict(Counter(record["primary_failure_category"] for record in records if record["primary_failure_category"])),
            "latency_seconds": {"mean": statistics.mean(latencies), "median": statistics.median(latencies)}, "dedup_removed": {"mean": statistics.mean(removed_counts), "total": sum(removed_counts)}},
        "records": records,
    }
    _write_json(result_path, artifact)
    return artifact


def _pages_by_source(cache_dir: Path) -> dict[str, dict[int, str]]:
    return {document.filename: {page.page_number: page.raw_text for page in document.pages} for document in _iter_structured_documents(cache_dir)}


def _summary_row(name: str, artifact: dict[str, Any]) -> list[str]:
    metrics = artifact["summary"]["metrics"]
    return [name, *(f"{metrics[key]:.2%}" for key in ("source_top1", "source_top5", "exact_page_top1", "exact_page_top5", "exact_page_top20")), f"{metrics['mrr_source']:.4f}", str(len(artifact["summary"]["repairs"])), str(len(artifact["summary"]["regressions"])), f"{artifact['summary']['latency_seconds']['mean']:.3f}s"]


def write_report(output_dir: Path, artifacts: dict[str, dict[str, Any]], previous_structured: Path) -> None:
    previous = json.loads(previous_structured.read_text(encoding="utf-8"))["summary"]["structured_ingestion"]
    lines = ["# Controlled structured-ingestion retrieval ablations", "", "Retrieval only: these results do not measure answer correctness, citations, grounding, or abstention.", "",
        "| Configuration | Source@1 | Source@5 | Page@1 | Page@5 | Page@20 | MRR | Repairs | Regressions | Mean latency |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines.append("| Original baseline | 75.33% | 92.02% | 67.63% | 86.07% | 92.45% | 0.8256 | 0 | 0 | 0.912s |"); lines.append(
        f"| Previous structured ingestion | {previous['source_top1']:.2%} | {previous['source_top5']:.2%} | {previous['exact_page_top1']:.2%} | {previous['exact_page_top5']:.2%} | {previous['exact_page_top20']:.2%} | {previous['mrr_source']:.4f} | 18 | 23 | 1.221s |")
    for name, artifact in artifacts.items():
        lines.append("| " + " | ".join(_summary_row(name, artifact)) + " |")
    lines.extend(["", "## Per-configuration paired Page@5 comparison", ""])
    for name, artifact in artifacts.items():
        paired = artifact["summary"]["paired_page5_vs_baseline"]
        lines.extend([f"### {name}", "", f"- Repairs/regressions/net: {paired['repaired']}/{paired['regressed']}/{paired['net']:+d}", f"- Exact two-sided McNemar p-value: {paired['p_value']:.6g}", f"- Mean/median latency: {artifact['summary']['latency_seconds']['mean']:.3f}s / {artifact['summary']['latency_seconds']['median']:.3f}s", f"- Primary failure categories: {json.dumps(artifact['summary']['primary_failure_category_counts'], ensure_ascii=False)}", "",
            "Repairs: " + ", ".join(artifact["summary"]["repairs"]), "", "Regressions: " + ", ".join(artifact["summary"]["regressions"]), ""])
    (output_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "run"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation_queries.json"))
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--baseline-index-dir", type=Path, required=True)
    parser.add_argument("--previous-structured-dir", type=Path, required=True)
    parser.add_argument("--config", action="append", choices=("structured_baseline_chunking", "structured_full_hierarchy_text", "structured_metadata_hierarchy", "dual_baseline_structured_baseline_chunking", "structured_metadata_dedup", "reranker_full_hierarchy", "reranker_article_body", "reranker_body_only", "reranker_concise_metadata_body", "union_budget30", "rrf_budget30"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.config or ["structured_baseline_chunking", "structured_full_hierarchy_text", "structured_metadata_hierarchy", "dual_baseline_structured_baseline_chunking"]
    required = {"structured_baseline_chunking", "structured_metadata_hierarchy"}
    manifests = {name: build_representation(args.cache_dir, args.output_dir, name) for name in required}
    if args.command == "build":
        return
    cases, baseline = _load_evaluation(args.evaluation, args.baseline_artifact)
    pages = _pages_by_source(args.cache_dir)
    reps = {name: _load_search_representation(manifest) for name, manifest in manifests.items()}
    if "structured_full_hierarchy_text" in selected:
        _validate_reused_structured_cache(args.cache_dir, args.previous_structured_dir)
        reps["structured_full_hierarchy_text"] = _load_existing_representation(
            "structured_full_hierarchy_text", args.previous_structured_dir / "chunks.jsonl", args.previous_structured_dir / "chroma_db", "bct_structured_ingestion_experiment"
        )
    baseline_rep = _load_existing_representation("baseline", None, args.baseline_index_dir, "bct_regulations")
    artifacts: dict[str, dict[str, Any]] = {}
    for name in selected:
        if name == "structured_baseline_chunking":
            artifacts[name] = run_configuration(name=name, cases=cases, baseline=baseline, representations=[reps[name]], output_dir=args.output_dir, pages_by_source=pages, reranker_style="body_only")
        elif name == "structured_full_hierarchy_text":
            artifacts[name] = run_configuration(name=name, cases=cases, baseline=baseline, representations=[reps[name]], output_dir=args.output_dir, pages_by_source=pages, reranker_style="full_hierarchy")
        elif name == "structured_metadata_hierarchy":
            artifacts[name] = run_configuration(name=name, cases=cases, baseline=baseline, representations=[reps[name]], output_dir=args.output_dir, pages_by_source=pages, reranker_style="article_body", candidate_cache_id="structured_metadata_hierarchy")
        elif name == "dual_baseline_structured_baseline_chunking":
            artifacts[name] = run_configuration(name=name, cases=cases, baseline=baseline, representations=[baseline_rep, reps["structured_baseline_chunking"]], output_dir=args.output_dir, pages_by_source=pages, reranker_style="body_only", component_dense_k=10, component_bm25_k=8)
        elif name == "structured_metadata_dedup":
            artifacts[name] = run_configuration(name=name, cases=cases, baseline=baseline, representations=[reps["structured_metadata_hierarchy"]], output_dir=args.output_dir, pages_by_source=pages, reranker_style="article_body", candidate_cache_id="structured_metadata_hierarchy", redundancy_deduplication=True)
        elif name.startswith("reranker_"):
            styles = {"reranker_full_hierarchy": "full_hierarchy", "reranker_article_body": "article_body", "reranker_body_only": "body_only", "reranker_concise_metadata_body": "concise_metadata_body"}
            artifacts[name] = run_configuration(name=name, cases=cases, baseline=baseline, representations=[reps["structured_metadata_hierarchy"]], output_dir=args.output_dir, pages_by_source=pages, reranker_style=styles[name], candidate_cache_id="structured_metadata_hierarchy")
        elif name in {"union_budget30", "rrf_budget30"}:
            artifacts[name] = run_configuration(name=name, cases=cases, baseline=baseline, representations=[reps["structured_metadata_hierarchy"]], output_dir=args.output_dir, pages_by_source=pages, reranker_style="article_body", candidate_cache_id="structured_metadata_hierarchy", select="rrf" if name == "rrf_budget30" else "union_budget")
    write_report(args.output_dir, artifacts, args.previous_structured_dir / "benchmark_results.json")


if __name__ == "__main__":
    main()
