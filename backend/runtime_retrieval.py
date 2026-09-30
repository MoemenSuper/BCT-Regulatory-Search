"""Local retrieval (default for local_hybrid and local): e5 + Chroma, BM25, BGE reranker.

The cloud profile's Voyage backend lives in cloud/voyage_retrieval.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from bm25 import retrieve_bm25
from retrieval_selection import (
    diversify_ranked_pages,
    build_identity_reranker_documents,
    expand_answer_pages,
    page_chunks,
    parse_query_identity,
    prefer_historical_hits,
    prefer_named_instrument_hits,
    query_instrument_refs,
    source_matches_identity,
    is_arabic_query,
)
from reranker import score_documents
from vector_store import retrieve_relevant_chunks
from langchain_core.documents import Document


def _doc_key(document):
    return (
        document.page_content,
        document.metadata.get("source"),
        document.metadata.get("page"),
    )


def dedupe(*groups):
    """Deduplicate documents across retrieval lanes by content + source + page."""
    combined = []
    seen = set()
    for documents in groups:
        for document in documents:
            key = _doc_key(document)
            if key not in seen:
                seen.add(key)
                combined.append(document)
    return combined


def _identity_diversified_rank(query, documents, scores):
    scores = [float(score) for score in scores]
    if len(scores) != len(documents):
        raise ValueError("Reranker returned an invalid score count")
    ranked = diversify_ranked_pages(
        sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
    )
    # Named before historical: "Avant 2025-13" must demote the named cutoff last.
    ranked = prefer_named_instrument_hits(ranked, query_instrument_refs(query))
    return prefer_historical_hits(ranked, query)


class LocalRetrievalBackend:
    def __init__(
        self,
        vector_store,
        reranker,
        bm25,
        bm25_documents,
        *,
        ocr_vector_store=None,
        ocr_bm25=None,
        ocr_documents=None,
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.bm25 = bm25
        self.bm25_documents = bm25_documents
        self.ocr_vector_store = ocr_vector_store
        self.ocr_bm25 = ocr_bm25
        self.ocr_documents = list(ocr_documents or [])
        self._pages = page_chunks([*bm25_documents, *self.ocr_documents])

    def expand_pages(self, ranked):
        """Full retrieved page text, plus bounded same-PDF neighbour pages."""
        return expand_answer_pages(ranked, self._pages)

    def retrieve(self, query):
        dense = retrieve_relevant_chunks(query, self.vector_store)
        sparse = retrieve_bm25(query, self.bm25, self.bm25_documents)
        groups = [dense, sparse]
        if is_arabic_query(query) and self.ocr_vector_store is not None:
            groups.extend(
                (
                    retrieve_relevant_chunks(query, self.ocr_vector_store, k=5),
                    retrieve_bm25(
                        query,
                        self.ocr_bm25,
                        self.ocr_documents,
                        k=5,
                    ),
                )
            )
        identity_refs = query_instrument_refs(query)
        if identity_refs:
            matched_docs = []
            for identity in identity_refs:
                matched_docs.extend(
                    document
                    for document in self.bm25_documents
                    if source_matches_identity(
                        str(document.metadata.get("source", "")), identity
                    )
                )
            if matched_docs:
                # ponytail: named-instrument lane before semantic/latest preference
                groups.append(matched_docs[:40])
        import chat_tracing

        lanes = ["dense", "bm25", "ocr_dense", "ocr_bm25"] if is_arabic_query(query) and self.ocr_vector_store is not None else ["dense", "bm25"]
        lanes += ["extra"] * (len(groups) - len(lanes))
        chat_tracing.event("retrieval-lanes", output={
            name: chat_tracing.brief(group, limit=8) for name, group in zip(lanes, groups)
        }, metadata={"identity_refs": [str(r) for r in identity_refs], "pool": len(dedupe(*groups))})
        return self.rank(query, dedupe(*groups))

    def rank(self, query, documents):
        reranker_documents = build_identity_reranker_documents(
            documents,
            parse_query_identity(query),
        )
        scores = score_documents(self.reranker, query, reranker_documents)
        return _identity_diversified_rank(query, documents, scores)


def _read_chunks(path: Path) -> list[Document]:
    """Load JSONL chunks. An empty file is a valid empty corpus (bootstrap / FR-only OCR)."""
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            documents.append(
                Document(
                    page_content=value["page_content"],
                    metadata=value["metadata"],
                )
            )
    return documents


def document_binding(documents: list[Document]) -> str:
    """Bind ordered text and citation metadata, not just embedding inputs."""
    digest = hashlib.sha256()
    for document in documents:
        record = {"page_content": document.page_content, "metadata": document.metadata}
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def create_local_backend():
    from embedding import create_embedding_model
    from vector_store import load_vector_store
    from reranker import create_reranker
    from bm25 import load_documents_from_chroma, create_bm25

    embedding_model = create_embedding_model()
    vector_store = load_vector_store(embedding_model)
    reranker = create_reranker()
    documents = load_documents_from_chroma(vector_store)
    bm25 = create_bm25(documents)

    ocr_vector_store = None
    ocr_documents = []
    ocr_bm25 = None
    if os.environ.get("BCT_OCR_CHROMA_DB"):
        ocr_vector_store = load_vector_store(
            embedding_model,
            persist_directory=os.environ["BCT_OCR_CHROMA_DB"],
            collection_name=os.environ.get("BCT_OCR_CHROMA_COLLECTION", "bct_arabic_ocr_secondary_v1"),
        )
        ocr_documents = load_documents_from_chroma(ocr_vector_store)
        ocr_bm25 = create_bm25(ocr_documents)
    if not documents:
        # Valid before the first ingest, but a wrong --assets root looks identical.
        print(
            f"WARNING: local corpus is empty (collection {os.environ.get('BCT_CHROMA_COLLECTION')!r} "
            f"in {os.environ.get('BCT_CHROMA_DB')!r}); every question will find no evidence.",
            flush=True,
        )
    backend = LocalRetrievalBackend(
        vector_store,
        reranker,
        bm25,
        documents,
        ocr_vector_store=ocr_vector_store,
        ocr_bm25=ocr_bm25,
        ocr_documents=ocr_documents,
    )
    from jsonl_supersession import maybe_wrap_backend

    native = os.environ.get("BCT_NATIVE_CHUNKS_PATH")
    return maybe_wrap_backend(backend, Path(native).resolve().parent if native else None)
