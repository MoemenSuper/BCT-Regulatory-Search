"""Retrieval backends used by the selectable interactive runtime profiles."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import requests

from bm25 import create_bm25, retrieve_bm25
from retrieval_selection import (
    diversify_ranked_pages,
    build_identity_reranker_documents,
    expand_answer_pages,
    expand_ranked_pages,
    explicit_instrument_identity,
    EXPORT_SETTLEMENT_BM25_QUERY,
    NONPRIORITY_IMPORT_BM25_QUERY,
    page_chunks,
    parse_query_identity,
    parse_source_identity,
    prefer_historical_hits,
    prefer_named_instrument_hits,
    prefer_regime_hits,
    query_instrument_refs,
    query_regulatory_regime,
    source_matches_identity,
    is_arabic_query,
)
from reranker import score_documents
from vector_store import retrieve_relevant_chunks
from langchain_core.documents import Document



from cloud_embed_clients import (
    CLOUD_EMBED_SPECS,
    CloudEmbedSpec,
    CloudRetrievalUsage,
    GoogleRuntimeClient,
    VoyageRuntimeClient,
    cloud_embed_spec,
    create_cloud_runtime_client,
    track_cloud_retrieval_usage,
    _record_voyage_usage,
)

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
    ranked = prefer_regime_hits(ranked, query)
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
        else:
            regime = query_regulatory_regime(query)
            if regime == "export_settlement":
                groups.append(
                    retrieve_bm25(
                        EXPORT_SETTLEMENT_BM25_QUERY,
                        self.bm25,
                        self.bm25_documents,
                        k=15,
                    )
                )
            elif regime == "nonpriority_import":
                groups.append(
                    retrieve_bm25(
                        NONPRIORITY_IMPORT_BM25_QUERY,
                        self.bm25,
                        self.bm25_documents,
                        k=15,
                    )
                )
        return self.rank(query, dedupe(*groups))

    def rank(self, query, documents):
        reranker_documents = build_identity_reranker_documents(
            documents,
            parse_query_identity(query),
        )
        scores = score_documents(self.reranker, query, reranker_documents)
        return _identity_diversified_rank(query, documents, scores)


def _provider_candidates(
    documents,
    vectors,
    query_vector,
    bm25,
    query,
    *,
    dense_k,
    bm25_k,
):
    similarities = vectors @ query_vector
    dense_indices = np.argsort(-similarities)[:dense_k]
    return dedupe(
        [documents[int(index)] for index in dense_indices],
        retrieve_bm25(query, bm25, documents, k=bm25_k),
    )


class VoyageRetrievalBackend:
    """Use persisted Context-4 document vectors and live query/rerank calls."""

    def __init__(
        self,
        *,
        native_documents,
        native_vectors,
        ocr_documents,
        ocr_vectors,
        client,
    ):
        self.native_documents = list(native_documents)
        self.native_vectors = np.asarray(native_vectors, dtype=np.float32)
        self.ocr_documents = list(ocr_documents)
        self.ocr_vectors = np.asarray(ocr_vectors, dtype=np.float32)
        self.client = client
        self.native_bm25 = create_bm25(self.native_documents)
        self.ocr_bm25 = create_bm25(self.ocr_documents)
        self._pages = page_chunks([*self.native_documents, *self.ocr_documents])

    def expand_pages(self, ranked):
        """Full retrieved page text, plus bounded same-PDF neighbour pages."""
        return expand_answer_pages(ranked, self._pages)

    def retrieve(self, query):
        query_vector = np.asarray(self.client.embed_query(query), dtype=np.float32)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        groups = [
            _provider_candidates(
                self.native_documents,
                self.native_vectors,
                query_vector,
                self.native_bm25,
                query,
                dense_k=20,
                bm25_k=15,
            )
        ]
        if is_arabic_query(query):
            groups.append(
                _provider_candidates(
                    self.ocr_documents,
                    self.ocr_vectors,
                    query_vector,
                    self.ocr_bm25,
                    query,
                    dense_k=5,
                    bm25_k=5,
                )
            )
        identity_refs = query_instrument_refs(query)
        if identity_refs:
            indices = []
            seen_idx: set[int] = set()
            for identity in identity_refs:
                for index, document in enumerate(self.native_documents):
                    if index in seen_idx:
                        continue
                    if source_matches_identity(
                        str(document.metadata.get("source", "")), identity
                    ):
                        seen_idx.add(index)
                        indices.append(index)
            if indices:
                matching = [self.native_documents[index] for index in indices]
                groups.append(_provider_candidates(
                    matching, self.native_vectors[indices], query_vector,
                    create_bm25(matching), query,
                    dense_k=20, bm25_k=15,
                ))
        else:
            regime = query_regulatory_regime(query)
            if regime == "export_settlement":
                groups.append(
                    retrieve_bm25(
                        EXPORT_SETTLEMENT_BM25_QUERY,
                        self.native_bm25,
                        self.native_documents,
                        k=15,
                    )
                )
            elif regime == "nonpriority_import":
                groups.append(
                    retrieve_bm25(
                        NONPRIORITY_IMPORT_BM25_QUERY,
                        self.native_bm25,
                        self.native_documents,
                        k=15,
                    )
                )
        documents = dedupe(*groups)
        reranker_documents = build_identity_reranker_documents(
            documents, parse_query_identity(query)
        )
        scores = self.client.rerank(
            query,
            [document.page_content for document in reranker_documents],
        )
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


def _load_bound_index(
    provider_root: Path,
    representation: str,
    documents: list[Document],
    spec: CloudEmbedSpec | None = None,
) -> np.ndarray:
    spec = spec or CLOUD_EMBED_SPECS["voyage"]
    if not documents:
        return np.empty((0, spec.dimension), dtype=np.float32)
    text_hashes = [
        hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
        for document in documents
    ]
    matches = []
    for manifest_path in (provider_root / "indexes").glob("*.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("provider") == spec.provider
            and manifest.get("model") == spec.model
            and manifest.get("task") == "document"
            and manifest.get("representation") == representation
            and manifest.get("contextual") is spec.contextual
            and manifest.get("texts") == text_hashes
        ):
            matches.append((manifest_path, manifest))
    if len(matches) != 1:
        raise ValueError(
            f"No bound {spec.provider} index for {representation}; "
            f"found {len(matches)} matches (provider indexes are not interchangeable)"
        )
    manifest_path, manifest = matches[0]
    if manifest.get("documents_sha256") != document_binding(documents):
        raise ValueError(
            f"{spec.provider} document metadata binding mismatch: {manifest_path}"
        )
    index_path = manifest_path.with_suffix(".npy")
    index_bytes = index_path.read_bytes()
    if hashlib.sha256(index_bytes).hexdigest().upper() != str(
        manifest.get("array_sha256", "")
    ).upper():
        raise ValueError(f"{spec.provider} index hash mismatch: {index_path}")
    vectors = np.load(io.BytesIO(index_bytes), allow_pickle=False)
    expected_shape = [len(documents), int(manifest["dimension"])]
    if (
        list(vectors.shape) != expected_shape
        or list(vectors.shape) != manifest.get("shape")
        or vectors.dtype != np.float32
        or manifest.get("dtype") != "float32"
        or not np.isfinite(vectors).all()
        or np.any(np.linalg.norm(vectors, axis=1) <= 0)
    ):
        raise ValueError(f"{spec.provider} index shape or dtype mismatch: {index_path}")
    return vectors


def load_voyage_backend(
    *,
    provider_root,
    native_chunks,
    ocr_chunks,
    client,
    spec: CloudEmbedSpec | None = None,
) -> VoyageRetrievalBackend:
    spec = spec or getattr(client, "spec", None) or CLOUD_EMBED_SPECS["voyage"]
    provider_root = Path(provider_root)
    native_documents = _read_chunks(Path(native_chunks))
    ocr_documents = _read_chunks(Path(ocr_chunks))
    return VoyageRetrievalBackend(
        native_documents=native_documents,
        native_vectors=_load_bound_index(
            provider_root, "native", native_documents, spec
        ),
        ocr_documents=ocr_documents,
        ocr_vectors=_load_bound_index(
            provider_root, "arabic_ocr_secondary", ocr_documents, spec
        ),
        client=client,
    )


def create_voyage_backend_from_environment():
    names = {
        "BCT_VOYAGE_PROVIDER_ROOT": os.environ.get("BCT_VOYAGE_PROVIDER_ROOT"),
        "BCT_NATIVE_CHUNKS_PATH": os.environ.get("BCT_NATIVE_CHUNKS_PATH"),
        "BCT_OCR_CHUNKS_PATH": os.environ.get("BCT_OCR_CHUNKS_PATH"),
    }
    missing = [name for name, value in names.items() if not value]
    if missing:
        raise RuntimeError(
            "Cloud profile requires: " + ", ".join(missing)
        )
    provider_root = Path(names["BCT_VOYAGE_PROVIDER_ROOT"])
    spec = cloud_embed_spec()
    client = create_cloud_runtime_client(provider_root, spec)
    backend = load_voyage_backend(
        provider_root=provider_root,
        native_chunks=names["BCT_NATIVE_CHUNKS_PATH"],
        ocr_chunks=names["BCT_OCR_CHUNKS_PATH"],
        client=client,
        spec=spec,
    )
    # Optional JSONL SUPERSEDES pin (force/imperfect instrument questions).
    from jsonl_supersession import maybe_wrap_backend

    assets_root = Path(names["BCT_NATIVE_CHUNKS_PATH"]).resolve().parent
    return maybe_wrap_backend(backend, assets_root)
