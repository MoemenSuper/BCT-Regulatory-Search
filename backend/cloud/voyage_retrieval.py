"""Cloud profile retrieval: persisted Voyage Context-4 vectors + live query embed/rerank."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from bm25 import create_bm25, retrieve_bm25
from cloud.voyage_client import VOYAGE_SPEC, CloudEmbedSpec, create_cloud_runtime_client
from retrieval_selection import (
    build_identity_reranker_documents,
    expand_answer_pages,
    is_arabic_query,
    page_chunks,
    parse_query_identity,
    query_instrument_refs,
    source_matches_identity,
)
from runtime_retrieval import _identity_diversified_rank, _read_chunks, dedupe, document_binding, script_scores


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

    def _embed(self, text):
        vector = np.asarray(self.client.embed_query(text), dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def retrieve(self, query, other_queries=(), instruments=()):
        """Same contract as the local backend: other_queries are other wordings of the question,
        instruments are extra instruments to search inside."""
        versions = [query, *other_queries]
        vectors = [self._embed(version) for version in versions]
        groups = []
        for version, query_vector in zip(versions, vectors):
            groups.append(
                _provider_candidates(
                    self.native_documents,
                    self.native_vectors,
                    query_vector,
                    self.native_bm25,
                    version,
                    # the other wordings bring fewer candidates each, as in the local backend
                    dense_k=20 if version == query else 10,
                    bm25_k=15 if version == query else 8,
                )
            )
            if is_arabic_query(version):
                groups.append(
                    _provider_candidates(
                        self.ocr_documents,
                        self.ocr_vectors,
                        query_vector,
                        self.ocr_bm25,
                        version,
                        dense_k=5,
                        bm25_k=5,
                    )
                )
        identity_refs = query_instrument_refs(query) + list(instruments)
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
                    matching, self.native_vectors[indices], vectors[0],
                    create_bm25(matching), query,
                    dense_k=20, bm25_k=15,
                ))
        documents = dedupe(*groups)
        reranker_documents = build_identity_reranker_documents(
            documents, parse_query_identity(query)
        )
        # Each chunk against one wording in its own script, as in the local backend.
        scores = script_scores(
            lambda wording, chunks: self.client.rerank(wording, [chunk.page_content for chunk in chunks]),
            query, other_queries, reranker_documents,
        )
        return _identity_diversified_rank(query, documents, scores)


def _load_bound_index(
    provider_root: Path,
    representation: str,
    documents: list[Document],
    spec: CloudEmbedSpec | None = None,
) -> np.ndarray:
    spec = spec or VOYAGE_SPEC
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
    spec = spec or getattr(client, "spec", None) or VOYAGE_SPEC
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
    spec = VOYAGE_SPEC
    client = create_cloud_runtime_client(provider_root)
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
