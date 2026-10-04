"""Voyage document indexes staged beside each asset version (cloud profile only).

Local profiles search Chroma; these indexes exist so the cloud profile can load the
same asset version. Ingest embeds only when BCT_INGEST_CLOUD_INDEX=1 or the default
profile is cloud; otherwise the previous indexes are carried forward.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langfuse import get_client

from cloud.voyage_client import VOYAGE_SPEC, create_cloud_runtime_client
from cloud.voyage_retrieval import _load_bound_index
from ingestion.index import _keep_indices
from runtime_retrieval import _read_chunks, document_binding


def ingest_cloud_embed_enabled() -> bool:
    """Whether admin/CLI ingest should call the Voyage embed API.

    local / local_hybrid retrieve from Chroma (e5); cloud embeds are only needed for
    the cloud profile. Override with BCT_INGEST_CLOUD_INDEX=0|1.
    """
    configured = os.environ.get("BCT_INGEST_CLOUD_INDEX")
    if configured is not None and str(configured).strip() != "":
        return str(configured).strip() == "1"
    profile = (os.environ.get("BCT_DEFAULT_PROFILE") or "local_hybrid").strip().casefold()
    return profile == "cloud"


def _array_bytes(vectors: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def _write_bound_index(root: Path, representation: str, documents: list[Document], vectors: np.ndarray, *, spec=None) -> None:
    spec = spec or VOYAGE_SPEC
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(documents), spec.dimension):
        raise ValueError(f"Invalid {representation} vector shape: {vectors.shape}")
    if not np.isfinite(vectors).all() or (len(vectors) and np.any(np.linalg.norm(vectors, axis=1) <= 0)):
        raise ValueError(f"Invalid {representation} vectors")
    texts = [hashlib.sha256(document.page_content.encode("utf-8")).hexdigest() for document in documents]
    binding = document_binding(documents)
    array = _array_bytes(vectors)
    index_dir = root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.model}-{representation}-{binding[:16]}"
    npy_path = index_dir / f"{stem}.npy"
    json_path = index_dir / f"{stem}.json"
    npy_path.write_bytes(array)
    manifest = {
        "provider": spec.provider,
        "model": spec.model,
        "task": "document",
        "representation": representation,
        "contextual": spec.contextual,
        "texts": texts,
        "documents_sha256": binding,
        "array_sha256": hashlib.sha256(array).hexdigest().upper(),
        "dimension": spec.dimension,
        "shape": [len(documents), spec.dimension],
        "dtype": "float32",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _load_old(active: Path, representation: str, filename: str, *, spec=None):
    """Load chunks + matching provider vectors. Missing provider index ⇒ vectors None (re-embed)."""
    spec = spec or VOYAGE_SPEC
    path = active / filename
    if not path.exists():
        return [], np.empty((0, spec.dimension), dtype=np.float32)
    documents = _read_chunks(path)
    try:
        vectors = _load_bound_index(active, representation, documents, spec)
    except ValueError as error:
        if "No bound" in str(error) and "not interchangeable" in str(error):
            # First build for this provider (or wrong provider indexes only).
            return documents, None
        raise
    return documents, vectors


def _embed_new(client, documents: list[Document]) -> np.ndarray:
    dimension = int(getattr(client, "dimension", VOYAGE_SPEC.dimension))
    if not documents:
        return np.empty((0, dimension), dtype=np.float32)
    texts = [document.page_content for document in documents]
    with get_client().start_as_current_observation(
        name="embed-chunks",
        as_type="embedding",
        model=str(getattr(client, "model", "") or VOYAGE_SPEC.model),
        input={"chunks": len(texts), "chars": sum(map(len, texts))},
    ):
        return client.embed_document_chunks(texts)


def stage_voyage_indexes(
    *,
    root: Path,
    active: Path,
    staging: Path,
    source_keys: set[str],
    new_primary: list[Document],
    new_visual: list[Document],
    embed: bool | None = None,
) -> dict:
    """Write the staged version's Voyage indexes; return its snapshot fields.

    The staged chunks are the active ones minus source_keys, then the new ones, in the
    same order stage_assets writes the JSONL. Without embedding, old vectors carry over
    only when nothing new needs a vector (a removal); otherwise the old indexes are
    kept as they were (stale against the JSONL) so the cloud profile still has them.
    """
    spec = VOYAGE_SPEC
    if embed is None:
        embed = ingest_cloud_embed_enabled()
    client = None
    primary_vectors = None
    for representation, filename, new_docs in (
        ("native", "native.jsonl", new_primary),
        ("arabic_ocr_secondary", "arabic_ocr_secondary.jsonl", new_visual),
    ):
        old_docs, old_vectors = _load_old(active, representation, filename, spec=spec)
        keep = _keep_indices(old_docs, source_keys)
        if len(keep) != len(old_docs):
            old_docs = [old_docs[index] for index in keep]
            if old_vectors is not None:
                old_vectors = old_vectors[keep] if keep else np.empty((0, spec.dimension), dtype=np.float32)
        documents = old_docs + list(new_docs)
        if embed:
            client = client or create_cloud_runtime_client(root)
            if old_vectors is None:
                vectors = _embed_new(client, documents)
            else:
                new_vectors = _embed_new(client, new_docs)
                vectors = np.vstack([old_vectors, new_vectors]) if len(old_vectors) else new_vectors
        else:
            vectors = old_vectors if not new_docs else None
        if vectors is not None:
            _write_bound_index(staging, representation, documents, vectors, spec=spec)
        if representation == "native":
            primary_vectors = vectors

    indexes_src = active / "indexes"
    if not embed and primary_vectors is None and indexes_src.is_dir():
        indexes_dst = staging / "indexes"
        indexes_dst.mkdir(parents=True, exist_ok=True)
        for manifest_path in indexes_src.glob("*.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("provider") != spec.provider or manifest.get("model") != spec.model:
                continue
            npy_path = manifest_path.with_suffix(".npy")
            if npy_path.is_file():
                shutil.copy2(manifest_path, indexes_dst / manifest_path.name)
                shutil.copy2(npy_path, indexes_dst / npy_path.name)
    return {"cloud_retrieval_provider": spec.key, "cloud_embed": bool(embed)}
