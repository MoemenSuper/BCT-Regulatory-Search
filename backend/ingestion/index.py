from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langfuse import get_client

from runtime_retrieval import (
    _load_bound_index,
    _read_chunks,
    cloud_embed_spec,
    create_cloud_runtime_client,
    document_binding,
)

logger = logging.getLogger(__name__)


def ingest_cloud_embed_enabled() -> bool:
    """Whether admin/CLI ingest should call Voyage/Google embed APIs.

    local / local_hybrid retrieve from Chroma (e5); cloud embeds are only needed for
    the cloud profile. Override with BCT_INGEST_CLOUD_INDEX=0|1.
    """
    configured = os.environ.get("BCT_INGEST_CLOUD_INDEX")
    if configured is not None and str(configured).strip() != "":
        return str(configured).strip() == "1"
    profile = (os.environ.get("BCT_DEFAULT_PROFILE") or "local_hybrid").strip().casefold()
    return profile == "cloud"


def resolve_active_assets(root: str | Path) -> Path:
    root = Path(root).resolve()
    pointer = root / "ACTIVE.json"
    if not pointer.exists():
        return root
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    relative = Path(str(payload["active_version"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid runtime asset pointer")
    active = (root / relative).resolve()
    active.relative_to(root)
    return active


def configure_runtime_assets(asset_root: str | Path, *, validate: bool = False) -> Path:
    """Point process env at the active asset version (shared by run_api + post-ingest refresh)."""
    asset_root = Path(asset_root).resolve()
    active = resolve_active_assets(asset_root)
    if validate:
        for filename in ("native.jsonl", "arabic_ocr_secondary.jsonl", "snapshot.json"):
            if not (active / filename).is_file():
                raise ValueError(f"Missing runtime asset: {active / filename}")
    snapshot = json.loads((active / "snapshot.json").read_text(encoding="utf-8"))
    # Legacy Codex snapshots are a list; versioned ingestion snapshots are dicts.
    meta = snapshot if isinstance(snapshot, dict) else {}
    os.environ.update(
        {
            "BCT_RUNTIME_ASSET_ROOT": str(asset_root),
            "BCT_VOYAGE_PROVIDER_ROOT": str(active),
            "BCT_NATIVE_CHUNKS_PATH": str(active / "native.jsonl"),
            "BCT_OCR_CHUNKS_PATH": str(active / "arabic_ocr_secondary.jsonl"),
        }
    )
    chroma_db = _resolve_local_chroma_db(meta.get("local_chroma_db"), asset_root=asset_root, active=active)
    if chroma_db is not None:
        os.environ["BCT_CHROMA_DB"] = str(chroma_db)
    if meta.get("local_collection"):
        os.environ["BCT_CHROMA_COLLECTION"] = str(meta["local_collection"])
    if meta.get("local_visual_collection") and chroma_db is not None:
        os.environ["BCT_OCR_CHROMA_DB"] = str(chroma_db)
        os.environ["BCT_OCR_CHROMA_COLLECTION"] = str(meta["local_visual_collection"])
    return active


def _resolve_local_chroma_db(recorded: object, *, asset_root: Path, active: Path) -> Path | None:
    """Prefer a real on-disk Chroma path; remap absolute bake-time paths after Docker seed."""
    if not recorded:
        fallback = asset_root / "local_chroma"
        return fallback.resolve() if fallback.is_dir() else None
    path = Path(str(recorded))
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((asset_root / path, active / path))
    candidates.extend((asset_root / "local_chroma", active / "local_chroma"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return (candidates[0] if candidates else asset_root / "local_chroma").resolve()


def _jsonl(path: Path, documents: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {"page_content": document.page_content, "metadata": document.metadata},
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def _array_bytes(vectors: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


def _write_bound_index(root: Path, representation: str, documents: list[Document], vectors: np.ndarray, *, spec=None) -> None:
    spec = spec or cloud_embed_spec()
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
    spec = spec or cloud_embed_spec()
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
    dimension = int(getattr(client, "dimension", cloud_embed_spec().dimension))
    if not documents:
        return np.empty((0, dimension), dtype=np.float32)
    texts = [document.page_content for document in documents]
    images = []
    titles = []
    for document in documents:
        titles.append(Path(str(document.metadata.get("source", ""))).stem or "none")
        image_path = str(document.metadata.get("page_image_path") or "").strip()
        png = None
        # Google-only multimodal: Voyage embed_document_chunks ignores images.
        if image_path:
            path = Path(image_path)
            if path.is_file():
                png = path.read_bytes()
        images.append(png)
    with get_client().start_as_current_observation(
        name="embed-chunks",
        as_type="embedding",
        model=str(getattr(client, "model", "") or cloud_embed_spec().model),
        input={"chunks": len(texts), "chars": sum(map(len, texts)), "images": sum(1 for png in images if png)},
    ):
        return client.embed_document_chunks(texts, images=images, titles=titles)


def stage_cloud_assets(
    *,
    asset_root: str | Path,
    new_primary: list[Document],
    new_visual: list[Document],
    content_sha256: str,
    source_filename: str,
    allow_empty: bool = False,
    removal: bool = False,
    embed_cloud: bool | None = None,
) -> tuple[Path, dict]:
    """Build a complete new asset version (JSONL + optional cloud embeddings).

    Voyage and Google indexes are separate files; they are never mixed. Switching
    provider on an existing corpus re-embeds kept chunks for that provider only.

    When embed_cloud is False (default for local / local_hybrid), JSONL is still
    updated and local Chroma can be staged separately — Voyage/Google are not called.
    """
    root = Path(asset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    active = resolve_active_assets(root)
    spec = cloud_embed_spec()
    if embed_cloud is None:
        embed_cloud = ingest_cloud_embed_enabled()
    old_primary, old_primary_vectors = _load_old(active, "native", "native.jsonl", spec=spec)
    old_visual, old_visual_vectors = _load_old(
        active, "arabic_ocr_secondary", "arabic_ocr_secondary.jsonl", spec=spec
    )
    source_key = Path(source_filename).name.casefold()

    def without_replaced_source(documents, vectors):
        keep = [
            index
            for index, document in enumerate(documents)
            if Path(str(document.metadata.get("source", ""))).name.casefold() != source_key
        ]
        if len(keep) == len(documents):
            return documents, vectors
        if vectors is None:
            return [documents[index] for index in keep], None
        return (
            [documents[index] for index in keep],
            vectors[keep] if keep else np.empty((0, spec.dimension), dtype=np.float32),
        )

    old_primary, old_primary_vectors = without_replaced_source(old_primary, old_primary_vectors)
    old_visual, old_visual_vectors = without_replaced_source(old_visual, old_visual_vectors)

    def merge_docs(old_docs, new_docs):
        return list(old_docs) + list(new_docs)

    if embed_cloud:
        client = create_cloud_runtime_client(root, spec)

        def merge(old_docs, old_vectors, new_docs):
            if old_vectors is None:
                combined = merge_docs(old_docs, new_docs)
                return combined, _embed_new(client, combined)
            new_vectors = _embed_new(client, new_docs)
            combined = merge_docs(old_docs, new_docs)
            if len(old_vectors):
                return combined, np.vstack([old_vectors, new_vectors])
            return combined, new_vectors

        all_primary, all_primary_vectors = merge(old_primary, old_primary_vectors, new_primary)
        all_visual, all_visual_vectors = merge(old_visual, old_visual_vectors, new_visual)
    else:
        all_primary = merge_docs(old_primary, new_primary)
        all_visual = merge_docs(old_visual, new_visual)
        # Carry forward prior cloud vectors only when no new chunks need embedding
        # (e.g. removal). Otherwise leave cloud indexes stale; local Chroma is enough.
        all_primary_vectors = old_primary_vectors if not new_primary else None
        all_visual_vectors = old_visual_vectors if not new_visual else None

    if not all_primary and not allow_empty:
        raise ValueError(
            f"Ingestion of {source_filename!r} produced no native searchable chunks "
            "(and the active corpus has none either)"
        )
    # Arabic visual/OCR secondary may stay empty for French-first corpora.
    # An empty secondary is valid; retrieval simply skips OCR fusion.

    version_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{content_sha256[:10]}-{uuid.uuid4().hex[:6]}"
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    staging = versions / f".staging-{version_id}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        _jsonl(staging / "native.jsonl", all_primary)
        _jsonl(staging / "arabic_ocr_secondary.jsonl", all_visual)
        if all_primary_vectors is not None:
            _write_bound_index(staging, "native", all_primary, all_primary_vectors, spec=spec)
        if all_visual_vectors is not None:
            _write_bound_index(
                staging, "arabic_ocr_secondary", all_visual, all_visual_vectors, spec=spec
            )
        # Preserve the other provider's indexes so switching back does not wipe them.
        # Also preserve current-provider indexes when we skipped embed (stale vs jsonl).
        other = "google" if spec.key == "voyage" else "voyage"
        other_spec = cloud_embed_spec(other)
        indexes_src = active / "indexes"
        if indexes_src.is_dir():
            indexes_dst = staging / "indexes"
            indexes_dst.mkdir(parents=True, exist_ok=True)
            for manifest_path in indexes_src.glob("*.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                provider = manifest.get("provider")
                model = manifest.get("model")
                keep_other = provider == other_spec.provider and model == other_spec.model
                keep_stale_current = (
                    not embed_cloud
                    and all_primary_vectors is None
                    and provider == spec.provider
                    and model == spec.model
                )
                if not (keep_other or keep_stale_current):
                    continue
                npy_path = manifest_path.with_suffix(".npy")
                if npy_path.is_file():
                    shutil.copy2(manifest_path, indexes_dst / manifest_path.name)
                    shutil.copy2(npy_path, indexes_dst / npy_path.name)
        snapshot = {
            "version": version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent": str(active.relative_to(root)) if active != root else "legacy-root",
            "cloud_retrieval_provider": spec.key,
            "cloud_embed": bool(embed_cloud),
            "native_chunks": len(all_primary),
            "arabic_visual_chunks": len(all_visual),
        }
        if removal:
            snapshot.update(
                {
                    "removed_document_sha256": content_sha256,
                    "removed_source": source_filename,
                    "added_native_chunks": 0,
                    "added_visual_chunks": 0,
                }
            )
        else:
            snapshot.update(
                {
                    "added_document_sha256": content_sha256,
                    "added_source": source_filename,
                    "added_native_chunks": len(new_primary),
                    "added_visual_chunks": len(new_visual),
                }
            )
        (staging / "snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        final = versions / version_id
        os.replace(staging, final)
        return final, snapshot
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def activate_assets(asset_root: str | Path, version_dir: str | Path, *, snapshot_updates: dict | None = None) -> dict:
    root = Path(asset_root).resolve()
    version = Path(version_dir).resolve()
    version.relative_to(root)
    snapshot_path = version / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot_updates:
        snapshot.update(snapshot_updates)
        snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    pointer = {
        "active_version": str(version.relative_to(root)),
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }
    temporary = root / f".ACTIVE-{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, root / "ACTIVE.json")
    return pointer


def _chroma_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return cleaned


def _document_chunk_id(document: Document) -> str:
    """Stable Chroma id; synthesize for legacy jsonl rows that predate chunk_id."""
    existing = document.metadata.get("chunk_id")
    if existing:
        return str(existing)
    seed = "|".join(
        [
            str(document.metadata.get("source", "")),
            str(document.metadata.get("page", "")),
            str(document.metadata.get("representation", "")),
            str(document.metadata.get("flat_part", "")),
            document.page_content,
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _collection_exists(client, name: str) -> bool:
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def _read_collection(source, *, batch_size: int = 500, exclude_source: str | None = None) -> list[tuple]:
    """Return (id, document, metadata, embedding) rows, skipping one source PDF.

    Chroma 1.5 can lose the not-yet-flushed tail of a collection's HNSW index
    ("Error finding id" / "Nothing found on disk") while documents and metadata
    stay readable; those pages are re-embedded from their text.
    """
    excluded = Path(exclude_source).name.casefold() if exclude_source else None
    rows: list[tuple] = []
    count = source.count()
    offset = 0
    while offset < count:
        limit = min(batch_size, count - offset)
        try:
            result = source.get(limit=limit, offset=offset, include=["documents", "metadatas", "embeddings"])
            embeddings = result.get("embeddings")
        except Exception as error:
            logger.warning("Chroma embeddings unreadable at offset %s of %s; re-embedding.", offset, source.name)
            get_client().create_event(
                name="reembed-unreadable-rows",
                level="WARNING",
                status_message=f"{type(error).__name__}: {error}",
                metadata={"collection": source.name, "offset": offset, "limit": limit},
            )
            result = source.get(limit=limit, offset=offset, include=["documents", "metadatas"])
            embeddings = None
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        kept = [
            index
            for index, metadata in enumerate(metadatas)
            if excluded is None or Path(str((metadata or {}).get("source", ""))).name.casefold() != excluded
        ]
        if embeddings is None and kept:
            embeddings = dict(zip(kept, _embed_local([documents[i] for i in kept])))
        rows.extend((ids[i], documents[i], metadatas[i], embeddings[i]) for i in kept)
        offset += len(ids)
        if not ids:
            break
    return rows


def _embed_local(texts: list[str]) -> list:
    from embedding import create_embedding_model

    model = create_embedding_model()
    with get_client().start_as_current_observation(
        name="embed-chunks",
        as_type="embedding",
        model=getattr(model, "model_name", None),
        input={"chunks": len(texts), "chars": sum(map(len, texts))},
    ):
        return model.embed_documents(texts)


def _add_rows(target, rows: list[tuple], *, batch_size: int = 500) -> None:
    for start in range(0, len(rows), batch_size):
        ids, documents, metadatas, embeddings = zip(*rows[start : start + batch_size])
        target.add(ids=list(ids), documents=list(documents), metadatas=list(metadatas), embeddings=list(embeddings))


def stage_local_collections(
    *,
    asset_root: str | Path,
    version_id: str,
    all_primary: list[Document],
    all_visual: list[Document],
    new_primary: list[Document],
    new_visual: list[Document],
    base_snapshot: dict,
    source_filename: str,
) -> dict[str, str]:
    """Create versioned Chroma collections, copying old embeddings when possible."""
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError("Local ingestion requires requirements-local.txt") from error

    root = Path(asset_root).resolve()
    db_path = Path(os.environ.get("BCT_CHROMA_DB", str(root / "local_chroma"))).resolve()
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    short = version_id.replace("-", "_")[-24:]
    primary_name = f"bct_regulations_{short}"
    visual_name = f"bct_arabic_visual_{short}"
    old_primary_name = str(base_snapshot.get("local_collection") or os.environ.get("BCT_CHROMA_COLLECTION", "bct_regulations"))
    old_visual_name = str(base_snapshot.get("local_visual_collection") or os.environ.get("BCT_OCR_CHROMA_COLLECTION", "bct_arabic_ocr_secondary_v1"))

    has_old_primary = _collection_exists(client, old_primary_name)
    has_old_visual = _collection_exists(client, old_visual_name)
    # ponytail: ceiling=full local re-embed via ingest CLI when seeding Chroma from a legacy
    # cloud-only corpus (no existing collection). Admin PDF upload should not block on that.
    if not has_old_primary and len(all_primary) > len(new_primary):
        get_client().create_event(
            name="skip-local-index",
            level="WARNING",
            status_message=f"No prior Chroma collection {old_primary_name!r}; new chunks are not searchable locally",
            metadata={"chroma_db": str(db_path), "all_primary": len(all_primary), "new_primary": len(new_primary)},
        )
        return {}

    # Read old rows before any write: with the serving retriever holding these collections
    # open, Chroma 1.5 fails reads ("Error finding id") once the same process starts writing.
    with get_client().start_as_current_observation(
        name="copy-prior-collections",
        input={"chroma_db": str(db_path), "primary": old_primary_name, "visual": old_visual_name},
    ) as copy_span:
        old_primary_rows = (
            _read_collection(client.get_collection(old_primary_name), exclude_source=source_filename)
            if has_old_primary
            else []
        )
        old_visual_rows = (
            _read_collection(client.get_collection(old_visual_name), exclude_source=source_filename)
            if has_old_visual
            else []
        )
        copy_span.update(output={"primary_rows": len(old_primary_rows), "visual_rows": len(old_visual_rows)})
    primary_target = client.create_collection(primary_name)
    visual_target = client.create_collection(visual_name)
    try:
        _add_rows(primary_target, old_primary_rows)
        _add_rows(visual_target, old_visual_rows)
        primary_to_add = new_primary if has_old_primary else all_primary
        visual_to_add = new_visual if has_old_visual else all_visual

        def add(collection, documents):
            if not documents:
                return
            texts = [doc.page_content for doc in documents]
            vectors = _embed_local(texts)
            collection.upsert(
                ids=[_document_chunk_id(doc) for doc in documents],
                documents=texts,
                metadatas=[_chroma_metadata(doc.metadata) for doc in documents],
                embeddings=vectors,
            )

        add(primary_target, primary_to_add)
        add(visual_target, visual_to_add)
        if primary_target.count() != len(all_primary) or visual_target.count() != len(all_visual):
            raise ValueError("Staged local Chroma collection counts do not match runtime chunks")
        return {
            "local_chroma_db": str(db_path),
            "local_collection": primary_name,
            "local_visual_collection": visual_name,
        }
    except Exception:
        try:
            client.delete_collection(primary_name)
        except Exception:
            pass
        try:
            client.delete_collection(visual_name)
        except Exception:
            pass
        raise


def discard_staged_local_collections(snapshot_updates: dict[str, object]) -> None:
    """Best-effort cleanup for versioned Chroma collections that were never activated."""
    db_value = snapshot_updates.get("local_chroma_db")
    names = [
        snapshot_updates.get("local_collection"),
        snapshot_updates.get("local_visual_collection"),
    ]
    if not db_value or not any(names):
        return
    try:
        import chromadb
    except ImportError:
        return
    try:
        client = chromadb.PersistentClient(path=str(db_value))
    except Exception:
        return
    for name in names:
        if not name:
            continue
        try:
            client.delete_collection(str(name))
        except Exception:
            pass
