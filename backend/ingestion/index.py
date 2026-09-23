from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from runtime_retrieval import (
    _load_bound_index,
    _read_chunks,
    cloud_embed_spec,
    create_cloud_runtime_client,
    document_binding,
)


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
    if meta.get("local_chroma_db"):
        os.environ["BCT_CHROMA_DB"] = str(meta["local_chroma_db"])
    if meta.get("local_collection"):
        os.environ["BCT_CHROMA_COLLECTION"] = str(meta["local_collection"])
    if meta.get("local_visual_collection"):
        os.environ["BCT_OCR_CHROMA_DB"] = str(meta.get("local_chroma_db"))
        os.environ["BCT_OCR_CHROMA_COLLECTION"] = str(meta["local_visual_collection"])
    return active


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
) -> tuple[Path, dict]:
    """Build a complete new cloud asset version while embedding only new chunks.

    Voyage and Google indexes are separate files; they are never mixed. Switching
    provider on an existing corpus re-embeds kept chunks for that provider only.
    """
    root = Path(asset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    active = resolve_active_assets(root)
    spec = cloud_embed_spec()
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

    client = create_cloud_runtime_client(root, spec)

    def merge(old_docs, old_vectors, new_docs):
        if old_vectors is None:
            combined = list(old_docs) + list(new_docs)
            return combined, _embed_new(client, combined)
        new_vectors = _embed_new(client, new_docs)
        combined = list(old_docs) + list(new_docs)
        if len(old_vectors):
            return combined, np.vstack([old_vectors, new_vectors])
        return combined, new_vectors

    all_primary, all_primary_vectors = merge(old_primary, old_primary_vectors, new_primary)
    all_visual, all_visual_vectors = merge(old_visual, old_visual_vectors, new_visual)

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
        _write_bound_index(staging, "native", all_primary, all_primary_vectors, spec=spec)
        _write_bound_index(
            staging, "arabic_ocr_secondary", all_visual, all_visual_vectors, spec=spec
        )
        # Preserve the other provider's indexes so switching back does not wipe them.
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
                if (
                    manifest.get("provider") == other_spec.provider
                    and manifest.get("model") == other_spec.model
                ):
                    npy_path = manifest_path.with_suffix(".npy")
                    if npy_path.is_file():
                        shutil.copy2(manifest_path, indexes_dst / manifest_path.name)
                        shutil.copy2(npy_path, indexes_dst / npy_path.name)
        snapshot = {
            "version": version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent": str(active.relative_to(root)) if active != root else "legacy-root",
            "cloud_retrieval_provider": spec.key,
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


def _copy_collection(source, target, *, batch_size: int = 500, exclude_source: str | None = None) -> None:
    count = source.count()
    offset = 0
    while offset < count:
        result = source.get(
            limit=min(batch_size, count - offset),
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = []
        selected = [
            index
            for index, metadata in enumerate(metadatas)
            if exclude_source is None
            or Path(str((metadata or {}).get("source", ""))).name.casefold() != Path(exclude_source).name.casefold()
        ]
        if selected:
            target.add(
                ids=[ids[index] for index in selected],
                documents=[documents[index] for index in selected],
                metadatas=[metadatas[index] for index in selected],
                embeddings=[embeddings[index] for index in selected],
            )
        offset += len(ids)
        if not ids:
            break


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
    from embedding import create_embedding_model

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
        return {}

    primary_target = client.create_collection(primary_name)
    visual_target = client.create_collection(visual_name)
    embedding = create_embedding_model()
    try:
        if has_old_primary:
            _copy_collection(client.get_collection(old_primary_name), primary_target, exclude_source=source_filename)
            primary_to_add = new_primary
        else:
            primary_to_add = all_primary
        if has_old_visual:
            _copy_collection(client.get_collection(old_visual_name), visual_target, exclude_source=source_filename)
            visual_to_add = new_visual
        else:
            visual_to_add = all_visual

        def add(collection, documents):
            if not documents:
                return
            texts = [doc.page_content for doc in documents]
            vectors = embedding.embed_documents(texts)
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
