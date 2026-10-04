from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langfuse import get_client

from runtime_retrieval import _read_chunks

logger = logging.getLogger(__name__)


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


def _keep_indices(documents: list[Document], source_keys: set[str]) -> list[int]:
    """Positions of the chunks that do not come from source_keys (the PDFs being replaced)."""
    return [
        index
        for index, document in enumerate(documents)
        if Path(str(document.metadata.get("source", ""))).name.casefold() not in source_keys
    ]


def _read_active(active: Path, filename: str) -> list[Document]:
    path = active / filename
    return _read_chunks(path) if path.exists() else []


def stage_assets(
    *,
    asset_root: str | Path,
    new_primary: list[Document],
    new_visual: list[Document],
    content_sha256: str,
    source_filenames: list[str],
    allow_empty: bool = False,
    removal: bool = False,
    embed_cloud: bool | None = None,
) -> tuple[Path, dict]:
    """Build a complete new asset version: the JSONL chunks, plus the cloud profile's
    Voyage indexes (cloud/voyage_index.py; embeds only when embed_cloud is on).

    source_filenames: the PDFs added (or removed) in this version, one or a whole upload batch;
    their old chunks are replaced by new_primary / new_visual.
    """
    root = Path(asset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    active = resolve_active_assets(root)
    source_keys = {Path(name).name.casefold() for name in source_filenames}
    sources = ", ".join(source_filenames)

    def merged(filename, new_docs):
        old = _read_active(active, filename)
        return [old[index] for index in _keep_indices(old, source_keys)] + list(new_docs)

    all_primary = merged("native.jsonl", new_primary)
    all_visual = merged("arabic_ocr_secondary.jsonl", new_visual)
    if not all_primary and not allow_empty:
        raise ValueError(
            f"Ingestion of {sources!r} produced no native searchable chunks "
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
        from cloud.voyage_index import stage_voyage_indexes

        cloud_fields = stage_voyage_indexes(
            root=root,
            active=active,
            staging=staging,
            source_keys=source_keys,
            new_primary=new_primary,
            new_visual=new_visual,
            embed=embed_cloud,
        )
        snapshot = {
            "version": version_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent": str(active.relative_to(root)) if active != root else "legacy-root",
            **cloud_fields,
            "native_chunks": len(all_primary),
            "arabic_visual_chunks": len(all_visual),
        }
        if removal:
            snapshot.update(
                {
                    "removed_document_sha256": content_sha256,
                    "removed_source": sources,
                    "added_native_chunks": 0,
                    "added_visual_chunks": 0,
                }
            )
        else:
            snapshot.update(
                {
                    "added_document_sha256": content_sha256,
                    "added_source": sources,
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


def _read_collection(source, *, batch_size: int = 500, exclude_sources: list[str] = ()) -> list[tuple]:
    """Return (id, document, metadata, embedding) rows, skipping the exclude_sources PDFs.

    Chroma 1.5 can lose the not-yet-flushed tail of a collection's HNSW index
    ("Error finding id" / "Nothing found on disk") while documents and metadata
    stay readable; those pages are re-embedded from their text.
    """
    excluded = {Path(name).name.casefold() for name in exclude_sources}
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
            if Path(str((metadata or {}).get("source", ""))).name.casefold() not in excluded
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
    source_filenames: list[str],
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
            _read_collection(client.get_collection(old_primary_name), exclude_sources=source_filenames)
            if has_old_primary
            else []
        )
        old_visual_rows = (
            _read_collection(client.get_collection(old_visual_name), exclude_sources=source_filenames)
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


_VERSION_DIR = re.compile(r"^\d{8}T\d{6}Z-")
_VERSIONED_COLLECTION = re.compile(r"^bct_(?:regulations|arabic_visual)_\w+$")


def prune_versions(asset_root: str | Path, *, keep: int = 3) -> dict[str, int]:
    """Delete asset versions older than the ``keep`` newest, and the Chroma collections only they used.

    Staged activation writes a full corpus copy per activation; without pruning every
    upload or enrichment batch adds one. Never touches the active version, the collections
    the process is serving (a version activated without its own local index keeps serving
    the previous one), hand-seeded folders, or unversioned collection names.
    """
    root = Path(asset_root).resolve()
    versions = root / "versions"
    if not versions.is_dir():
        return {"removed_versions": 0, "removed_collections": 0}
    active = resolve_active_assets(root)
    dated = sorted((p for p in versions.iterdir() if p.is_dir() and _VERSION_DIR.match(p.name)), key=lambda p: p.name)
    kept = set(dated[-keep:]) | {active}

    def snapshot(path: Path) -> dict:
        try:
            return json.loads((path / "snapshot.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    live = {os.environ.get("BCT_CHROMA_COLLECTION"), os.environ.get("BCT_OCR_CHROMA_COLLECTION")}
    for path in kept:
        live |= {snapshot(path).get("local_collection"), snapshot(path).get("local_visual_collection")}
    for path in reversed([p for p in dated if p.name <= active.name]):
        if snapshot(path).get("local_collection"):  # the index the active version falls back to
            live |= {snapshot(path)["local_collection"], snapshot(path).get("local_visual_collection")}
            break
    removed_versions = removed_collections = 0
    for path in dated:
        if path in kept:
            continue
        values = snapshot(path)
        doomed = {
            key: values.get(key)
            for key in ("local_collection", "local_visual_collection")
            if values.get(key) not in live and _VERSIONED_COLLECTION.match(str(values.get(key) or ""))
        }
        if doomed:
            discard_staged_local_collections({**doomed, "local_chroma_db": values.get("local_chroma_db")})
            removed_collections += len(doomed)
        shutil.rmtree(path, ignore_errors=True)
        removed_versions += 1
    return {"removed_versions": removed_versions, "removed_collections": removed_collections}


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
