"""The index on disk, and how a new version of it goes live without ever breaking the old one.

The index folder (the "asset root") looks like this:
    ACTIVE.json              which version is live
    versions/<id>/           one complete copy of the corpus per version:
        native.jsonl             every chunk, one JSON line each
        arabic_ocr_secondary.jsonl   the Arabic text read from page images
        snapshot.json            counts, the Chroma collection names, what this version changed
        supersession_edges.jsonl  which circular replaced which
    local_chroma/            the Chroma database (one pair of collections per version)

Adding or removing a PDF never edits the live version. stage_assets() builds a full new version
next to it, stage_local_collections() builds its Chroma collections, and only when all of that
succeeded does activate_assets() switch ACTIVE.json to it ("staged activation"). A failure at any
step leaves the old version live.
"""
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

from rag.bm25 import searchable_text
from rag.runtime_retrieval import _read_chunks

logger = logging.getLogger(__name__)


def resolve_active_assets(root: str | Path) -> Path:
    """The live version's folder, read from ACTIVE.json (the root itself when there is none).
    The pointer must stay inside the root: a ".." or absolute path is refused."""
    root = Path(root).resolve()
    pointer = root / "ACTIVE.json"
    if not pointer.exists():
        return root
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    relative = Path(str(payload["active_version"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid runtime asset pointer")
    active = (root / relative).resolve()
    active.relative_to(root)  # raises ValueError if the resolved path left the root
    return active


def configure_runtime_assets(asset_root: str | Path, *, validate: bool = False) -> Path:
    """Tell the rest of the app where the live index is, through environment variables (the
    chunk files, the Chroma folder and collection names). Called at startup and after each new
    version goes live. Returns the live version's folder."""
    asset_root = Path(asset_root).resolve()
    active = resolve_active_assets(asset_root)
    if validate:
        for filename in ("native.jsonl", "arabic_ocr_secondary.jsonl", "snapshot.json"):
            if not (active / filename).is_file():
                raise ValueError(f"Missing runtime asset: {active / filename}")
    snapshot = json.loads((active / "snapshot.json").read_text(encoding="utf-8"))
    # Very old snapshots were a list; today's are a dict.
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
    """The Chroma folder of this index. The path saved in the snapshot may be from another machine
    (the index is built on a laptop, then copied into Docker), so the usual places inside the index
    folder are tried too."""
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
    """Write chunks to a JSONL file: one {"page_content", "metadata"} object per line."""
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
    """The chunks of one file of the live version (none if the file does not exist)."""
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
        """The live chunks minus those of the PDFs being replaced, plus the new ones."""
        old = _read_active(active, filename)
        return [old[index] for index in _keep_indices(old, source_keys)] + list(new_docs)

    all_primary = merged("native.jsonl", new_primary)
    all_visual = merged("arabic_ocr_secondary.jsonl", new_visual)
    if not all_primary and not allow_empty:
        raise ValueError(
            f"Ingestion of {sources!r} produced no native searchable chunks "
            "(and the active corpus has none either)"
        )
    # The Arabic image-text file may be empty (no Arabic scans yet): search then skips it.

    # A version id sorts by time: 20261009T212850Z-<hash start>-<random>.
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
        os.replace(staging, final)  # the folder appears under its final name only once complete
        return final, snapshot
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def activate_assets(asset_root: str | Path, version_dir: str | Path, *, snapshot_updates: dict | None = None) -> dict:
    """Make a staged version live: write ACTIVE.json pointing at it. The pointer is written to a
    temporary file and renamed over the old one, so readers see the old or the new, never half."""
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
    """Metadata Chroma accepts: only text, numbers and booleans, so lists become JSON text."""
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
    """The chunk's ID in Chroma: its chunk_id, or one computed from it for very old chunks."""
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
    """Does Chroma have a collection with this name?"""
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def _read_collection(source, *, batch_size: int = 500, exclude_sources: list[str] = ()) -> list[tuple]:
    """Return (id, document, metadata, embedding) rows, skipping the exclude_sources PDFs.

    The stored vectors are reused, so an upload only computes vectors for its new chunks. Chroma
    sometimes loses the vectors of its last writes while their text stays readable; those chunks
    get their vectors computed again from their text.
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
            logger.warning("Chroma embeddings unreadable at offset %s of %s; re-embedding (%s: %s).",
                           offset, source.name, type(error).__name__, error)
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
            embeddings = dict(zip(kept, _embed_local([searchable_text(documents[i], metadatas[i]) for i in kept])))
        rows.extend((ids[i], documents[i], metadatas[i], embeddings[i]) for i in kept)
        offset += len(ids)
        if not ids:
            break
    return rows


def _embed_local(texts: list[str]) -> list:
    """Vectors for these texts from the local embedding model (e5)."""
    from rag.embedding import create_embedding_model

    return create_embedding_model().embed_documents(texts)


def _add_rows(target, rows: list[tuple], *, batch_size: int = 500) -> None:
    """Copy (id, text, metadata, vector) rows into a collection, 500 at a time."""
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
    """Build the new version's two Chroma collections (main chunks, Arabic image text): copy the
    live collections' rows except the replaced PDFs, then embed and add the new chunks. Returns
    the names to record in the snapshot ({} when there is no live collection to build on)."""
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError("Local ingestion requires requirements-local.txt") from error

    root = Path(asset_root).resolve()
    db_path = Path(os.environ.get("BCT_CHROMA_DB", str(root / "local_chroma"))).resolve()
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    short = version_id.replace("-", "_")[-24:]  # collection names have a length limit
    primary_name = f"bct_regulations_{short}"
    visual_name = f"bct_arabic_visual_{short}"
    old_primary_name = str(base_snapshot.get("local_collection") or os.environ.get("BCT_CHROMA_COLLECTION", "bct_regulations"))
    old_visual_name = str(base_snapshot.get("local_visual_collection") or os.environ.get("BCT_OCR_CHROMA_COLLECTION", "bct_arabic_ocr_secondary_v1"))

    has_old_primary = _collection_exists(client, old_primary_name)
    has_old_visual = _collection_exists(client, old_visual_name)
    # ponytail: with no live collection, an upload does not embed the whole corpus (that takes
    # hours); rebuild the local index with the ingest command instead.
    if not has_old_primary and len(all_primary) > len(new_primary):
        logger.warning("No prior Chroma collection %r in %s; new chunks are not searchable locally.",
                       old_primary_name, db_path)
        return {}

    # Read old rows before any write: with the serving retriever holding these collections
    # open, Chroma 1.5 fails reads ("Error finding id") once the same process starts writing.
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
    primary_target = client.create_collection(primary_name)
    visual_target = client.create_collection(visual_name)
    try:
        _add_rows(primary_target, old_primary_rows)
        _add_rows(visual_target, old_visual_rows)
        primary_to_add = new_primary if has_old_primary else all_primary
        visual_to_add = new_visual if has_old_visual else all_visual

        def add(collection, documents):
            """Embed these chunks (with their context header) and store them."""
            if not documents:
                return
            texts = [doc.page_content for doc in documents]
            vectors = _embed_local([searchable_text(doc.page_content, doc.metadata) for doc in documents])
            collection.upsert(
                ids=[_document_chunk_id(doc) for doc in documents],
                documents=texts,
                metadatas=[_chroma_metadata(doc.metadata) for doc in documents],
                embeddings=vectors,
            )

        add(primary_target, primary_to_add)
        add(visual_target, visual_to_add)
        # The collections must hold exactly the version's chunks, or the version is not used.
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


_VERSION_DIR = re.compile(r"^\d{8}T\d{6}Z-")  # a version folder name starts with its time
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
        """A version's snapshot.json ({} if unreadable)."""
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
