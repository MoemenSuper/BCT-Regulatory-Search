from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock

from .chunk import build_runtime_chunks
from .extract import PdfExtractor, sha256_file
from .gemini_visual import GeminiVisualTranscriber
from .index import (
    activate_assets,
    discard_staged_local_collections,
    resolve_active_assets,
    stage_cloud_assets,
    stage_local_collections,
)
from .registry import IngestionRegistry
from source_metadata import safe_pdf_filename


@dataclass(frozen=True)
class IngestionConfig:
    asset_root: Path
    documents_dir: Path
    registry_path: Path
    gemini_cache_dir: Path
    max_pdf_bytes: int = 50 * 1024 * 1024
    build_local: bool = True

    @classmethod
    def from_environment(cls, *, asset_root: str | Path | None = None) -> "IngestionConfig":
        if asset_root is None:
            value = os.environ.get("BCT_RUNTIME_ASSET_ROOT") or os.environ.get("BCT_VOYAGE_PROVIDER_ROOT")
            if not value:
                raise RuntimeError("BCT_RUNTIME_ASSET_ROOT or --assets is required for ingestion")
            asset_root = value
        root = Path(asset_root).resolve()
        data_root = Path(os.environ.get("BCT_INGESTION_DATA_DIR", str(root.parent / "ingestion-data"))).resolve()
        return cls(
            asset_root=root,
            documents_dir=Path(os.environ.get("BCT_INGESTED_DOCUMENTS_DIR", str(data_root / "documents"))).resolve(),
            registry_path=Path(os.environ.get("BCT_INGESTION_DB", str(data_root / "ingestion.sqlite3"))).resolve(),
            gemini_cache_dir=Path(os.environ.get("BCT_GEMINI_CACHE", str(data_root / "gemini-cache"))).resolve(),
            max_pdf_bytes=int(os.environ.get("BCT_MAX_PDF_BYTES", str(50 * 1024 * 1024))),
            build_local=os.environ.get("BCT_INGEST_LOCAL_INDEX", "1") == "1",
        )


def _safe_filename(name: str) -> str:
    return safe_pdf_filename(name, ensure_pdf_suffix=True, max_length=200)


def _clean_metadata(metadata: dict | None) -> dict[str, str]:
    limits = {
        "title": 300,
        "publication_date": 64,
        "type": 120,
        "category": 120,
        "document_number": 120,
        "doc_kind": 32,
        "related_to": 300,
    }
    cleaned: dict[str, str] = {}
    for key, limit in limits.items():
        raw = (metadata or {}).get(key)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value:
            continue
        if len(value) > limit or any(ord(character) < 32 and character not in "\t\n" for character in value):
            raise ValueError(f"Invalid administrator metadata field: {key}")
        cleaned[key] = value
    if "doc_kind" in cleaned:
        from document_authority import normalize_doc_kind

        cleaned["doc_kind"] = normalize_doc_kind(cleaned["doc_kind"])
    return cleaned


def _persist_page_images(structured, immutable_dir: Path) -> int:
    """Write chart-page PNGs beside the immutable PDF; drop in-memory bytes."""
    images_dir = immutable_dir / "page-images"
    written = 0
    for page in structured.pages:
        png = page.metadata.pop("page_image_png", None)
        if not png:
            continue
        images_dir.mkdir(parents=True, exist_ok=True)
        path = images_dir / f"page-{page.page_number}.png"
        path.write_bytes(png)
        page.metadata["page_image_path"] = str(path)
        page.metadata["has_chart"] = True
        written += 1
    return written


def _ensure_secondary_page_images(structured, pdf_path: Path, immutable_dir: Path) -> int:
    """Persist page PNGs for statistical/internal when chart heuristics did not already."""
    if str(structured.metadata.get("doc_kind") or "") not in {"statistical", "internal"}:
        return 0
    try:
        import pymupdf
    except ImportError:
        return 0
    images_dir = immutable_dir / "page-images"
    written = 0
    with pymupdf.open(pdf_path) as pdf:
        for page in structured.pages:
            if page.metadata.get("page_image_path"):
                continue
            images_dir.mkdir(parents=True, exist_ok=True)
            pixmap = pdf.load_page(page.page_number - 1).get_pixmap(
                matrix=pymupdf.Matrix(2.0, 2.0), alpha=False
            )
            path = images_dir / f"page-{page.page_number}.png"
            path.write_bytes(pixmap.tobytes("png"))
            page.metadata["page_image_path"] = str(path)
            written += 1
    return written


def validate_pdf_file(path: str | Path, *, max_bytes: int) -> None:
    path = Path(path)
    size = path.stat().st_size
    if size < 5 or size > max_bytes:
        raise ValueError(f"PDF size must be between 5 bytes and {max_bytes} bytes")
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError("Uploaded file does not have a PDF signature")
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError("PDF validation requires PyMuPDF") from error
    try:
        with pymupdf.open(path) as pdf:
            if pdf.page_count < 1:
                raise ValueError("PDF contains no pages")
            # Force a basic parse before copying the file into immutable storage.
            pdf.load_page(0).rect
    except Exception as error:
        raise ValueError(f"PDF cannot be parsed safely: {error}") from error


def _read_snapshot(active: Path) -> dict:
    path = active / "snapshot.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Legacy Codex/provider snapshots are a list of representation bindings;
    # versioned ingestion snapshots are dicts with local_collection keys.
    return payload if isinstance(payload, dict) else {}


def _restore_active_pointer(root: Path, previous: bytes | None) -> None:
    """Restore the runtime pointer if activation cannot be committed to the ledger."""
    pointer = root / "ACTIVE.json"
    if previous is None:
        pointer.unlink(missing_ok=True)
        return
    temporary = root / ".ACTIVE-rollback.tmp"
    temporary.write_bytes(previous)
    os.replace(temporary, pointer)


class IngestionPipeline:
    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self.config.asset_root.mkdir(parents=True, exist_ok=True)
        self.config.documents_dir.mkdir(parents=True, exist_ok=True)
        self.registry = IngestionRegistry(self.config.registry_path)

    def close(self) -> None:
        self.registry.close()

    def ingest(self, pdf_path: str | Path, *, original_filename: str | None = None, metadata: dict | None = None) -> dict:
        source_path = Path(pdf_path).resolve(strict=True)
        filename = _safe_filename(original_filename or source_path.name)
        metadata = _clean_metadata(metadata)
        validate_pdf_file(source_path, max_bytes=self.config.max_pdf_bytes)
        content_hash = sha256_file(source_path)
        known = self.registry.get(content_hash)
        if known and known.get("status") == "ready" and known.get("report"):
            return {**known["report"], "duplicate": True}

        lock = FileLock(str(self.config.asset_root / ".ingestion.lock"), timeout=5)
        with lock:
            # Check again after acquiring the cross-process lock.
            known = self.registry.get(content_hash)
            if known and known.get("status") == "ready" and known.get("report"):
                return {**known["report"], "duplicate": True}

            immutable_dir = self.config.documents_dir / content_hash
            immutable_dir.mkdir(parents=True, exist_ok=True)
            immutable_pdf = immutable_dir / filename
            if immutable_pdf.exists() and sha256_file(immutable_pdf) != content_hash:
                raise ValueError("Immutable document path already contains different bytes")
            if not immutable_pdf.exists():
                shutil.copy2(source_path, immutable_pdf)
            self.registry.start(content_hash, filename, str(immutable_pdf))

            staged_version: Path | None = None
            pointer_path = self.config.asset_root / "ACTIVE.json"
            previous_pointer = pointer_path.read_bytes() if pointer_path.exists() else None
            activation_committed = False
            try:
                transcriber = None
                if os.environ.get("BCT_GEMINI_VISUAL", "1") == "1":
                    transcriber = GeminiVisualTranscriber(self.config.gemini_cache_dir)
                structured = PdfExtractor(visual_transcriber=transcriber).extract(immutable_pdf)
                from document_authority import authority_for_kind, resolve_doc_kind

                doc_kind = resolve_doc_kind(
                    explicit=metadata.get("doc_kind") or metadata.get("type"),
                    filename=filename,
                )
                metadata["doc_kind"] = doc_kind
                metadata["authority"] = authority_for_kind(doc_kind)
                structured.document_number = metadata.get("document_number")
                structured.publication_date = metadata.get("publication_date")
                structured.metadata["administrator_metadata"] = metadata
                structured.metadata["doc_kind"] = doc_kind
                structured.metadata["authority"] = metadata["authority"]
                if metadata.get("related_to"):
                    structured.metadata["related_to"] = metadata["related_to"]
                page_images = _persist_page_images(structured, immutable_dir)
                page_images += _ensure_secondary_page_images(structured, immutable_pdf, immutable_dir)
                structured.metadata["chart_page_images"] = page_images
                structured_path = immutable_dir / "structured.json"
                structured_path.write_text(
                    json.dumps(structured.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

                primary, visual = build_runtime_chunks(structured)
                active_before = resolve_active_assets(self.config.asset_root)
                base_snapshot = _read_snapshot(active_before)
                staged_version, staged_snapshot = stage_cloud_assets(
                    asset_root=self.config.asset_root,
                    new_primary=primary,
                    new_visual=visual,
                    content_sha256=content_hash,
                    source_filename=filename,
                )

                snapshot_updates: dict[str, object] = {}
                if self.config.build_local:
                    from runtime_retrieval import _read_chunks

                    all_primary = _read_chunks(staged_version / "native.jsonl")
                    all_visual = _read_chunks(staged_version / "arabic_ocr_secondary.jsonl")
                    local = stage_local_collections(
                        asset_root=self.config.asset_root,
                        version_id=staged_snapshot["version"],
                        all_primary=all_primary,
                        all_visual=all_visual,
                        new_primary=primary,
                        new_visual=visual,
                        base_snapshot=base_snapshot,
                        source_filename=filename,
                    )
                    snapshot_updates.update(local)

                # Flat SUPERSEDES edges for topical/force currentness. Written into
                # the staged version before activate so a failed activation rolls back.
                supersession_report: dict[str, object] = {"enabled": True}
                try:
                    from jsonl_supersession import merge_supersession_edges_for_ingest

                    supersession_report.update(
                        merge_supersession_edges_for_ingest(
                            active_before=active_before,
                            staged_version=staged_version,
                            filename=filename,
                            pages=structured.pages,
                            asset_root=self.config.asset_root,
                        )
                    )
                except Exception as supersession_error:
                    supersession_report["warning"] = (
                        f"{type(supersession_error).__name__}: {supersession_error}"
                    )
                    # Never activate a version that silently drops the prior edge list.
                    try:
                        from jsonl_supersession import (
                            load_prior_edges,
                            write_edges,
                        )

                        write_edges(
                            staged_version / "supersession_edges.jsonl",
                            load_prior_edges(
                                active_before, asset_root=self.config.asset_root
                            ),
                        )
                        supersession_report["fallback"] = "copied_prior_edges"
                    except Exception as copy_error:
                        supersession_report["fallback_error"] = (
                            f"{type(copy_error).__name__}: {copy_error}"
                        )

                pointer = activate_assets(
                    self.config.asset_root,
                    staged_version,
                    snapshot_updates=snapshot_updates,
                )

                report = {
                    "status": "ready",
                    "duplicate": False,
                    "filename": filename,
                    "title": metadata.get("title") or filename,
                    "administrator_metadata": metadata,
                    "content_sha256": content_hash,
                    "stored_pdf": str(immutable_pdf),
                    "structured_document": str(structured_path),
                    "language": structured.language,
                    "pages": len(structured.pages),
                    "gemini_visual_pages": int(structured.metadata.get("visual_page_count", 0)),
                    "native_chunks_added": len(primary),
                    "visual_chunks_added": len(visual),
                    "asset_version": staged_snapshot["version"],
                    "active_pointer": pointer,
                    "local_indexed": bool(snapshot_updates.get("local_collection")),
                    "supersession": supersession_report,
                }
                try:
                    self.registry.ready(
                        content_hash,
                        stored_path=str(immutable_pdf),
                        asset_version=staged_snapshot["version"],
                        report=report,
                    )
                except Exception:
                    _restore_active_pointer(self.config.asset_root, previous_pointer)
                    raise
                activation_committed = True
                return report
            except Exception as error:
                if staged_version is not None and not activation_committed:
                    # A pre-commit failure must not leave the new version selected
                    # or orphan versioned local collections that can never become
                    # active. Cleanup is best-effort; the old active corpus remains
                    # authoritative either way.
                    try:
                        _restore_active_pointer(self.config.asset_root, previous_pointer)
                    except Exception:
                        pass
                    discard_staged_local_collections(snapshot_updates)
                    shutil.rmtree(staged_version, ignore_errors=True)
                if not activation_committed:
                    self.registry.fail(content_hash, f"{type(error).__name__}: {error}")
                raise

    def remove(self, content_sha256: str) -> dict:
        """Drop a ready PDF from the active index; failed activation keeps the prior corpus."""
        content_hash = (content_sha256 or "").strip().lower()
        if not content_hash or len(content_hash) < 16:
            raise ValueError("Invalid document id")
        known = self.registry.get(content_hash)
        if known is None or known.get("status") != "ready":
            raise KeyError(f"Ready document not found: {content_hash}")
        filename = _safe_filename(str(known.get("original_filename") or "document.pdf"))

        lock = FileLock(str(self.config.asset_root / ".ingestion.lock"), timeout=5)
        with lock:
            known = self.registry.get(content_hash)
            if known is None or known.get("status") != "ready":
                raise KeyError(f"Ready document not found: {content_hash}")
            filename = _safe_filename(str(known.get("original_filename") or filename))

            staged_version: Path | None = None
            pointer_path = self.config.asset_root / "ACTIVE.json"
            previous_pointer = pointer_path.read_bytes() if pointer_path.exists() else None
            activation_committed = False
            snapshot_updates: dict[str, object] = {}
            try:
                active_before = resolve_active_assets(self.config.asset_root)
                base_snapshot = _read_snapshot(active_before)
                staged_version, staged_snapshot = stage_cloud_assets(
                    asset_root=self.config.asset_root,
                    new_primary=[],
                    new_visual=[],
                    content_sha256=content_hash,
                    source_filename=filename,
                    allow_empty=True,
                    removal=True,
                )

                if self.config.build_local:
                    from runtime_retrieval import _read_chunks

                    all_primary = _read_chunks(staged_version / "native.jsonl")
                    all_visual = _read_chunks(staged_version / "arabic_ocr_secondary.jsonl")
                    local = stage_local_collections(
                        asset_root=self.config.asset_root,
                        version_id=staged_snapshot["version"],
                        all_primary=all_primary,
                        all_visual=all_visual,
                        new_primary=[],
                        new_visual=[],
                        base_snapshot=base_snapshot,
                        source_filename=filename,
                    )
                    snapshot_updates.update(local)

                supersession_report: dict[str, object] = {"enabled": True}
                try:
                    from jsonl_supersession import (
                        load_prior_edges,
                        write_edges,
                    )

                    prior = load_prior_edges(active_before, asset_root=self.config.asset_root)
                    basename = Path(filename).name.casefold()
                    kept = [
                        edge
                        for edge in prior
                        if Path(edge.source_file).name.casefold() != basename
                    ]
                    write_edges(staged_version / "supersession_edges.jsonl", kept)
                    supersession_report.update(
                        {"prior": len(prior), "kept": len(kept), "from_pdf": 0, "total": len(kept)}
                    )
                except Exception as supersession_error:
                    supersession_report["warning"] = (
                        f"{type(supersession_error).__name__}: {supersession_error}"
                    )
                    try:
                        from jsonl_supersession import load_prior_edges, write_edges

                        write_edges(
                            staged_version / "supersession_edges.jsonl",
                            load_prior_edges(active_before, asset_root=self.config.asset_root),
                        )
                        supersession_report["fallback"] = "copied_prior_edges"
                    except Exception as copy_error:
                        supersession_report["fallback_error"] = (
                            f"{type(copy_error).__name__}: {copy_error}"
                        )

                pointer = activate_assets(
                    self.config.asset_root,
                    staged_version,
                    snapshot_updates=snapshot_updates,
                )
                try:
                    self.registry.mark_removed(
                        content_hash, asset_version=staged_snapshot["version"]
                    )
                except Exception:
                    _restore_active_pointer(self.config.asset_root, previous_pointer)
                    raise
                activation_committed = True

                stored = str(known.get("stored_path") or "").strip()
                if stored:
                    stored_path = Path(stored)
                    # Immutable dir is documents_dir / sha256 / file.pdf
                    immutable_dir = stored_path.parent
                    if immutable_dir.is_dir() and immutable_dir.parent == self.config.documents_dir:
                        shutil.rmtree(immutable_dir, ignore_errors=True)

                return {
                    "status": "removed",
                    "document_id": content_hash,
                    "filename": filename,
                    "asset_version": staged_snapshot["version"],
                    "active_pointer": pointer,
                    "native_chunks_remaining": staged_snapshot.get("native_chunks", 0),
                    "supersession": supersession_report,
                }
            except Exception:
                if staged_version is not None and not activation_committed:
                    try:
                        _restore_active_pointer(self.config.asset_root, previous_pointer)
                    except Exception:
                        pass
                    discard_staged_local_collections(snapshot_updates)
                    shutil.rmtree(staged_version, ignore_errors=True)
                raise
