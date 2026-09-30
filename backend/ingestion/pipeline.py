from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock
from langfuse import get_client, propagate_attributes

from .chunk import build_runtime_chunks
from .extract import ImageRegions, PdfExtractor, sha256_file
from .index import (
    activate_assets,
    discard_staged_local_collections,
    prune_versions,
    resolve_active_assets,
    stage_assets,
    stage_local_collections,
)
from .registry import SEARCHABLE, IngestionRegistry
from runtime_retrieval import _read_chunks
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
            value = os.environ.get("BCT_RUNTIME_ASSET_ROOT")
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


def ingest_session_id(content_sha256: str) -> str:
    """Langfuse session: the quick pass plus every enrichment trace of one PDF version."""
    return f"ingest-{content_sha256}"


def _step(name: str, **kwargs):
    return get_client().start_as_current_observation(name=name, **kwargs)


def _memory_mb() -> dict[str, float]:
    try:
        import psutil
    except ImportError:
        return {}
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    return {"rss_mb": round(info.rss / 2**20, 1), **({"peak_mb": round(peak / 2**20, 1)} if peak else {})}


def _extract_summary(structured) -> dict:
    degraded = [
        {"page": page.page_number, "visual_error": page.metadata.get("visual_error"), "flags": page.quality_flags}
        for page in structured.pages
        if page.metadata.get("visual_error") or "native_unusable_retained" in page.quality_flags
    ]
    return {
        "pages": len(structured.pages),
        "language": structured.language,
        "visual_pages": int(structured.metadata.get("visual_page_count", 0)),
        "visual_attempted": sum(1 for page in structured.pages if page.metadata.get("visual_attempted")),
        "image_region_pages": sum(1 for page in structured.pages if "image_regions" in page.quality_flags),
        "replaced_by_visual": sum(1 for page in structured.pages if page.extraction_method == "vlm"),
        "degraded_pages": degraded[:50],
        "degraded_count": len(degraded),
        "visual_pending": len(structured.metadata.get("visual_pending_pages") or []),
    }


def _traced(name: str, run, *, input: dict, tags: list[str]) -> dict:
    """One Langfuse trace per ingest/remove; a no-op client when LANGFUSE_* keys are unset."""
    langfuse = get_client()
    with langfuse.start_as_current_observation(name=name, input=input) as root, propagate_attributes(
        trace_name=name,
        tags=tags,
        metadata={"filename": str(input.get("filename", ""))[:200]},
    ):
        memory_before = _memory_mb()
        try:
            report = run()
        except Exception as error:
            root.update(
                level="ERROR",
                status_message=f"{type(error).__name__}: {error}"[:1000],
                output={"status": "failed", "error": f"{type(error).__name__}: {error}"},
                metadata={"memory_before": memory_before, "memory_after": _memory_mb()},
            )
            raise
        report = {**report, "langfuse_trace_id": langfuse.get_current_trace_id()}
        root.update(
            output={key: report.get(key) for key in (
                "status", "searchable", "enrichment", "duplicate", "pages", "visual_pages", "native_chunks_added",
                "visual_chunks_added", "local_indexed", "asset_version", "native_chunks_remaining",
            ) if key in report},
            metadata={"memory_before": memory_before, "memory_after": _memory_mb()},
        )
        return report


def _prune(root: Path) -> None:
    """Drop superseded asset versions; a cleanup failure never fails the committed activation."""
    try:
        with _step("prune-versions") as span:
            span.update(output=prune_versions(root))
    except Exception:
        logging.getLogger(__name__).warning("Asset version pruning failed.", exc_info=True)


def _restore_active_pointer(root: Path, previous: bytes | None) -> None:
    """Restore the runtime pointer if activation cannot be committed to the ledger."""
    pointer = root / "ACTIVE.json"
    if previous is None:
        pointer.unlink(missing_ok=True)
        return
    temporary = root / ".ACTIVE-rollback.tmp"
    temporary.write_bytes(previous)
    os.replace(temporary, pointer)


@dataclass(frozen=True)
class ActivatedVersion:
    """A new asset version that is now live (returned by _stage_and_activate)."""

    version: str  # folder name under versions/, e.g. 20260930T101500Z-ab12cd34ef-1a2b3c
    snapshot: dict  # the version's snapshot.json (chunk counts, parent version, ...)
    pointer: dict  # the new ACTIVE.json content
    local_index: dict  # local Chroma collection names for this version ({} when not built)
    supersession: dict  # what happened to supersession_edges.jsonl (counts or warning)


def _stage_supersession_edges(active_before: Path, staged_version: Path, filename: str, pages, asset_root: Path) -> dict:
    """Write the staged version's SUPERSEDES edges: the old edges minus this PDF's, plus the
    edges found in pages (none for a removal). Written before activation, so a failed
    activation also rolls the edges back."""
    from jsonl_supersession import load_prior_edges, merge_supersession_edges_for_ingest, write_edges

    report: dict[str, object] = {"enabled": True}
    try:
        report.update(merge_supersession_edges_for_ingest(
            active_before=active_before,
            staged_version=staged_version,
            filename=filename,
            pages=pages,
            asset_root=asset_root,
        ))
    except Exception as error:
        report["warning"] = f"{type(error).__name__}: {error}"
        # Never activate a version that silently drops the prior edge list.
        try:
            write_edges(staged_version / "supersession_edges.jsonl", load_prior_edges(active_before, asset_root=asset_root))
            report["fallback"] = "copied_prior_edges"
        except Exception as copy_error:
            report["fallback_error"] = f"{type(copy_error).__name__}: {copy_error}"
    get_client().create_event(
        name="merge-supersession-edges",
        output=report,
        **({"level": "WARNING", "status_message": str(report["warning"])} if "warning" in report else {}),
    )
    return report


class IngestionPipeline:
    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self.config.asset_root.mkdir(parents=True, exist_ok=True)
        self.config.documents_dir.mkdir(parents=True, exist_ok=True)
        self.registry = IngestionRegistry(self.config.registry_path)

    def close(self) -> None:
        self.registry.close()

    def ingest(self, pdf_path: str | Path, *, original_filename: str | None = None, metadata: dict | None = None) -> dict:
        from .local_visual import visual_backend_name

        profile = (os.environ.get("BCT_DEFAULT_PROFILE") or "local_hybrid").strip().casefold()
        trace_input = {
            "filename": original_filename or Path(pdf_path).name,
            "bytes": Path(pdf_path).stat().st_size if Path(pdf_path).exists() else None,
            "metadata": metadata or {},
            "profile": profile,
            "visual_backend": visual_backend_name(),
            "build_local_index": self.config.build_local,
        }
        return _traced(
            "ingest-document",
            lambda: self._ingest(pdf_path, original_filename=original_filename, metadata=metadata),
            input=trace_input,
            tags=["ingestion", profile, str((metadata or {}).get("doc_kind") or (metadata or {}).get("type") or "unspecified")],
        )

    def _ingest(self, pdf_path: str | Path, *, original_filename: str | None = None, metadata: dict | None = None) -> dict:
        source_path = Path(pdf_path).resolve(strict=True)
        filename = _safe_filename(original_filename or source_path.name)
        metadata = _clean_metadata(metadata)
        with _step("validate-pdf", input={"filename": filename}) as span:
            validate_pdf_file(source_path, max_bytes=self.config.max_pdf_bytes)
            content_hash = sha256_file(source_path)
            span.update(output={"content_sha256": content_hash})
        # Groups this quick pass with the background enrichment traces of the same PDF.
        with propagate_attributes(session_id=ingest_session_id(content_hash)):
            return self._ingest_hashed(source_path, content_hash, filename, metadata)

    def _ingest_hashed(self, source_path: Path, content_hash: str, filename: str, metadata: dict) -> dict:
        known = self.registry.get(content_hash)
        if known and known.get("status") in SEARCHABLE and known.get("report"):
            return {**known["report"], "duplicate": True}

        # One ingest at a time. Wait for the holder instead of failing parallel admin uploads in 5s.
        lock = FileLock(str(self.config.asset_root / ".ingestion.lock"), timeout=60 * 30)
        with _step("wait-ingestion-lock", input={"timeout_s": lock.timeout}):
            lock.acquire()
        try:
            # Check again after acquiring the cross-process lock.
            known = self.registry.get(content_hash)
            if known and known.get("status") in SEARCHABLE and known.get("report"):
                return {**known["report"], "duplicate": True}

            immutable_dir = self.config.documents_dir / content_hash
            immutable_dir.mkdir(parents=True, exist_ok=True)
            immutable_pdf = immutable_dir / filename
            if immutable_pdf.exists() and sha256_file(immutable_pdf) != content_hash:
                raise ValueError("Immutable document path already contains different bytes")
            if not immutable_pdf.exists():
                shutil.copy2(source_path, immutable_pdf)
            self.registry.start(content_hash, filename, str(immutable_pdf))

            return self._activate(content_hash, filename, immutable_pdf, metadata, quick=True)
        finally:
            lock.release()

    def _activate(self, content_hash: str, filename: str, immutable_pdf: Path, metadata: dict, *, quick: bool) -> dict:
        """Extract (visual results from the page ledger), chunk, then stage and activate. Caller holds the lock."""
        immutable_dir = immutable_pdf.parent
        try:
            with _step("extract-document", input={"pdf": filename, "quick": quick}) as span:
                visual_results, visual_model = self._visual_results(content_hash)
                structured = PdfExtractor(visual_model=visual_model).extract(
                    immutable_pdf, visual_results=visual_results, layout_cache=immutable_dir / "docling-layout.json"
                )
                summary = _extract_summary(structured)
                span.update(
                    output=summary,
                    **({"level": "WARNING", "status_message": f"{summary['degraded_count']} degraded pages"}
                       if summary["degraded_count"] else {}),
                )
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
            structured_path = immutable_dir / "structured.json"
            structured_path.write_text(
                json.dumps(structured.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self.registry.seed_pages(
                content_hash,
                [
                    {"page": page.page_number, **page.metadata["visual_plan"]}
                    for page in structured.pages
                    if "visual_pending" in page.quality_flags
                ],
            )
            progress = self.registry.progress(content_hash)
            status = "enriching" if progress["pending"] else "ready_degraded" if progress["failed"] else "ready"

            with _step("chunk-document") as span:
                try:
                    primary, visual = build_runtime_chunks(structured)
                except ValueError:
                    if not progress["pending"]:
                        raise
                    # Nothing readable natively (scanned PDF): searchable once enrichment reads it.
                    return self._commit_unindexed(content_hash, filename, immutable_pdf, metadata, structured, progress)
                span.update(output={"native_chunks": len(primary), "visual_chunks": len(visual)})

            def record(activated: ActivatedVersion) -> dict:
                report = {
                    "status": status,
                    "searchable": True,
                    "enrichment": progress,
                    "duplicate": False,
                    "filename": filename,
                    "title": metadata.get("title") or filename,
                    "administrator_metadata": metadata,
                    "content_sha256": content_hash,
                    "stored_pdf": str(immutable_pdf),
                    "structured_document": str(structured_path),
                    "language": structured.language,
                    "pages": len(structured.pages),
                    "visual_pages": int(structured.metadata.get("visual_page_count", 0)),
                    "native_chunks_added": len(primary),
                    "visual_chunks_added": len(visual),
                    "asset_version": activated.version,
                    "active_pointer": activated.pointer,
                    "local_indexed": bool(activated.local_index.get("local_collection")),
                    "supersession": activated.supersession,
                }
                self.registry.ready(
                    content_hash,
                    stored_path=str(immutable_pdf),
                    asset_version=activated.version,
                    report=report,
                    status=status,
                )
                return report

            return self._stage_and_activate(
                content_hash=content_hash,
                filename=filename,
                new_primary=primary,
                new_visual=visual,
                pages=structured.pages,
                record=record,
            )
        except Exception as error:
            if quick:
                # A failed first ingest is recorded; a failed enrichment batch keeps the
                # document live on its prior version.
                self.registry.fail(content_hash, f"{type(error).__name__}: {error}")
            raise

    def _stage_and_activate(
        self,
        *,
        content_hash: str,
        filename: str,
        new_primary: list,
        new_visual: list,
        pages: list,
        record,
        removal: bool = False,
    ) -> dict:
        """Build a new asset version, make it live, then call record(activated) to note it in
        the registry; returns what record returns.

        Shared by ingest (new chunks and the PDF's pages) and removal (none of either).
        Nothing is final until record() succeeds: on any error before that the previous
        version is made live again and the staged files are deleted, so the old corpus
        keeps serving (staged activation).
        """
        root = self.config.asset_root
        pointer_path = root / "ACTIVE.json"
        previous_pointer = pointer_path.read_bytes() if pointer_path.exists() else None
        staged_version: Path | None = None
        local_index: dict[str, object] = {}
        try:
            active_before = resolve_active_assets(root)
            base_snapshot = _read_snapshot(active_before)
            with _step("stage-assets", input={"active_before": active_before.name}) as span:
                staged_version, staged_snapshot = stage_assets(
                    asset_root=root,
                    new_primary=new_primary,
                    new_visual=new_visual,
                    content_sha256=content_hash,
                    source_filename=filename,
                    allow_empty=removal,
                    removal=removal,
                )
                span.update(output=staged_snapshot)
            version = staged_snapshot["version"]

            if self.config.build_local:
                with _step("stage-local-index", input={"version": version}) as span:
                    local_index = stage_local_collections(
                        asset_root=root,
                        version_id=version,
                        all_primary=_read_chunks(staged_version / "native.jsonl"),
                        all_visual=_read_chunks(staged_version / "arabic_ocr_secondary.jsonl"),
                        new_primary=new_primary,
                        new_visual=new_visual,
                        base_snapshot=base_snapshot,
                        source_filename=filename,
                    )
                    span.update(
                        output=local_index or {"skipped": True},
                        **({} if local_index else {"level": "WARNING", "status_message": "local Chroma index not updated"}),
                    )

            supersession = _stage_supersession_edges(active_before, staged_version, filename, pages, root)

            with _step("activate-version", input={"version": version}):
                pointer = activate_assets(root, staged_version, snapshot_updates=local_index)
            activated = ActivatedVersion(version, staged_snapshot, pointer, local_index, supersession)
            try:
                result = record(activated)
            except Exception:
                _restore_active_pointer(root, previous_pointer)
                raise
        except Exception:
            if staged_version is not None:
                # Cleanup is best-effort; the old active corpus remains authoritative either way.
                try:
                    _restore_active_pointer(root, previous_pointer)
                except Exception:
                    pass
                discard_staged_local_collections(local_index)
                shutil.rmtree(staged_version, ignore_errors=True)
            raise
        _prune(root)
        return result

    def _visual_results(self, content_hash: str) -> tuple[dict, str | None]:
        from .models import VisualPage

        results: dict = {}
        models: set[str] = set()
        for page, (state, payload, model) in self.registry.page_results(content_hash).items():
            if state == "done":
                data = json.loads(payload)
                results[page] = ImageRegions.model_validate(data) if "texts" in data else VisualPage.model_validate(data)
                if model:
                    models.add(model)
            else:
                results[page] = f"visual_failed: {payload}"
        return results, ", ".join(sorted(models)) or None

    def _commit_unindexed(self, content_hash, filename, immutable_pdf, metadata, structured, progress) -> dict:
        report = {
            "status": "enriching",
            "searchable": False,
            "enrichment": progress,
            "duplicate": False,
            "filename": filename,
            "title": metadata.get("title") or filename,
            "administrator_metadata": metadata,
            "content_sha256": content_hash,
            "stored_pdf": str(immutable_pdf),
            "language": structured.language,
            "pages": len(structured.pages),
            "native_chunks_added": 0,
            "visual_chunks_added": 0,
        }
        self.registry.ready(
            content_hash, stored_path=str(immutable_pdf), asset_version=None, report=report, status="enriching"
        )
        return report

    def enrich_activate(self, content_hash: str) -> dict:
        """Re-extract with every page read so far and make it live (background worker)."""
        lock = FileLock(str(self.config.asset_root / ".ingestion.lock"), timeout=60 * 30)
        with _step("wait-ingestion-lock", input={"timeout_s": lock.timeout}):
            lock.acquire()
        try:
            # Checked under the lock: the document may have been removed while we waited.
            known = self.registry.get(content_hash)
            if known is None or known.get("status") not in SEARCHABLE:
                raise KeyError(f"Document is not live: {content_hash}")
            report = known.get("report") or {}
            metadata = dict(report.get("administrator_metadata") or {})
            stored = Path(str(known.get("stored_path") or ""))
            if not stored.is_file():
                raise FileNotFoundError(f"Immutable PDF missing for {content_hash}")
            filename = str(known.get("original_filename") or stored.name)
            return self._activate(content_hash, filename, stored, metadata, quick=False)
        finally:
            lock.release()

    def remove(self, content_sha256: str) -> dict:
        return _traced(
            "remove-document",
            lambda: self._remove(content_sha256),
            input={"document_id": content_sha256},
            tags=["ingestion", "removal"],
        )

    def _remove(self, content_sha256: str) -> dict:
        """Drop a ready PDF from the active index; failed activation keeps the prior corpus."""
        content_hash = (content_sha256 or "").strip().lower()
        if not content_hash or len(content_hash) < 16:
            raise ValueError("Invalid document id")
        known = self.registry.get(content_hash)
        if known is None or known.get("status") not in SEARCHABLE:
            raise KeyError(f"Ready document not found: {content_hash}")
        filename = _safe_filename(str(known.get("original_filename") or "document.pdf"))

        # One ingest at a time. Wait for the holder instead of failing parallel admin uploads in 5s.
        lock = FileLock(str(self.config.asset_root / ".ingestion.lock"), timeout=60 * 30)
        with lock:
            known = self.registry.get(content_hash)
            if known is None or known.get("status") not in SEARCHABLE:
                raise KeyError(f"Ready document not found: {content_hash}")
            filename = _safe_filename(str(known.get("original_filename") or filename))

            def record(activated: ActivatedVersion) -> dict:
                self.registry.mark_removed(content_hash, asset_version=activated.version)
                return {
                    "status": "removed",
                    "document_id": content_hash,
                    "filename": filename,
                    "asset_version": activated.version,
                    "active_pointer": activated.pointer,
                    "native_chunks_remaining": activated.snapshot.get("native_chunks", 0),
                    "supersession": activated.supersession,
                }

            report = self._stage_and_activate(
                content_hash=content_hash,
                filename=filename,
                new_primary=[],
                new_visual=[],
                pages=[],
                record=record,
                removal=True,
            )

            stored = str(known.get("stored_path") or "").strip()
            if stored:
                # Immutable dir is documents_dir / sha256 / file.pdf
                immutable_dir = Path(stored).parent
                if immutable_dir.is_dir() and immutable_dir.parent == self.config.documents_dir:
                    shutil.rmtree(immutable_dir, ignore_errors=True)
            return report
