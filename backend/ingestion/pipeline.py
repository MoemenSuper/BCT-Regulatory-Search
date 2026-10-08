from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock

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
from .docling_layout import NotEnoughMemory
from .registry import SEARCHABLE, IngestionRegistry
from runtime_retrieval import _read_chunks
from source_metadata import safe_pdf_filename


logger = logging.getLogger(__name__)
# A CLI ingest takes everything queued along with its own PDF, in batches of at most this.
_MAX_BATCH = 25


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


def _logged(name: str, run, *, subject: str) -> dict:
    """Run an ingest or a removal and leave one log line: how it ended, or the error and memory use."""
    memory_before = _memory_mb()
    try:
        report = run()
    except Exception:
        logger.exception("%s failed for %s (memory before %s, after %s)", name, subject, memory_before, _memory_mb())
        raise
    logger.info("%s %s: status=%s", name, subject, report.get("status"))
    return report


def _prune(root: Path) -> None:
    """Drop superseded asset versions; a cleanup failure never fails the committed activation."""
    try:
        prune_versions(root)
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


@dataclass(frozen=True)
class PdfChange:
    """One PDF's part of a new asset version: its new chunks and pages (none for a removal)."""

    content_hash: str
    filename: str
    primary: list
    visual: list
    pages: list


@dataclass(frozen=True)
class _Prepared:
    """A PDF read and chunked, waiting to go live with its batch."""

    content_hash: str
    filename: str
    immutable_pdf: Path
    metadata: dict
    structured: object
    structured_path: Path
    primary: list
    visual: list
    progress: dict


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
    if "warning" in report:
        logger.warning("Relations between texts not updated for %s: %s", filename, report)
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
        return _logged(
            "ingest-document",
            lambda: self._ingest(pdf_path, original_filename=original_filename, metadata=metadata),
            subject=original_filename or Path(pdf_path).name,
        )

    def _ingest(self, pdf_path: str | Path, *, original_filename: str | None = None, metadata: dict | None = None) -> dict:
        """Index one PDF now (CLI, tests): queue it, then run the queue."""
        queued = self.queue(pdf_path, original_filename=original_filename, metadata=metadata)
        if queued.get("duplicate"):
            return queued
        reports, errors = self._ingest_batch(self.registry.next_queued(_MAX_BATCH))
        if queued["content_sha256"] in errors:
            raise errors[queued["content_sha256"]]
        return reports[queued["content_sha256"]]

    def queue(self, pdf_path: str | Path, *, original_filename: str | None = None, metadata: dict | None = None) -> dict:
        """Keep an uploaded PDF and queue it for indexing; returns at once.

        The background worker indexes queued PDFs in batches (ingest_queued). A PDF already
        searchable with the same bytes is a duplicate; one already queued stays queued.
        """
        source_path = Path(pdf_path).resolve(strict=True)
        filename = _safe_filename(original_filename or source_path.name)
        metadata = _clean_metadata(metadata)
        validate_pdf_file(source_path, max_bytes=self.config.max_pdf_bytes)
        content_hash = sha256_file(source_path)
        known = self.registry.get(content_hash)
        if known and known.get("status") in SEARCHABLE and known.get("report"):
            return {**known["report"], "duplicate": True}
        if known and known.get("status") in ("queued", "processing"):
            return {"status": known["status"], "duplicate": False, "filename": filename, "content_sha256": content_hash}
        immutable_dir = self.config.documents_dir / content_hash
        immutable_dir.mkdir(parents=True, exist_ok=True)
        immutable_pdf = immutable_dir / filename
        if immutable_pdf.exists() and sha256_file(immutable_pdf) != content_hash:
            raise ValueError("Immutable document path already contains different bytes")
        if not immutable_pdf.exists():
            shutil.copy2(source_path, immutable_pdf)
        self.registry.queue(content_hash, filename, str(immutable_pdf), metadata)
        return {"status": "queued", "duplicate": False, "filename": filename, "content_sha256": content_hash}

    def ingest_queued(self, limit: int = _MAX_BATCH, *, before_each=None) -> dict[str, dict]:
        """Index up to `limit` queued PDFs with ONE new asset version (background worker).

        Each PDF is read on its own: one that fails is marked failed with its reason and the
        others go on. Then the whole batch is staged and activated once, so a PDF costs its own
        reading, not a copy of the whole index. before_each(job) runs before a PDF is read (the
        worker waits there for chat). NotEnoughMemory leaves that PDF and the rest queued.
        Returns content_sha256 -> report.
        """
        reports, _errors = self._ingest_batch(self.registry.next_queued(limit), before_each=before_each)
        return reports

    def _ingest_batch(self, jobs: list[dict], *, before_each=None) -> tuple[dict[str, dict], dict[str, Exception]]:
        lock = FileLock(str(self.config.asset_root / ".ingestion.lock"), timeout=60 * 30)
        lock.acquire()
        reports: dict[str, dict] = {}
        errors: dict[str, Exception] = {}
        prepared: list[_Prepared] = []
        try:
            names: set[str] = set()
            for job in jobs:
                content_hash = str(job["content_sha256"])
                filename = str(job["original_filename"])
                if filename.casefold() in names:
                    continue  # two versions of one PDF: the newer one goes in the next batch
                metadata = dict((job.get("report") or {}).get("administrator_metadata") or {})
                stored = Path(str(job["stored_path"]))
                started = time.monotonic()
                try:
                    if before_each is not None:
                        before_each(job)
                    self.registry.start(content_hash, filename, str(stored))
                    prepared.append(self._prepare(content_hash, filename, stored, metadata))
                except NotEnoughMemory as error:
                    self.registry.queue(content_hash, filename, str(stored), metadata)
                    logger.warning("Not enough memory to read %s; it stays queued: %s", filename, error)
                    break
                except Exception as error:
                    errors[content_hash] = error
                    message = f"{type(error).__name__}: {error}"
                    self.registry.fail(content_hash, message)
                    reports[content_hash] = {"status": "failed", "filename": filename, "error": message,
                                             "content_sha256": content_hash, "duplicate": False}
                    logger.warning("Indexing failed for %s: %s", filename, message)
                    continue
                names.add(filename.casefold())
                logger.info("Read %s in %.1fs", filename, time.monotonic() - started)
            if prepared:
                from . import docling_layout

                docling_layout.release()  # give its memory back before the index is rebuilt
                try:
                    reports.update(self._activate_prepared(prepared))
                except Exception as error:
                    for item in prepared:
                        errors[item.content_hash] = error
                        self.registry.fail(item.content_hash, f"Activation failed: {type(error).__name__}: {error}")
                    logger.exception("Activating a batch of %d PDF(s) failed; the previous index stays live.", len(prepared))
                    raise
                logger.info("Indexed a batch of %d PDF(s)", len(prepared))
            return reports, errors
        finally:
            lock.release()

    def _prepare(self, content_hash: str, filename: str, immutable_pdf: Path, metadata: dict) -> "_Prepared":
        """Extract one PDF (visual results from the page ledger) and chunk it."""
        immutable_dir = immutable_pdf.parent
        visual_results, visual_model = self._visual_results(content_hash)
        structured = PdfExtractor(visual_model=visual_model).extract(
            immutable_pdf, visual_results=visual_results, layout_cache=immutable_dir / "docling-layout.json"
        )
        summary = _extract_summary(structured)
        if summary["degraded_count"]:
            logger.warning("%s: %d page(s) could not be read well: %s", filename, summary["degraded_count"],
                           summary["degraded_pages"][:5])
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
        try:
            primary, visual = build_runtime_chunks(structured)
        except ValueError:
            if not progress["pending"]:
                raise
            # Nothing readable natively (scanned PDF): searchable once enrichment reads it.
            primary, visual = [], []
        return _Prepared(content_hash, filename, immutable_pdf, metadata, structured, structured_path,
                         primary, visual, progress)

    def _activate_prepared(self, prepared: list["_Prepared"]) -> dict[str, dict]:
        """Make prepared PDFs live with one new asset version; returns content_sha256 -> report."""
        reports = {
            item.content_hash: self._commit_unindexed(item.content_hash, item.filename, item.immutable_pdf,
                                                      item.metadata, item.structured, item.progress)
            for item in prepared
            if not item.primary
        }
        indexed = [item for item in prepared if item.primary]
        if not indexed:
            return reports

        def record(activated: ActivatedVersion) -> dict:
            done = {}
            for item in indexed:
                progress = item.progress
                status = "enriching" if progress["pending"] else "ready_degraded" if progress["failed"] else "ready"
                report = {
                    "status": status,
                    "searchable": True,
                    "enrichment": progress,
                    "duplicate": False,
                    "filename": item.filename,
                    "title": item.metadata.get("title") or item.filename,
                    "administrator_metadata": item.metadata,
                    "content_sha256": item.content_hash,
                    "stored_pdf": str(item.immutable_pdf),
                    "structured_document": str(item.structured_path),
                    "language": item.structured.language,
                    "pages": len(item.structured.pages),
                    "visual_pages": int(item.structured.metadata.get("visual_page_count", 0)),
                    "native_chunks_added": len(item.primary),
                    "visual_chunks_added": len(item.visual),
                    "asset_version": activated.version,
                    "active_pointer": activated.pointer,
                    "local_indexed": bool(activated.local_index.get("local_collection")),
                    "supersession": activated.supersession,
                }
                self.registry.ready(
                    item.content_hash,
                    stored_path=str(item.immutable_pdf),
                    asset_version=activated.version,
                    report=report,
                    status=status,
                )
                done[item.content_hash] = report
            return done

        reports.update(self._stage_and_activate(
            changes=[PdfChange(item.content_hash, item.filename, item.primary, item.visual, item.structured.pages)
                     for item in indexed],
            record=record,
        ))
        return reports

    def _activate(self, content_hash: str, filename: str, immutable_pdf: Path, metadata: dict) -> dict:
        """Re-extract one live PDF with the pages read so far and make it live (enrichment). Caller holds the lock."""
        return self._activate_prepared([self._prepare(content_hash, filename, immutable_pdf, metadata)])[content_hash]

    def _stage_and_activate(self, *, changes: list["PdfChange"], record, removal: bool = False):
        """Build a new asset version, make it live, then call record(activated) to note it in
        the registry; returns what record returns.

        Shared by ingest (each PDF's new chunks and pages; one version for a whole batch) and
        removal (one PDF, none of either).
        Nothing is final until record() succeeds: on any error before that the previous
        version is made live again and the staged files are deleted, so the old corpus
        keeps serving (staged activation).
        """
        root = self.config.asset_root
        pointer_path = root / "ACTIVE.json"
        previous_pointer = pointer_path.read_bytes() if pointer_path.exists() else None
        staged_version: Path | None = None
        local_index: dict[str, object] = {}
        new_primary = [chunk for change in changes for chunk in change.primary]
        new_visual = [chunk for change in changes for chunk in change.visual]
        filenames = [change.filename for change in changes]
        try:
            active_before = resolve_active_assets(root)
            base_snapshot = _read_snapshot(active_before)
            staged_version, staged_snapshot = stage_assets(
                asset_root=root,
                new_primary=new_primary,
                new_visual=new_visual,
                content_sha256=changes[0].content_hash,
                source_filenames=filenames,
                allow_empty=removal,
                removal=removal,
            )
            version = staged_snapshot["version"]

            if self.config.build_local:
                local_index = stage_local_collections(
                    asset_root=root,
                    version_id=version,
                    all_primary=_read_chunks(staged_version / "native.jsonl"),
                    all_visual=_read_chunks(staged_version / "arabic_ocr_secondary.jsonl"),
                    new_primary=new_primary,
                    new_visual=new_visual,
                    base_snapshot=base_snapshot,
                    source_filenames=filenames,
                )
                if not local_index:
                    logger.warning("Search index (Chroma) not updated for version %s", version)

            # Each PDF's edges replace its old ones; the next PDF starts from the edges just staged.
            supersession = {}
            for index, change in enumerate(changes):
                supersession = _stage_supersession_edges(
                    active_before if index == 0 else staged_version, staged_version, change.filename, change.pages, root
                )

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
            return self._activate(content_hash, filename, stored, metadata)
        finally:
            lock.release()

    def remove(self, content_sha256: str) -> dict:
        return _logged("remove-document", lambda: self._remove(content_sha256), subject=content_sha256)

    def _drop_unindexed(self, content_hash: str, known: dict) -> dict:
        """A queued or failed upload never reached the index: forget it, no new asset version."""
        with FileLock(str(self.config.asset_root / ".ingestion.lock"), timeout=60 * 30):
            if not self.registry.drop_unindexed(content_hash):
                raise ValueError("This PDF is being indexed right now; delete it once it is done.")
        stored = Path(str(known.get("stored_path") or ""))
        if stored.parent.is_dir() and stored.parent.parent == self.config.documents_dir:
            shutil.rmtree(stored.parent, ignore_errors=True)
        return {"status": "removed", "document_id": content_hash, "filename": known.get("original_filename")}

    def _remove(self, content_sha256: str) -> dict:
        """Drop a ready PDF from the active index; failed activation keeps the prior corpus."""
        content_hash = (content_sha256 or "").strip().lower()
        if not content_hash or len(content_hash) < 16:
            raise ValueError("Invalid document id")
        known = self.registry.get(content_hash)
        if known is not None and known.get("status") in ("queued", "failed"):
            return self._drop_unindexed(content_hash, known)
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
                changes=[PdfChange(content_hash, filename, [], [], [])], record=record, removal=True,
            )

            stored = str(known.get("stored_path") or "").strip()
            if stored:
                # Immutable dir is documents_dir / sha256 / file.pdf
                immutable_dir = Path(stored).parent
                if immutable_dir.is_dir() and immutable_dir.parent == self.config.documents_dir:
                    shutil.rmtree(immutable_dir, ignore_errors=True)
            return report
