from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock
from langfuse import get_client, propagate_attributes

from .chunk import build_runtime_chunks
from .extract import PdfExtractor, render_page_png, sha256_file
from .index import (
    activate_assets,
    discard_staged_local_collections,
    resolve_active_assets,
    stage_cloud_assets,
    stage_local_collections,
)
from .registry import SEARCHABLE, IngestionRegistry
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
    """Move spilled chart-page PNGs beside the immutable PDF; drop temp paths/bytes."""
    images_dir = immutable_dir / "page-images"
    written = 0
    for page in structured.pages:
        png = page.metadata.pop("page_image_png", None)
        tmp = page.metadata.pop("page_image_tmp", None)
        if not png and not tmp:
            continue
        images_dir.mkdir(parents=True, exist_ok=True)
        path = images_dir / f"page-{page.page_number}.png"
        src = Path(tmp) if tmp else None
        if src is not None and src.is_file():
            shutil.move(str(src), str(path))
        elif png:
            path.write_bytes(png)
            if src is not None:
                src.unlink(missing_ok=True)
        else:
            continue
        page.metadata["page_image_path"] = str(path)
        page.metadata["has_chart"] = True
        written += 1
    return written


def _ensure_secondary_page_images(structured, pdf_path: Path, immutable_dir: Path) -> int:
    """Persist capped PNGs only for chart/empty pages still missing an image.

    Full-document re-renders of 100–250 page statistical PDFs were a major OOM /
    latency source; born-digital pages with native text do not need a raster.
    """
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
            flags = set(page.quality_flags or [])
            needs_image = bool(page.metadata.get("has_chart") or page.metadata.get("chart_suspect")) or (
                "chart_suspect" in flags
            ) or (not str(page.raw_text or "").strip())
            if not needs_image:
                continue
            images_dir.mkdir(parents=True, exist_ok=True)
            try:
                png = render_page_png(pdf.load_page(page.page_number - 1))
            except Exception:
                continue
            path = images_dir / f"page-{page.page_number}.png"
            path.write_bytes(png)
            page.metadata["page_image_path"] = str(path)
            written += 1
            png = None
            try:
                pymupdf.TOOLS.store_shrink(100)
            except Exception:
                pass
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
        "chart_suspect": sum(1 for page in structured.pages if "chart_suspect" in page.quality_flags),
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
        """Extract (visual results from the page ledger), chunk, stage and activate. Caller holds the lock."""
        immutable_dir = immutable_pdf.parent
        staged_version: Path | None = None
        pointer_path = self.config.asset_root / "ACTIVE.json"
        previous_pointer = pointer_path.read_bytes() if pointer_path.exists() else None
        activation_committed = False
        try:
            with _step("extract-document", input={"pdf": filename, "quick": quick}) as span:
                visual_results, visual_model = self._visual_results(content_hash)
                structured = PdfExtractor(visual_model=visual_model).extract(
                    immutable_pdf, visual_results=visual_results
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
            with _step("persist-page-images", input={"doc_kind": doc_kind}) as span:
                page_images = _persist_page_images(structured, immutable_dir)
                page_images += _ensure_secondary_page_images(structured, immutable_pdf, immutable_dir)
                span.update(output={"page_images": page_images})
            structured.metadata["chart_page_images"] = page_images
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
            active_before = resolve_active_assets(self.config.asset_root)
            base_snapshot = _read_snapshot(active_before)
            with _step("stage-assets", input={"active_before": active_before.name}) as span:
                staged_version, staged_snapshot = stage_cloud_assets(
                    asset_root=self.config.asset_root,
                    new_primary=primary,
                    new_visual=visual,
                    content_sha256=content_hash,
                    source_filename=filename,
                )
                span.update(output=staged_snapshot)

            snapshot_updates: dict[str, object] = {}
            if self.config.build_local:
                from runtime_retrieval import _read_chunks

                with _step("stage-local-index", input={"version": staged_snapshot["version"]}) as span:
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
                    span.update(
                        output=local or {"skipped": True},
                        **({} if local else {"level": "WARNING", "status_message": "local Chroma index not updated"}),
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
            get_client().create_event(
                name="merge-supersession-edges",
                output=supersession_report,
                **({"level": "WARNING", "status_message": str(supersession_report["warning"])}
                   if "warning" in supersession_report else {}),
            )

            with _step("activate-version", input={"version": staged_snapshot["version"]}):
                pointer = activate_assets(
                    self.config.asset_root,
                    staged_version,
                    snapshot_updates=snapshot_updates,
                )

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
                "gemini_visual_pages": int(structured.metadata.get("visual_page_count", 0)),
                "visual_pages": int(structured.metadata.get("visual_page_count", 0)),
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
                    status=status,
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
            if quick and not activation_committed:
                # An enrichment batch failing keeps the document live on its prior version.
                self.registry.fail(content_hash, f"{type(error).__name__}: {error}")
            raise

    def _visual_results(self, content_hash: str) -> tuple[dict, str | None]:
        from .gemini_visual import VisualPage

        results: dict = {}
        models: set[str] = set()
        for page, (state, payload, model) in self.registry.page_results(content_hash).items():
            if state == "done":
                results[page] = VisualPage.model_validate_json(payload)
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
