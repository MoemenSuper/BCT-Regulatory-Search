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
    build_graph: bool = False

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
            build_graph=os.environ.get("BCT_INGEST_GRAPH", os.environ.get("BCT_ENABLE_GRAPH", "0")) == "1",
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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
            graph_driver = None
            graph_store = None
            staged_candidate_ids: list[str] = []
            pointer_path = self.config.asset_root / "ACTIVE.json"
            previous_pointer = pointer_path.read_bytes() if pointer_path.exists() else None
            activation_committed = False
            try:
                transcriber = None
                if os.environ.get("BCT_GEMINI_VISUAL", "1") == "1":
                    transcriber = GeminiVisualTranscriber(self.config.gemini_cache_dir)
                structured = PdfExtractor(visual_transcriber=transcriber).extract(immutable_pdf)
                structured.document_number = metadata.get("document_number")
                structured.publication_date = metadata.get("publication_date")
                structured.metadata["administrator_metadata"] = metadata
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

                graph_report = {"enabled": self.config.build_graph, "candidate_count": 0, "activated_count": 0}
                if self.config.build_graph:
                    try:
                        from regulatory_graph_lite.builder import create_builder_from_environment
                        from regulatory_graph_lite.identity import instrument_from_filename
                        from regulatory_graph_lite.store import Neo4jGraphLiteStore, open_neo4j_driver_from_environment

                        if instrument_from_filename(filename) is None:
                            graph_report["skipped_reason"] = "unparseable_bct_instrument_filename"
                        else:
                            graph_driver = open_neo4j_driver_from_environment()
                            graph_store = Neo4jGraphLiteStore(
                                graph_driver,
                                database=os.environ.get("BCT_NEO4J_DATABASE", "neo4j"),
                            )
                            graph_store.ensure_schema()
                            builder = create_builder_from_environment(
                                driver=graph_driver,
                                store=graph_store,
                                documents_dir=self.config.documents_dir,
                                chunk_files=(staged_version / "native.jsonl",),
                                activate_immediately=False,
                            )
                            for page in structured.pages:
                                # Prefer the complete Gemini transcription for
                                # Arabic relationship extraction when available;
                                # Graph Lite still verifies the exact quotation and
                                # target identity before accepting an edge.
                                graph_text = (
                                    str(page.metadata.get("visual_text") or "").strip()
                                    if page.metadata.get("visual_complete") is True
                                    else page.raw_text
                                )
                                report = builder.ingest_page(
                                    source_file=structured.filename,
                                    page=page.page_number,
                                    text=graph_text,
                                )
                                staged_candidate_ids.extend(candidate.candidate_id for candidate in report.accepted)
                            graph_report["candidate_count"] = len(staged_candidate_ids)
                    except Exception as graph_error:
                        if graph_store is not None and staged_candidate_ids:
                            try:
                                graph_store.delete_candidates(staged_candidate_ids)
                            except Exception:
                                pass
                        staged_candidate_ids.clear()
                        graph_report["warning"] = f"{type(graph_error).__name__}: {graph_error}"

                pointer = activate_assets(
                    self.config.asset_root,
                    staged_version,
                    snapshot_updates=snapshot_updates,
                )

                # Commit searchable assets to the durable local ledger before graph
                # promotion. Graph Lite is an optional sidecar and must never make a
                # successfully indexed PDF disappear from the primary RAG runtime.
                report = {
                    "status": "ready",
                    "duplicate": False,
                    "filename": filename,
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
                    "local_indexed": self.config.build_local,
                    "graph": graph_report,
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

                # Promote staged graph relationships only after the document is
                # searchable. A graph outage degrades to normal RAG rather than
                # rolling back a valid ingestion. Activate new edges first so a
                # cleanup failure cannot temporarily erase all graph evidence.
                if graph_store is not None:
                    try:
                        if staged_candidate_ids:
                            graph_report["activated_count"] = graph_store.activate_candidates(staged_candidate_ids)
                        graph_report["deactivated_previous_count"] = graph_store.deactivate_source_relationships(
                            filename, keep_candidate_ids=staged_candidate_ids
                        )
                    except Exception as graph_error:
                        graph_report["warning"] = f"{type(graph_error).__name__}: {graph_error}"

                # Refresh the stored report with post-activation graph status. If
                # this cosmetic update fails, the asset/ledger commit remains valid.
                try:
                    self.registry.ready(
                        content_hash,
                        stored_path=str(immutable_pdf),
                        asset_version=staged_snapshot["version"],
                        report=report,
                    )
                except Exception:
                    pass
                return report
            except Exception as error:
                if not activation_committed and graph_store is not None and staged_candidate_ids:
                    try:
                        graph_store.delete_candidates(staged_candidate_ids)
                    except Exception:
                        pass
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
            finally:
                if graph_driver is not None:
                    graph_driver.close()
