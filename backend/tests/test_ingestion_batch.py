"""Uploads are queued, then indexed in batches: one new asset version per batch, not per PDF."""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion import docling_layout
from ingestion import pipeline as pipeline_module
from ingestion.index import resolve_active_assets
from ingestion.pipeline import IngestionConfig, IngestionPipeline
from ingestion.registry import IngestionRegistry
from runtime_retrieval import _read_chunks


def _pdf(path: Path, text: str) -> Path:
    import pymupdf

    document = pymupdf.open()
    document.new_page().insert_text((72, 100), text)
    document.save(path)
    document.close()
    return path


@pytest.fixture
def config(tmp_path: Path, monkeypatch) -> IngestionConfig:
    monkeypatch.setenv("BCT_VISUAL_BACKEND", "off")
    monkeypatch.delenv("BCT_INGEST_CLOUD_INDEX", raising=False)
    root = tmp_path / "assets"
    active = root / "versions" / "empty"
    active.mkdir(parents=True)
    for name in ("native.jsonl", "arabic_ocr_secondary.jsonl"):
        (active / name).write_text("", encoding="utf-8")
    (active / "snapshot.json").write_text('{"documents": 0}', encoding="utf-8")
    (root / "ACTIVE.json").write_text('{"active_version": "versions/empty"}', encoding="utf-8")
    return IngestionConfig(
        asset_root=root,
        documents_dir=tmp_path / "ingested",
        registry_path=tmp_path / "ingestion.sqlite3",
        gemini_cache_dir=tmp_path / "cache",
        build_local=False,
    )


def _queue(config: IngestionConfig, tmp_path: Path, names: list[str]) -> IngestionPipeline:
    pipeline = IngestionPipeline(config)
    for index, name in enumerate(names):
        report = pipeline.queue(_pdf(tmp_path / name, f"Circulaire numero {index} relative aux banques."))
        assert report["status"] == "queued"
    return pipeline


def _versions(config: IngestionConfig) -> set[str]:
    return {path.name for path in (config.asset_root / "versions").iterdir() if not path.name.startswith(".")}


def test_a_batch_makes_one_version_with_every_pdf(config, tmp_path):
    pipeline = _queue(config, tmp_path, ["Cir_2024_01_fr.pdf", "Cir_2024_02_fr.pdf", "Cir_2024_03_fr.pdf"])
    before = _versions(config)
    try:
        reports = pipeline.ingest_queued(10)
    finally:
        pipeline.close()

    assert [report["status"] for report in reports.values()] == ["ready"] * 3
    assert len(_versions(config) - before) == 1
    active = resolve_active_assets(config.asset_root)
    sources = {chunk.metadata["source"] for chunk in _read_chunks(active / "native.jsonl")}
    assert sources == {"Cir_2024_01_fr.pdf", "Cir_2024_02_fr.pdf", "Cir_2024_03_fr.pdf"}


def test_a_failing_pdf_does_not_stop_its_batch(config, tmp_path, monkeypatch):
    pipeline = _queue(config, tmp_path, ["Cir_2024_01_fr.pdf", "Cir_2024_02_fr.pdf"])
    real_prepare = IngestionPipeline._prepare

    def prepare(self, content_hash, filename, immutable_pdf, metadata):
        if filename == "Cir_2024_01_fr.pdf":
            raise ValueError("broken PDF")
        return real_prepare(self, content_hash, filename, immutable_pdf, metadata)

    monkeypatch.setattr(IngestionPipeline, "_prepare", prepare)
    try:
        reports = pipeline.ingest_queued(10)
        listed = {row["filename"]: row for row in pipeline.registry.list_ready(include_pending=True)}
    finally:
        pipeline.close()

    assert sorted(report["status"] for report in reports.values()) == ["failed", "ready"]
    assert listed["Cir_2024_01_fr.pdf"]["status"] == "failed"
    assert "broken PDF" in listed["Cir_2024_01_fr.pdf"]["error"]
    assert listed["Cir_2024_02_fr.pdf"]["searchable"] is True


def test_not_enough_memory_leaves_the_queue_waiting(config, tmp_path, monkeypatch):
    pipeline = _queue(config, tmp_path, ["Cir_2024_01_fr.pdf", "Cir_2024_02_fr.pdf"])

    def no_memory(path, raw_cache=None):
        raise docling_layout.NotEnoughMemory("needs 4 GB, 1.0 GB free")

    monkeypatch.setattr(pipeline_module.PdfExtractor, "extract", lambda self, *a, **k: no_memory(None))
    try:
        assert pipeline.ingest_queued(10) == {}
        statuses = [row["status"] for row in pipeline.registry.list_ready(include_pending=True)]
    finally:
        pipeline.close()

    assert statuses == ["queued", "queued"]


def test_a_pdf_interrupted_twice_fails_instead_of_looping(config, tmp_path):
    pipeline = _queue(config, tmp_path, ["Cir_2024_01_fr.pdf"])
    pipeline.close()
    registry = IngestionRegistry(config.registry_path)
    try:
        [job] = registry.next_queued(1)
        for expected in ("queued", "failed"):
            registry.start(job["content_sha256"], job["original_filename"], job["stored_path"])
            registry.requeue_interrupted()
            assert registry.get(job["content_sha256"])["status"] == expected
    finally:
        registry.close()


def test_a_queued_or_failed_upload_can_be_deleted_without_a_new_version(config, tmp_path):
    pipeline = _queue(config, tmp_path, ["Cir_2024_01_fr.pdf"])
    before = _versions(config)
    try:
        [job] = pipeline.registry.next_queued(1)
        report = pipeline.remove(job["content_sha256"])
        statuses = pipeline.registry.list_ready(include_pending=True)
    finally:
        pipeline.close()

    assert report["status"] == "removed"
    assert statuses == []
    assert _versions(config) == before


def test_uploads_are_queued_again_when_their_index_was_replaced(config, tmp_path):
    pipeline = _queue(config, tmp_path, ["Cir_2024_01_fr.pdf"])
    try:
        pipeline.ingest_queued(10)
        assert pipeline.registry.requeue_indexed() == 1
        assert [row["status"] for row in pipeline.registry.list_ready(include_pending=True)] == ["queued"]
        reports = pipeline.ingest_queued(10)
    finally:
        pipeline.close()

    assert [report["status"] for report in reports.values()] == ["ready"]
