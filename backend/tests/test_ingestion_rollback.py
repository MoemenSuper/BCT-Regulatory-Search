import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

import ingestion.pipeline as pipeline_module
from ingestion.models import Block, Page, StructuredDocument
from ingestion.pipeline import IngestionConfig, IngestionPipeline


def _make_pdf(path: Path):
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "A valid BCT regulatory page for rollback testing.")
    document.save(path)
    document.close()


def test_failed_ledger_commit_restores_previous_active_version(tmp_path: Path, monkeypatch):
    asset_root = tmp_path / "assets"
    old_version = asset_root / "versions" / "old"
    old_version.mkdir(parents=True)
    (old_version / "snapshot.json").write_text(json.dumps({"version": "old"}), encoding="utf-8")
    previous_pointer = {"active_version": "versions/old", "activated_at": "old"}
    (asset_root / "ACTIVE.json").write_text(json.dumps(previous_pointer), encoding="utf-8")

    source = tmp_path / "Cir_2026_99_fr.pdf"
    _make_pdf(source)

    structured = StructuredDocument(
        filename=source.name,
        language="fr",
        content_sha256="will-be-replaced",
        pages=[
            Page(
                page_number=1,
                raw_text="A valid BCT regulatory page for rollback testing.",
                blocks=[Block(type="paragraph", text="A valid BCT regulatory page for rollback testing.", page_number=1)],
            )
        ],
    )

    class FakeExtractor:
        def __init__(self, visual_transcriber=None):
            pass

        def extract(self, _path):
            return structured

    def fake_chunks(document):
        return [Document(page_content=document.pages[0].raw_text, metadata={"source": document.filename, "page": 1, "chunk_id": "c1"})], []

    staged_holder = {}

    def fake_stage_cloud_assets(*, asset_root, new_primary, new_visual, content_sha256, source_filename):
        staged = Path(asset_root) / "versions" / "new"
        staged.mkdir(parents=True, exist_ok=False)
        (staged / "snapshot.json").write_text(json.dumps({"version": "new"}), encoding="utf-8")
        staged_holder["path"] = staged
        return staged, {"version": "new"}

    def fake_activate(asset_root, version_dir, *, snapshot_updates=None):
        pointer = {"active_version": "versions/new", "activated_at": "new"}
        Path(asset_root, "ACTIVE.json").write_text(json.dumps(pointer), encoding="utf-8")
        return pointer

    monkeypatch.setenv("BCT_GEMINI_VISUAL", "0")
    monkeypatch.setattr(pipeline_module, "PdfExtractor", FakeExtractor)
    monkeypatch.setattr(pipeline_module, "build_runtime_chunks", fake_chunks)
    monkeypatch.setattr(pipeline_module, "stage_cloud_assets", fake_stage_cloud_assets)
    monkeypatch.setattr(pipeline_module, "activate_assets", fake_activate)

    config = IngestionConfig(
        asset_root=asset_root,
        documents_dir=tmp_path / "ingested",
        registry_path=tmp_path / "ingestion.sqlite3",
        gemini_cache_dir=tmp_path / "gemini-cache",
        build_local=False,
        build_graph=False,
    )
    ingestion = IngestionPipeline(config)
    original_ready = ingestion.registry.ready

    def fail_ready(*args, **kwargs):
        raise RuntimeError("simulated ledger commit failure")

    monkeypatch.setattr(ingestion.registry, "ready", fail_ready)
    try:
        with pytest.raises(RuntimeError, match="simulated ledger commit failure"):
            ingestion.ingest(source)
    finally:
        # Restore so the registry can be inspected normally on implementations
        # where monkeypatch cleanup occurs after this test body.
        monkeypatch.setattr(ingestion.registry, "ready", original_ready)
        ingestion.close()

    active = json.loads((asset_root / "ACTIVE.json").read_text(encoding="utf-8"))
    assert active == previous_pointer
    assert not staged_holder["path"].exists()
