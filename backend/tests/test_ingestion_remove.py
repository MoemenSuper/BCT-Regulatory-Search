"""Remove a ready PDF from the active corpus (index staging + registry)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from ingestion import index as index_module
from ingestion.pipeline import IngestionConfig, IngestionPipeline
from ingestion.registry import IngestionRegistry
from runtime_retrieval import _read_chunks


def test_registry_mark_removed(tmp_path: Path):
    registry = IngestionRegistry(tmp_path / "ingestion.sqlite3")
    try:
        registry.start("hash-ready", "Cir_2026_01_fr.pdf", "/doc.pdf")
        registry.ready(
            "hash-ready",
            stored_path="/doc.pdf",
            asset_version="v1",
            report={"pages": 1},
        )
        registry.mark_removed("hash-ready", asset_version="v2")
        assert registry.get("hash-ready")["status"] == "removed"
        assert registry.list_ready() == []
    finally:
        registry.close()


def test_pipeline_remove_strips_source_and_keeps_peers(tmp_path: Path, monkeypatch):
    asset_root = tmp_path / "assets"
    docs_dir = tmp_path / "ingested"
    keep_hash = "a" * 64
    drop_hash = "b" * 64
    keep_dir = docs_dir / keep_hash
    drop_dir = docs_dir / drop_hash
    keep_dir.mkdir(parents=True)
    drop_dir.mkdir(parents=True)
    keep_pdf = keep_dir / "Keep_fr.pdf"
    drop_pdf = drop_dir / "Drop_fr.pdf"
    keep_pdf.write_bytes(b"%PDF-1.4 keep")
    drop_pdf.write_bytes(b"%PDF-1.4 drop")

    active = asset_root / "versions" / "old"
    active.mkdir(parents=True)
    keep_doc = Document(page_content="keep text", metadata={"source": "Keep_fr.pdf", "page": 1})
    drop_doc = Document(page_content="drop text", metadata={"source": "Drop_fr.pdf", "page": 1})
    (active / "native.jsonl").write_text(
        "\n".join(
            json.dumps({"page_content": d.page_content, "metadata": d.metadata}, ensure_ascii=False)
            for d in (keep_doc, drop_doc)
        )
        + "\n",
        encoding="utf-8",
    )
    (active / "arabic_ocr_secondary.jsonl").write_text("", encoding="utf-8")
    (active / "snapshot.json").write_text(json.dumps({"version": "old"}), encoding="utf-8")
    (active / "supersession_edges.jsonl").write_text(
        json.dumps(
            {
                "source_instrument": "cir:2026:2",
                "target_instrument": "cir:2025:1",
                "action": "ABROGATE",
                "source_file": "Drop_fr.pdf",
                "source_page": 1,
                "target_article": None,
                "quote": "La présente circulaire abroge la circulaire n° 2025-01.",
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "source_instrument": "cir:2026:1",
                "target_instrument": "cir:2024:1",
                "action": "AMEND",
                "source_file": "Keep_fr.pdf",
                "source_page": 1,
                "target_article": None,
                "quote": "La présente circulaire modifie la circulaire n° 2024-01.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (asset_root / "ACTIVE.json").write_text(
        json.dumps({"active_version": "versions/old"}), encoding="utf-8"
    )

    class FakeClient:
        dimension = 4

        def embed_document_chunks(self, texts, **_kwargs):
            return np.ones((len(texts), self.dimension), dtype=np.float32)

    from runtime_retrieval import CloudEmbedSpec

    fake_spec = CloudEmbedSpec(
        key="voyage",
        provider="voyage",
        model="fake",
        dimension=4,
        contextual=False,
    )
    monkeypatch.setattr(index_module, "cloud_embed_spec", lambda value=None: fake_spec)
    monkeypatch.setattr(
        index_module, "create_cloud_runtime_client", lambda *a, **k: FakeClient()
    )
    monkeypatch.setenv("BCT_GEMINI_VISUAL", "0")

    registry = IngestionRegistry(tmp_path / "ingestion.sqlite3")
    try:
        registry.start(keep_hash, "Keep_fr.pdf", str(keep_pdf))
        registry.ready(keep_hash, stored_path=str(keep_pdf), asset_version="old", report={"pages": 1})
        registry.start(drop_hash, "Drop_fr.pdf", str(drop_pdf))
        registry.ready(drop_hash, stored_path=str(drop_pdf), asset_version="old", report={"pages": 1})
    finally:
        registry.close()

    config = IngestionConfig(
        asset_root=asset_root,
        documents_dir=docs_dir,
        registry_path=tmp_path / "ingestion.sqlite3",
        gemini_cache_dir=tmp_path / "gemini-cache",
        build_local=False,
    )
    pipeline = IngestionPipeline(config)
    try:
        report = pipeline.remove(drop_hash)
    finally:
        pipeline.close()

    assert report["status"] == "removed"
    assert report["filename"] == "Drop_fr.pdf"
    assert report["native_chunks_remaining"] == 1
    assert not drop_dir.exists()
    assert keep_dir.exists()

    active_after = json.loads((asset_root / "ACTIVE.json").read_text(encoding="utf-8"))
    staged = asset_root / active_after["active_version"]
    natives = _read_chunks(staged / "native.jsonl")
    assert [d.metadata["source"] for d in natives] == ["Keep_fr.pdf"]
    edges = (staged / "supersession_edges.jsonl").read_text(encoding="utf-8")
    assert "Drop_fr.pdf" not in edges
    assert "Keep_fr.pdf" in edges

    registry = IngestionRegistry(tmp_path / "ingestion.sqlite3")
    try:
        assert registry.get(drop_hash)["status"] == "removed"
        ready = registry.list_ready()
        assert len(ready) == 1
        assert ready[0]["document_id"] == keep_hash
    finally:
        registry.close()
