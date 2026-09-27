"""First-ingest / empty-corpus staging (no Voyage / Gemini)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from ingestion import index as index_module
from runtime_retrieval import _read_chunks, document_binding


def test_read_chunks_allows_empty_bootstrap_file(tmp_path: Path):
    path = tmp_path / "arabic_ocr_secondary.jsonl"
    path.write_text("", encoding="utf-8")
    assert _read_chunks(path) == []


def test_stage_cloud_allows_french_first_with_empty_visual(tmp_path: Path, monkeypatch):
    """Empty Docker bootstrap + FR circular must activate (no Arabic OCR chunks)."""
    root = tmp_path / "assets"
    active = root / "versions" / "empty"
    active.mkdir(parents=True)
    (active / "native.jsonl").write_text("", encoding="utf-8")
    (active / "arabic_ocr_secondary.jsonl").write_text("", encoding="utf-8")
    (active / "snapshot.json").write_text('{"documents": 0}', encoding="utf-8")
    (root / "ACTIVE.json").write_text(
        '{"active_version": "versions/empty"}', encoding="utf-8"
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
    monkeypatch.setenv("BCT_INGEST_CLOUD_INDEX", "1")

    primary = [
        Document(
            page_content="Horaire de la seance unique des banques.",
            metadata={"source": "Cir_2016_01_fr.pdf", "page": 1, "pages": [1]},
        )
    ]
    staged, snapshot = index_module.stage_cloud_assets(
        asset_root=root,
        new_primary=primary,
        new_visual=[],
        content_sha256="abc123def456",
        source_filename="Cir_2016_01_fr.pdf",
    )
    assert staged.is_dir()
    assert snapshot["native_chunks"] == 1
    assert snapshot["arabic_visual_chunks"] == 0
    natives = _read_chunks(staged / "native.jsonl")
    assert len(natives) == 1
    assert _read_chunks(staged / "arabic_ocr_secondary.jsonl") == []


def test_stage_skips_voyage_embed_on_local_hybrid(tmp_path: Path, monkeypatch):
    """local_hybrid ingest must not call Voyage (429 / quota); Chroma is the search index."""
    root = tmp_path / "assets"
    active = root / "versions" / "empty"
    active.mkdir(parents=True)
    (active / "native.jsonl").write_text("", encoding="utf-8")
    (active / "arabic_ocr_secondary.jsonl").write_text("", encoding="utf-8")
    (active / "snapshot.json").write_text('{"documents": 0}', encoding="utf-8")
    (root / "ACTIVE.json").write_text(
        '{"active_version": "versions/empty"}', encoding="utf-8"
    )

    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("Voyage client must not be constructed on local_hybrid ingest")

    from runtime_retrieval import CloudEmbedSpec

    fake_spec = CloudEmbedSpec(
        key="voyage",
        provider="voyage",
        model="fake",
        dimension=4,
        contextual=False,
    )
    monkeypatch.setenv("BCT_DEFAULT_PROFILE", "local_hybrid")
    monkeypatch.delenv("BCT_INGEST_CLOUD_INDEX", raising=False)
    monkeypatch.setattr(index_module, "cloud_embed_spec", lambda value=None: fake_spec)
    monkeypatch.setattr(index_module, "create_cloud_runtime_client", boom)

    staged, snapshot = index_module.stage_cloud_assets(
        asset_root=root,
        new_primary=[
            Document(
                page_content="Bulletin statistique page.",
                metadata={"source": "BSF_fr.pdf", "page": 1, "pages": [1]},
            )
        ],
        new_visual=[],
        content_sha256="stathash001",
        source_filename="BSF_fr.pdf",
    )
    assert calls["n"] == 0
    assert snapshot.get("cloud_embed") is False
    assert snapshot["native_chunks"] == 1
    assert staged.is_dir()
    # New native chunks are not Voyage-embedded; empty visual index may still be written.
    assert not list((staged / "indexes").glob("*native*.npy"))


def test_load_bound_index_empty_documents_is_empty_array(tmp_path: Path):
    from runtime_retrieval import CLOUD_EMBED_SPECS, _load_bound_index

    vectors = _load_bound_index(tmp_path, "arabic_ocr_secondary", [], CLOUD_EMBED_SPECS["voyage"])
    assert vectors.shape == (0, CLOUD_EMBED_SPECS["voyage"].dimension)
