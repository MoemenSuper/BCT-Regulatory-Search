"""Two-phase ingest: quick native pass (searchable, `enriching`), then background page reading."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from ingestion import enrichment as enrichment_module
from ingestion.enrichment import EnrichmentWorker, Foreground
from ingestion.models import VisualPage
from ingestion.pipeline import IngestionConfig, IngestionPipeline
from runtime_retrieval import _read_chunks

NATIVE = "Circulaire aux banques relative au taux directeur et aux conditions de refinancement."
VISUAL = "Tableau statistique des encours de credit 2024 transcrit depuis l'image."


def _make_pdf(path: Path, *, blank_pages: int = 0) -> Path:
    import pymupdf

    document = pymupdf.open()
    document.new_page().insert_text((72, 100), NATIVE)
    # Drawn-only page: nothing extractable, needs visual reading.
    document.new_page().draw_rect(pymupdf.Rect(72, 72, 300, 200), fill=(0, 0, 0))
    for _ in range(blank_pages):
        document.new_page()
    document.save(path)
    document.close()
    return path


@pytest.fixture
def config(tmp_path: Path, monkeypatch) -> IngestionConfig:
    monkeypatch.setenv("BCT_DEFAULT_PROFILE", "local_hybrid")
    monkeypatch.setenv("BCT_VISUAL_BACKEND", "local")
    monkeypatch.delenv("BCT_INGEST_CLOUD_INDEX", raising=False)
    monkeypatch.setenv("BCT_ENRICH_IDLE_SECONDS", "0")
    root = tmp_path / "assets"
    active = root / "versions" / "empty"
    active.mkdir(parents=True)
    (active / "native.jsonl").write_text("", encoding="utf-8")
    (active / "arabic_ocr_secondary.jsonl").write_text("", encoding="utf-8")
    (active / "snapshot.json").write_text('{"documents": 0}', encoding="utf-8")
    (root / "ACTIVE.json").write_text('{"active_version": "versions/empty"}', encoding="utf-8")
    return IngestionConfig(
        asset_root=root,
        documents_dir=tmp_path / "ingested",
        registry_path=tmp_path / "ingestion.sqlite3",
        gemini_cache_dir=tmp_path / "gemini-cache",
        build_local=False,
    )


class FakeTranscriber:
    model = "fake-ocr"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[int] = []
        self.closed = 0

    def transcribe(self, *, image_png: bytes, page_number: int, **_):
        assert image_png.startswith(b"\x89PNG")
        self.calls.append(page_number)
        if self.fail:
            raise RuntimeError("Paddle worker died (exit 3221225477)")
        return VisualPage(transcription=VISUAL, complete=True)

    def close(self):
        self.closed += 1


def _active_texts(config: IngestionConfig) -> str:
    pointer = json.loads((config.asset_root / "ACTIVE.json").read_text(encoding="utf-8"))
    chunks = _read_chunks(config.asset_root / pointer["active_version"] / "native.jsonl")
    return "\n".join(chunk.page_content for chunk in chunks)


def _quick_ingest(config: IngestionConfig, tmp_path: Path) -> dict:
    pipeline = IngestionPipeline(config)
    try:
        return pipeline.ingest(_make_pdf(tmp_path / "Cir_2024_01_fr.pdf"))
    finally:
        pipeline.close()


def _registry_row(config: IngestionConfig, sha: str) -> dict:
    from ingestion.registry import IngestionRegistry

    registry = IngestionRegistry(config.registry_path)
    try:
        return {**(registry.get(sha) or {}), "progress": registry.progress(sha)}
    finally:
        registry.close()


def test_quick_pass_is_searchable_then_enrichment_completes(config, tmp_path, monkeypatch):
    report = _quick_ingest(config, tmp_path)
    assert report["status"] == "enriching"
    assert report["searchable"] is True
    assert report["enrichment"]["pending"] == 1
    assert NATIVE[:40] in _active_texts(config)
    assert VISUAL not in _active_texts(config)

    fake = FakeTranscriber()
    monkeypatch.setattr(enrichment_module, "build_visual_transcriber", lambda *_a, **_k: fake)
    refreshed = []
    worker = EnrichmentWorker(config, on_activated=lambda: refreshed.append(True))
    worker.run_until_idle()

    assert fake.calls == [2]
    assert fake.closed == 1
    assert refreshed
    row = _registry_row(config, report["content_sha256"])
    assert row["status"] == "ready"
    assert row["progress"] == {**row["progress"], "done": 1, "pending": 0, "failed": 0}
    texts = _active_texts(config)
    assert VISUAL in texts and NATIVE[:40] in texts
    assert texts.count(NATIVE[:40]) == 1  # re-activation replaced, not duplicated, the document


def test_failing_backend_degrades_honestly_and_retry_requeues(config, tmp_path, monkeypatch):
    monkeypatch.setenv("BCT_ENRICH_BREAKER_FAILURES", "1")
    report = _quick_ingest(config, tmp_path)
    fake = FakeTranscriber(fail=True)
    monkeypatch.setattr(enrichment_module, "build_visual_transcriber", lambda *_a, **_k: fake)
    worker = EnrichmentWorker(config)
    worker.run_until_idle()

    sha = report["content_sha256"]
    row = _registry_row(config, sha)
    assert len(fake.calls) == 3  # max attempts, each pass tripping the breaker and resuming
    assert row["status"] == "ready_degraded"
    assert row["progress"]["failed"] == 1
    assert NATIVE[:40] in _active_texts(config)  # native pages stay live
    assert worker.snapshot()["state"] == "cooldown"

    from ingestion.registry import IngestionRegistry

    registry = IngestionRegistry(config.registry_path)
    try:
        assert registry.retry_failed_pages(sha) == 1
        assert registry.get(sha)["status"] == "enriching"
    finally:
        registry.close()
    fake.fail = False
    worker.run_until_idle()
    assert _registry_row(config, sha)["status"] == "ready"
    assert VISUAL in _active_texts(config)


def test_pages_without_extractable_text_are_still_queued_for_visual_reading(config, tmp_path):
    pipeline = IngestionPipeline(config)
    try:
        report = pipeline.ingest(_make_pdf(tmp_path / "Balance_fr.pdf", blank_pages=3))
    finally:
        pipeline.close()
    assert report["enrichment"]["pending"] == 4  # the OCR/VLM decides whether a page is empty


def test_empty_reading_settles_the_page_without_tripping_the_breaker(config, tmp_path, monkeypatch):
    monkeypatch.setenv("BCT_ENRICH_BREAKER_FAILURES", "1")
    report = _quick_ingest(config, tmp_path)

    class Empty(FakeTranscriber):
        def transcribe(self, **kwargs):
            super().transcribe(**kwargs)
            return VisualPage(transcription="", complete=True)

    fake = Empty()
    monkeypatch.setattr(enrichment_module, "build_visual_transcriber", lambda *_a, **_k: fake)
    worker = EnrichmentWorker(config)
    worker.run_until_idle()
    assert fake.calls == [2]
    assert worker.snapshot()["state"] != "cooldown"
    row = _registry_row(config, report["content_sha256"])
    assert row["status"] == "ready"
    assert row["progress"]["failed"] == 0


def test_a_page_that_keeps_failing_does_not_block_the_next_pdf(config, tmp_path, monkeypatch):
    first = _quick_ingest(config, tmp_path)
    pipeline = IngestionPipeline(config)
    try:
        second = pipeline.ingest(_make_pdf(tmp_path / "Cir_2024_02_fr.pdf", blank_pages=1))
    finally:
        pipeline.close()
    order: list[str] = []

    class FailsFirst(FakeTranscriber):
        def transcribe(self, *, source_pdf_sha256: str, **kwargs):
            order.append(source_pdf_sha256)
            if source_pdf_sha256 == first["content_sha256"]:
                raise RuntimeError("ResourceExhaustedError: Out of memory error on GPU 0")
            return super().transcribe(**kwargs)

    monkeypatch.setattr(enrichment_module, "build_visual_transcriber", lambda *_a, **_k: FailsFirst())
    EnrichmentWorker(config).run_until_idle()
    assert order[:2] == [first["content_sha256"], second["content_sha256"]]
    assert _registry_row(config, second["content_sha256"])["status"] == "ready"
    assert _registry_row(config, first["content_sha256"])["status"] == "ready_degraded"


def test_chat_preempts_the_page_being_read_without_counting_an_attempt(config, tmp_path, monkeypatch):
    report = _quick_ingest(config, tmp_path)

    class Blocking(FakeTranscriber):
        def __init__(self):
            super().__init__()
            self.started, self.killed = threading.Event(), threading.Event()

        def transcribe(self, **kwargs):
            page = super().transcribe(**kwargs)
            if len(self.calls) == 1:
                self.started.set()
                assert self.killed.wait(5)
                raise RuntimeError("PaddleOCR-VL worker died (exit=1)")
            return page

        def kill(self):
            self.killed.set()

    fake = Blocking()
    monkeypatch.setattr(enrichment_module, "build_visual_transcriber", lambda *_a, **_k: fake)
    gate = Foreground()
    worker = EnrichmentWorker(config, foreground=gate)
    gate.on_busy = worker._preempt
    thread = threading.Thread(target=worker.run_until_idle)
    thread.start()
    assert fake.started.wait(5)
    with gate.busy():  # a chat request arrives mid-page
        assert fake.killed.wait(5)
        time.sleep(0.2)
        progress = _registry_row(config, report["content_sha256"])["progress"]
        assert progress["pending"] == 1 and progress["failed"] == 0
    thread.join(15)
    assert fake.calls == [2, 2]
    assert _registry_row(config, report["content_sha256"])["status"] == "ready"


def test_restart_resumes_from_page_checkpoint(config, tmp_path, monkeypatch):
    report = _quick_ingest(config, tmp_path)
    sha = report["content_sha256"]
    from ingestion.registry import IngestionRegistry

    registry = IngestionRegistry(config.registry_path)
    try:
        registry.page_done(sha, 2, result_json=VisualPage(transcription=VISUAL, complete=True).model_dump_json(),
                           model="fake-ocr", seconds=1.0)
    finally:
        registry.close()
    fake = FakeTranscriber()
    monkeypatch.setattr(enrichment_module, "build_visual_transcriber", lambda *_a, **_k: fake)
    EnrichmentWorker(config).run_until_idle()
    assert fake.calls == []  # checkpointed page is not read again
    assert _registry_row(config, sha)["status"] == "ready"
    assert VISUAL in _active_texts(config)


def test_admin_api_lists_progress_and_retries_degraded_pages(config, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import app as app_module
    from identity import require_admin
    from ingestion.registry import IngestionRegistry

    report = _quick_ingest(config, tmp_path)
    sha = report["content_sha256"]
    registry = IngestionRegistry(config.registry_path)
    registry.page_failed(sha, 2, error="RuntimeError: boom", max_attempts=1)
    registry.close()

    monkeypatch.setattr(IngestionConfig, "from_environment", classmethod(lambda cls: config))
    wakes = []
    api = FastAPI()
    api.state.enrichment = SimpleNamespace(
        wake=lambda **kw: wakes.append(kw), snapshot=lambda: {"state": "reading", "page": 2}
    )
    api.dependency_overrides[require_admin] = lambda: None
    api.include_router(app_module.documents_router)
    client = TestClient(api)

    [listed] = client.get("/documents").json()
    assert listed["status"] == "enriching"
    assert listed["enrichment"]["failed"] == 1
    assert client.get("/documents/enrichment").json() == {"state": "reading", "page": 2}

    response = client.post(f"/documents/{sha}/retry-enrichment")
    assert response.status_code == 200
    assert response.json()["requeued_pages"] == 1
    assert response.json()["enrichment"]["pending"] == 1
    assert wakes == [{"reset_cooldown": True}]
    assert client.post(f"/documents/{'0' * 64}/retry-enrichment").status_code == 404


def test_chat_route_holds_the_foreground_gate():
    import app as app_module
    from ingestion.enrichment import FOREGROUND

    [route] = [r for r in app_module.app.routes if getattr(r, "path", None) == "/chat"]
    assert any(dep.call is app_module._foreground for dep in route.dependant.dependencies)
    stop = threading.Event()
    dependency = app_module._foreground()
    next(dependency)
    assert FOREGROUND._active == 1
    with pytest.raises(StopIteration):
        next(dependency)
    assert FOREGROUND._active == 0
    assert FOREGROUND.wait_idle(stop, grace=0.0) < 0.1


def test_foreground_gate_waits_for_chat():
    gate = Foreground()
    stop = threading.Event()
    released = threading.Event()

    def chat():
        with gate.busy():
            released.wait(5)

    thread = threading.Thread(target=chat)
    thread.start()
    time.sleep(0.05)
    started = time.monotonic()
    threading.Timer(0.3, released.set).start()
    waited = gate.wait_idle(stop, grace=0.2)
    thread.join()
    assert waited >= 0.45 and time.monotonic() - started >= 0.45
    assert gate.wait_idle(stop, grace=0.0) < 0.1
