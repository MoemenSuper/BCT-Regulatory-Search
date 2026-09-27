"""Batch PDF removal reloads the search index once."""

from types import SimpleNamespace

import app as app_module
import ingestion.pipeline as pipeline_module
from fastapi import FastAPI
from fastapi.testclient import TestClient
from identity import require_admin


class _FakePipeline:
    removed: list[str] = []

    def __init__(self, _config):
        pass

    def remove(self, document_id):
        if document_id == "missing":
            raise KeyError("Ready document not found: missing")
        _FakePipeline.removed.append(document_id)
        return {}

    def close(self):
        pass


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("BCT_RUNTIME_ASSET_ROOT", str(tmp_path))
    monkeypatch.setattr(pipeline_module, "IngestionPipeline", _FakePipeline)
    monkeypatch.setattr(pipeline_module.IngestionConfig, "from_environment", classmethod(lambda cls: None))
    monkeypatch.setattr(app_module, "_refresh_runtime_asset_environment", lambda: None)
    _FakePipeline.removed = []
    resets: list[int] = []
    api = FastAPI()
    api.state.profile_manager = SimpleNamespace(reset=lambda: resets.append(1))
    api.state.source_resolver = SimpleNamespace(refresh=lambda: None)
    api.dependency_overrides[require_admin] = lambda: None
    app_module._install_ingestion_routes(api)
    return TestClient(api), resets


def test_batch_delete_removes_each_and_resets_once(monkeypatch, tmp_path):
    client, resets = _client(monkeypatch, tmp_path)
    response = client.post("/documents/delete", json={"document_ids": ["a" * 64, "missing", "b" * 64, "a" * 64]})
    assert response.status_code == 200
    body = response.json()
    assert body["removed"] == ["a" * 64, "b" * 64]
    assert [item["document_id"] for item in body["failed"]] == ["missing"]
    assert _FakePipeline.removed == ["a" * 64, "b" * 64]
    assert len(resets) == 1


def test_batch_delete_skips_reset_when_nothing_removed(monkeypatch, tmp_path):
    client, resets = _client(monkeypatch, tmp_path)
    response = client.post("/documents/delete", json={"document_ids": ["missing"]})
    assert response.status_code == 200
    assert response.json()["removed"] == []
    assert resets == []
