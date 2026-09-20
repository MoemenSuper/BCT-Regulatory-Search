from fastapi.testclient import TestClient
import app as app_module
import subprocess
import sys


def test_cloud_startup_and_health_do_not_load_local_models(monkeypatch, tmp_path):
    monkeypatch.setenv("BCT_DEFAULT_PROFILE", "cloud")
    monkeypatch.setenv("BCT_CONVERSATION_DB", str(tmp_path / "conversations.sqlite3"))
    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    def forbidden():
        raise AssertionError("Cloud startup must not load local models")
    monkeypatch.setattr(app_module, "create_local_backend", forbidden)
    with TestClient(app_module.app) as client:
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert set(payload["supersession"]) == {"ready", "edge_count"}


def test_cloud_import_does_not_require_graph_ocr_or_local_model_packages():
    code = """
import sys
import app
for name in ('regulatory_graph', 'neo4j', 'langchain_chroma', 'sentence_transformers', 'docling'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
