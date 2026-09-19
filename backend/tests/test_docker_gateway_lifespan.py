"""Gateway must run the mounted API lifespan (auth bootstrap, graph, etc.)."""

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient


def test_docker_gateway_bootstraps_admin_via_lifespan(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    monkeypatch.setenv("BCT_CONVERSATION_DB", str(tmp_path / "conversations.sqlite3"))
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_EMAIL", "admin@example.tn")
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_PASSWORD", "changeme1")
    monkeypatch.setenv("BCT_ENABLE_GRAPH", "0")
    monkeypatch.setenv("BCT_DEFAULT_PROFILE", "cloud")
    monkeypatch.setenv("BCT_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("BCT_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("BCT_DOCUMENTS_DIR", str(tmp_path / "documents"))

    # Avoid loading real voyage/cloud backends during lifespan profile init.
    import app as app_module

    class FakeManager:
        def get(self, _profile):
            raise RuntimeError("not needed for auth bootstrap")

    monkeypatch.setattr(app_module, "create_runtime_profile_manager", lambda: FakeManager())
    monkeypatch.setattr(app_module, "open_relationship_graph_runtime", lambda: None)

    import docker_serve

    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html></html>", encoding="utf-8")

    gateway = docker_serve.build_gateway(static)
    with TestClient(gateway) as client:
        response = client.post(
            "/api/auth/login",
            json={"email": "admin@example.tn", "password": "changeme1"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["user"]["role"] == "admin"
