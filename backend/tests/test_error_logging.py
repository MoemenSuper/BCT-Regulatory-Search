"""On a server, every problem must leave a log line that points to it: an unexpected error gives
the user a reference found in the log, and a missing setting is named at startup."""
import logging

from fastapi.testclient import TestClient

import app as app_module
from app import app
from conversation_memory import ConversationStore
from identity import require_admin, require_approved_user, require_user
from llm import ANSWER_PROVIDERS


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_EMAIL", "admin@bct.tn")
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_PASSWORD", "AdminPass123")
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: ConversationStore(tmp_path / "c.sqlite3"))
    for dependency in (require_user, require_approved_user, require_admin):
        app.dependency_overrides.pop(dependency, None)
    return TestClient(app)


def test_an_unexpected_error_gives_the_user_a_reference_found_in_the_log(tmp_path, monkeypatch, caplog):
    def broken(*_args, **_kwargs):
        raise KeyError("missing field")

    monkeypatch.setattr(app_module, "chat", broken)
    with _client(tmp_path, monkeypatch) as client, caplog.at_level(logging.ERROR):
        client.post("/auth/login", json={"email": "admin@bct.tn", "password": "AdminPass123"})
        response = client.post("/chat", json={"question": "Quel est le taux directeur ?"})
    assert response.status_code == 500
    reference = response.json()["detail"].split("reference ")[1].rstrip(").")
    [record] = [r for r in caplog.records if reference in r.getMessage()]
    assert "POST /chat" in record.getMessage() and "missing field" in caplog.text  # with its traceback


def test_startup_names_a_missing_answer_model_key(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("BCT_ANSWER_PROVIDER", raising=False)
    for key, _model, _default in ANSWER_PROVIDERS.values():
        monkeypatch.delenv(key, raising=False)
    with caplog.at_level(logging.WARNING), _client(tmp_path, monkeypatch):
        pass
    assert "Startup check: no key for the answer model" in caplog.text
