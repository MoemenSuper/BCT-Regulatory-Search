"""Answer-model outages must surface as outages, never as refusals or 'out of scope'."""
import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIError
from langchain_core.runnables import RunnableLambda

import app as app_module
import conversation
from app import app
from conversation_memory import ConversationStore, new_memory_state
from identity import require_admin, require_approved_user, require_user


def _outage(*_args, **_kwargs):
    raise APIError("Request too large", request=httpx.Request("POST", "https://api.groq.com"), body=None)


def test_general_chat_reply_does_not_turn_an_outage_into_out_of_scope():
    with pytest.raises(APIError):
        conversation.general_chat_reply(RunnableLambda(_outage), "Bonjour", new_memory_state())


def test_chat_endpoint_reports_provider_outage_as_503(tmp_path, monkeypatch):
    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_EMAIL", "admin@bct.tn")
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_PASSWORD", "AdminPass123")
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: ConversationStore(tmp_path / "c.sqlite3"))
    monkeypatch.setattr(app_module, "chat", _outage)
    for dependency in (require_user, require_approved_user, require_admin):
        app.dependency_overrides.pop(dependency, None)
    with TestClient(app) as client:
        client.post("/auth/login", json={"email": "admin@bct.tn", "password": "AdminPass123"})
        response = client.post("/chat", json={"question": "Quel est le taux directeur ?"})
    assert response.status_code == 503
    assert "indisponible" in response.json()["detail"]
