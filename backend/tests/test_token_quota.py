from fastapi.testclient import TestClient
import pytest

import app as app_module
from app import app
from conversation_memory import ConversationStore
from identity import require_admin, require_approved_user, require_user
from runtime_retrieval import track_cloud_retrieval_usage, _record_voyage_usage


@pytest.fixture()
def auth_client(tmp_path, monkeypatch):
    auth_db = tmp_path / "auth.sqlite3"
    settings_db = tmp_path / "settings.sqlite3"
    conversations_db = tmp_path / "conversations.sqlite3"
    monkeypatch.setenv("BCT_AUTH_DB", str(auth_db))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(settings_db))
    monkeypatch.setenv("BCT_CONVERSATION_DB", str(conversations_db))
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_EMAIL", "admin@bct.tn")
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_PASSWORD", "AdminPass123")
    monkeypatch.setenv("BCT_DEFAULT_TOKEN_LIMIT", "1000")
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setattr(
        app_module,
        "open_conversation_store",
        lambda: ConversationStore(conversations_db),
    )
    app.dependency_overrides.pop(require_user, None)
    app.dependency_overrides.pop(require_approved_user, None)
    app.dependency_overrides.pop(require_admin, None)
    with TestClient(app) as client:
        yield client


def test_admin_can_set_and_reset_token_quota(auth_client):
    registered = auth_client.post(
        "/auth/register",
        json={"email": "quota@bct.tn", "password": "Password123"},
    ).json()["user"]
    auth_client.post("/auth/login", json={"email": "admin@bct.tn", "password": "AdminPass123"})
    auth_client.post(f"/admin/users/{registered['id']}/approve")
    limited = auth_client.put(
        f"/admin/users/{registered['id']}/token-limit",
        json={"token_limit": 250},
    )
    assert limited.status_code == 200
    assert limited.json()["user"]["token_limit"] == 250
    auth_client.app.state.auth_store.consume_cloud_usage(
        registered["id"], llm=40, embed=10, rerank=5
    )
    users = auth_client.get("/admin/users").json()
    row = next(item for item in users if item["id"] == registered["id"])
    assert row["tokens_llm"] == 40
    assert row["tokens_embed"] == 10
    assert row["tokens_rerank"] == 5
    assert row["tokens_used"] == 55
    assert row["tokens_remaining"] == 195
    reset = auth_client.post(f"/admin/users/{registered['id']}/reset-tokens")
    assert reset.status_code == 200
    assert reset.json()["user"]["tokens_used"] == 0
    assert reset.json()["user"]["tokens_llm"] == 0


def test_chat_blocked_when_token_quota_exhausted(auth_client, monkeypatch):
    registered = auth_client.post(
        "/auth/register",
        json={"email": "spent@bct.tn", "password": "Password123"},
    ).json()["user"]
    auth_client.post("/auth/login", json={"email": "admin@bct.tn", "password": "AdminPass123"})
    auth_client.post(f"/admin/users/{registered['id']}/approve")
    auth_client.put(f"/admin/users/{registered['id']}/token-limit", json={"token_limit": 10})
    auth_client.app.state.auth_store.consume_tokens(registered["id"], 10)
    auth_client.post("/auth/logout")
    auth_client.post("/auth/login", json={"email": "spent@bct.tn", "password": "Password123"})
    monkeypatch.setattr(
        app_module,
        "chat",
        lambda *args, **kwargs: {
            "answer": "x",
            "sources": [],
            "memory_state": {},
            "graph_trace": {},
            "status": "answered",
        },
    )
    denied = auth_client.post("/chat", json={"question": "Quelle est la circulaire applicable ?"})
    assert denied.status_code == 402


def test_billable_llm_tokens_skip_local_fallback():
    usage = {
        "openai/gpt-oss-120b": {
            "input_tokens": 1200,
            "output_tokens": 300,
            "total_tokens": 1500,
        }
    }
    assert app_module._tokens_from_usage(usage) == 1500
    assert app_module._billable_llm_tokens(usage, "groq", "hi", "bye") == 1500
    assert app_module._billable_llm_tokens({}, "groq", "abcd") == 1
    assert app_module._billable_llm_tokens({}, "ollama", "abcd") == 0


def test_voyage_usage_tracker_records_live_api_tokens():
    with track_cloud_retrieval_usage() as bucket:
        _record_voyage_usage("contextualizedembeddings", {"usage": {"total_tokens": 12}})
        _record_voyage_usage("rerank", {"usage": {"total_tokens": 7}})
        assert bucket.embed_tokens == 12
        assert bucket.rerank_tokens == 7
