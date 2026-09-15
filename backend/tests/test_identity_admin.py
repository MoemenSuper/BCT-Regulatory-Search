from fastapi.testclient import TestClient
import pytest

import app as app_module
from app import app
from app_settings import AppSettingsStore
from conversation_memory import ConversationStore
from identity import AuthStore, require_admin, require_approved_user, require_user


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
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setattr(app_module, "open_relationship_graph_runtime", lambda: None)
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


def test_register_creates_pending_user(auth_client):
    response = auth_client.post(
        "/auth/register",
        json={"email": "new.user@bct.tn", "password": "Password123"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user"]["role"] == "user"
    assert payload["user"]["status"] == "pending"


def test_pending_user_cannot_access_chat(auth_client):
    auth_client.post(
        "/auth/register",
        json={"email": "pending@bct.tn", "password": "Password123"},
    )
    login = auth_client.post(
        "/auth/login",
        json={"email": "pending@bct.tn", "password": "Password123"},
    )
    assert login.status_code == 200
    assert login.json()["user"]["status"] == "pending"
    denied = auth_client.post("/chat", json={"question": "Quelle est la circulaire?"})
    assert denied.status_code == 403


def test_admin_approves_user_and_user_can_list_conversations(auth_client):
    registered = auth_client.post(
        "/auth/register",
        json={"email": "analyst@bct.tn", "password": "Password123"},
    ).json()["user"]

    admin_login = auth_client.post(
        "/auth/login",
        json={"email": "admin@bct.tn", "password": "AdminPass123"},
    )
    assert admin_login.status_code == 200
    assert admin_login.json()["user"]["role"] == "admin"

    approve = auth_client.post(f"/admin/users/{registered['id']}/approve")
    assert approve.status_code == 200
    assert approve.json()["user"]["status"] == "approved"

    auth_client.post("/auth/logout")
    user_login = auth_client.post(
        "/auth/login",
        json={"email": "analyst@bct.tn", "password": "Password123"},
    )
    assert user_login.status_code == 200
    conversations = auth_client.get("/conversations")
    assert conversations.status_code == 200
    assert conversations.json() == []


def test_normal_user_cannot_call_admin_endpoints(auth_client):
    registered = auth_client.post(
        "/auth/register",
        json={"email": "normal@bct.tn", "password": "Password123"},
    ).json()["user"]
    auth_client.post(
        "/auth/login",
        json={"email": "admin@bct.tn", "password": "AdminPass123"},
    )
    auth_client.post(f"/admin/users/{registered['id']}/approve")
    auth_client.post("/auth/logout")
    auth_client.post(
        "/auth/login",
        json={"email": "normal@bct.tn", "password": "Password123"},
    )
    denied = auth_client.get("/admin/users")
    assert denied.status_code == 403
    denied_docs = auth_client.get("/documents")
    assert denied_docs.status_code == 403


def test_admin_can_switch_runtime_profile(auth_client):
    auth_client.post(
        "/auth/login",
        json={"email": "admin@bct.tn", "password": "AdminPass123"},
    )
    response = auth_client.put("/admin/config/profile", json={"profile": "local_hybrid"})
    assert response.status_code == 200
    assert response.json()["active_profile"] == "local_hybrid"
    config = auth_client.get("/admin/config").json()
    assert config["active_profile"] == "local_hybrid"


def test_admin_secrets_are_masked(auth_client):
    auth_client.post(
        "/auth/login",
        json={"email": "admin@bct.tn", "password": "AdminPass123"},
    )
    updated = auth_client.put(
        "/admin/config/secrets",
        json={"secrets": {"GROQ_API_KEY": "super-secret-key-value"}},
    )
    assert updated.status_code == 200
    secret = next(item for item in updated.json()["secrets"] if item["key"] == "GROQ_API_KEY")
    assert secret["configured"] is True
    assert "super-secret" not in (secret["masked"] or "")
    assert secret["masked"].startswith("sup")


def test_passwords_are_hashed(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    user = store.create_user(email="hash@bct.tn", password="Password123")
    row = store._conn.execute(
        "SELECT password_hash FROM users WHERE id = ?", (user.id,)
    ).fetchone()
    assert row is not None
    assert row[0] != "Password123"
    assert row[0].startswith("scrypt$")
    store.close()


def test_settings_store_masks_and_applies(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    store = AppSettingsStore(tmp_path / "settings.sqlite3")
    store.update_secrets({"GROQ_API_KEY": "abcd1234secret"})
    public = store.public_configuration()
    secret = next(item for item in public["secrets"] if item["key"] == "GROQ_API_KEY")
    assert secret["masked"] != "abcd1234secret"
    assert store.get("GROQ_API_KEY") == "abcd1234secret"
    store.close()


def test_admin_can_promote_approved_user_but_not_demote_or_delete_admins(auth_client):
    registered = auth_client.post(
        "/auth/register",
        json={"email": "future.admin@bct.tn", "password": "Password123"},
    ).json()["user"]
    auth_client.post(
        "/auth/login",
        json={"email": "admin@bct.tn", "password": "AdminPass123"},
    )
    pending_promote = auth_client.post(f"/admin/users/{registered['id']}/promote")
    assert pending_promote.status_code == 400

    auth_client.post(f"/admin/users/{registered['id']}/approve")
    promote = auth_client.post(f"/admin/users/{registered['id']}/promote")
    assert promote.status_code == 200
    assert promote.json()["user"]["role"] == "admin"

    # No demote route exists.
    demote = auth_client.post(f"/admin/users/{registered['id']}/demote")
    assert demote.status_code == 404

    blocked_delete = auth_client.delete(f"/admin/users/{registered['id']}")
    assert blocked_delete.status_code == 400
