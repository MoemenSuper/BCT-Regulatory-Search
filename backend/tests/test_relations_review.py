"""Relations between texts (supersession edges): found ones are used at once and listed "to review";
an administrator approves, rejects or adds one, and the decision survives a restart."""
import json
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

import app as app_module
from extras import supersession_review as review
from app import app
from extras.conversation_memory import ConversationStore
from extras.identity import require_admin, require_approved_user, require_user
from extras.supersession_edges import SupersessionEdge

FOUND = SupersessionEdge("cir:2018:13", "Cir_2018_13_fr.pdf", 2, "ABROGATE", "cir:2017:9", None,
                         "Est abrogée la circulaire n° 2017-09 du 27 octobre 2017.")
WRONG = SupersessionEdge("cir:2019:7", "Cir_2019_07_fr.pdf", 1, "REPLACE", "cir:2016:8", None,
                         "La présente circulaire annule et remplace la circulaire n° 2016-08.")


@pytest.fixture()
def database(tmp_path, monkeypatch):
    monkeypatch.setenv("BCT_INGESTION_DB", str(tmp_path / "ingestion.sqlite3"))
    review._cache = None
    yield tmp_path


def test_a_rejected_relation_is_not_used_and_an_added_one_is(database):
    added = SupersessionEdge("cir:2024:1", "Cir_2024_01_fr.pdf", 6, "REPLACE", "cir:2021:1", "3", "…")
    assert review.effective_edges([FOUND, WRONG]) == [FOUND, WRONG]
    review.decide(WRONG, "rejected", "admin@bct.tn")
    review.decide(added, "added", "admin@bct.tn")
    assert review.effective_edges([FOUND, WRONG]) == [FOUND, added]

    review._cache = None  # a restart reads the decisions back from the database
    statuses = {row["source_file"]: row["status"] for row in review.listing([FOUND, WRONG])}
    assert statuses == {"Cir_2018_13_fr.pdf": "review", "Cir_2019_07_fr.pdf": "rejected", "Cir_2024_01_fr.pdf": "added"}

    review.decide(WRONG, None, "admin@bct.tn")  # back to "to review": used again
    assert WRONG in review.effective_edges([FOUND, WRONG])


@pytest.fixture()
def admin_client(database, monkeypatch):
    version = database / "assets" / "versions" / "v1"
    version.mkdir(parents=True)
    page = "Décide :\nArticle premier : Est abrogée la circulaire n° 2017-09 du 27 octobre 2017. Article 2 : fin."
    rows = [{"page_content": page, "metadata": {"source": "Cir_2018_13_fr.pdf", "page": 2, "pages": [2]}},
            {"page_content": "Conditions de financement.", "metadata": {"source": "Cir_2017_09_fr.pdf", "page": 1, "pages": [1]}}]
    (version / "native.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (version / "supersession_edges.jsonl").write_text(json.dumps(asdict(WRONG), ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setenv("BCT_NATIVE_CHUNKS_PATH", str(version / "native.jsonl"))
    monkeypatch.setenv("BCT_AUTH_DB", str(database / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(database / "settings.sqlite3"))
    monkeypatch.setenv("BCT_CONVERSATION_DB", str(database / "conversations.sqlite3"))
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_EMAIL", "admin@bct.tn")
    monkeypatch.setenv("BCT_BOOTSTRAP_ADMIN_PASSWORD", "AdminPass123")
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: ConversationStore(database / "conversations.sqlite3"))
    for dependency in (require_user, require_approved_user, require_admin):
        app.dependency_overrides.pop(dependency, None)
    with TestClient(app) as client:
        client.post("/auth/login", json={"email": "admin@bct.tn", "password": "AdminPass123"})
        yield client


def test_the_admin_reviews_and_adds_relations_from_the_admin_page(admin_client):
    [found] = admin_client.get("/admin/relations").json()["items"]
    assert found["status"] == "review" and found["source_page"] == 1

    rejected = admin_client.post(f"/admin/relations/{found['id']}/decision", json={"status": "rejected"}).json()
    assert rejected["status"] == "rejected" and rejected["decided_by"] == "admin@bct.tn"

    # A page that does not name the old text cannot be the proof of a relation.
    refused = admin_client.post("/admin/relations", json={
        "source_file": "Cir_2018_13_fr.pdf", "source_page": 2, "action": "ABROGATE", "target": "Cir 2016-08"})
    assert refused.status_code == 400 and "does not mention" in refused.json()["detail"]

    added = admin_client.post("/admin/relations", json={
        "source_file": "Cir_2018_13_fr.pdf", "source_page": 2, "action": "ABROGATE", "target": "Cir 2017-09"}).json()
    assert added["status"] == "added" and added["target_instrument"] == "cir:2017:9"
    assert added["quote"] == "Article premier : Est abrogée la circulaire n° 2017-09 du 27 octobre 2017."

    listed = {item["id"]: item for item in admin_client.get("/admin/relations").json()["items"]}
    assert listed[added["id"]]["target_file"] == "Cir_2017_09_fr.pdf"  # the old text can be opened too
    assert admin_client.post(f"/admin/relations/{added['id']}/decision", json={"status": "review"}).json() == {"removed": True}
    assert added["id"] not in {item["id"] for item in admin_client.get("/admin/relations").json()["items"]}
