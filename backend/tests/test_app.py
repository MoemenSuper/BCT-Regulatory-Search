from fastapi.testclient import TestClient
from types import SimpleNamespace
import pytest
import app as app_module
from app import app
from conversation_memory import ConversationStore


client = TestClient(app)


class _FakeTitleLLM:
    def __init__(self, reply="Dépôt à distance d'une demande de change"):
        self.reply = reply

    def invoke(self, _prompt):
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(content=self.reply)


@pytest.fixture(autouse=True)
def _offline_title_llm(monkeypatch):
    """Conversation titles come from an LLM call; never reach Groq from tests."""
    monkeypatch.setattr(app_module, "create_llm", lambda provider="groq": _FakeTitleLLM())
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert set(payload) == {
        "status",
        "graph_enabled",
        "neo4j_connected",
        "graph_ready",
    }
    if payload["graph_ready"]:
        assert payload["graph_enabled"] is True
        assert payload["neo4j_connected"] is True


def test_health_reports_graph_lite_not_ready_when_enabled_but_unavailable(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("BCT_ENABLE_GRAPH", "1")
    monkeypatch.setenv("BCT_CONVERSATION_DB", str(tmp_path / "conversations.sqlite3"))
    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setattr(app_module, "open_relationship_graph_runtime", lambda: None)
    monkeypatch.setattr(
        app_module,
        "open_conversation_store",
        lambda: ConversationStore(tmp_path / "conversations.sqlite3"),
    )

    with TestClient(app_module.app) as live_client:
        payload = live_client.get("/health").json()

    assert payload == {
        "status": "ok",
        "graph_enabled": True,
        "neo4j_connected": False,
        "graph_ready": False,
    }
    announced = capsys.readouterr().out
    assert "graph_enabled: True" in announced
    assert "neo4j_connected: False" in announced
    assert "graph_ready: False" in announced


def test_health_reports_graph_ready_when_runtime_is_attached(
    monkeypatch, tmp_path, capsys
):
    class FakeDriver:
        def verify_connectivity(self):
            return None

        def close(self):
            return None

    class FakeGraphRuntime:
        def __init__(self):
            self.retriever = object()
            self.driver = FakeDriver()

        def close(self):
            self.driver.close()

    monkeypatch.setenv("BCT_ENABLE_GRAPH", "1")
    monkeypatch.setenv("BCT_CONVERSATION_DB", str(tmp_path / "conversations.sqlite3"))
    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setattr(
        app_module, "open_relationship_graph_runtime", lambda: FakeGraphRuntime()
    )
    monkeypatch.setattr(
        app_module,
        "open_conversation_store",
        lambda: ConversationStore(tmp_path / "conversations.sqlite3"),
    )

    with TestClient(app_module.app) as live_client:
        payload = live_client.get("/health").json()

    assert payload == {
        "status": "ok",
        "graph_enabled": True,
        "neo4j_connected": True,
        "graph_ready": True,
    }
    announced = capsys.readouterr().out
    assert "graph_enabled: True" in announced
    assert "graph_ready: True" in announced


def test_profiles_exposes_the_three_runtime_choices():
    response = client.get("/profiles")

    assert response.status_code == 200
    assert {item["value"] for item in response.json()} == {
        "local",
        "local_hybrid",
        "cloud",
    }

def test_chat_rejects_empty_question():
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422


def test_lifespan_passes_relationship_graph_to_chat_and_closes_driver(
    monkeypatch, tmp_path
):
    class FakeGraphRuntime:
        def __init__(self):
            self.closed = False
            self.retriever = object()

        def close(self):
            self.closed = True

    fake_runtime = FakeGraphRuntime()
    conversation_store = ConversationStore(tmp_path / "conversations.sqlite3")
    received = {}
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setenv("BCT_ENABLE_GRAPH", "1")
    monkeypatch.setattr(
        app_module,
        "open_relationship_graph_runtime",
        lambda: fake_runtime,
    )
    monkeypatch.setattr(
        app_module,
        "open_conversation_store",
        lambda: conversation_store,
    )

    def fake_chat(*_args, graph_retriever=None, **_kwargs):
        received["graph_retriever"] = graph_retriever
        return {
            "answer": "answer",
            "sources": [],
            "memory_state": {},
            "graph_trace": {"status": "NOT_REQUESTED"},
        }

    monkeypatch.setattr(app_module, "chat", fake_chat)

    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    with TestClient(app_module.app) as live_client:
        response = live_client.post("/chat", json={"question": "Bonjour"})

    assert response.status_code == 200
    assert received["graph_retriever"] is fake_runtime.retriever
    assert fake_runtime.closed is True


def test_chat_uses_application_active_runtime_profile(monkeypatch, tmp_path):
    class FakeGraphRuntime:
        retriever = None

        def close(self):
            return None

    selected_backend = object()
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setenv("BCT_ENABLE_GRAPH", "1")
    monkeypatch.setattr(
        app_module,
        "open_relationship_graph_runtime",
        lambda: FakeGraphRuntime(),
    )
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: store)

    class Manager:
        def get(self, profile):
            assert profile == "local"
            return SimpleNamespace(
                retrieval_backend=selected_backend,
                answer_provider="ollama",
                spec=SimpleNamespace(value=SimpleNamespace(value="local")),
            )

        def reset(self):
            return None

    monkeypatch.setattr(
        app_module,
        "create_runtime_profile_manager",
        lambda *_args, **_kwargs: Manager(),
    )
    received = {}

    def fake_chat(*_args, **kwargs):
        received.update(kwargs)
        return {
            "answer": "answer",
            "sources": [],
            "memory_state": {},
            "graph_trace": {"status": "NOT_REQUESTED"},
        }

    monkeypatch.setattr(app_module, "chat", fake_chat)

    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    with TestClient(app_module.app) as live_client:
        live_client.app.state.settings_store.set_active_profile("local")
        response = live_client.post(
            "/chat",
            json={"question": "Question", "profile": "cloud"},
        )

    assert response.status_code == 200
    assert response.json()["profile"] == "local"
    assert received["retrieval_backend"] is selected_backend
    assert received["llm_provider"] == "ollama"


def test_chat_creates_and_resumes_a_persistent_conversation(monkeypatch, tmp_path):
    class FakeGraphRuntime:
        retriever = None

        def close(self):
            return None

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    received_states = []
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setenv("BCT_ENABLE_GRAPH", "1")
    monkeypatch.setattr(
        app_module,
        "open_relationship_graph_runtime",
        lambda: FakeGraphRuntime(),
    )
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: store)

    def fake_chat(message, memory_state, *_args, **_kwargs):
        received_states.append(memory_state)
        next_state = {
            **memory_state,
            "topics": ["Circular 2019-07"],
            "first_topic": "Circular 2019-07",
            "current_topic": "Circular 2019-07",
            "turns": [
                *memory_state.get("turns", []),
                {
                    "user_message": message,
                    "standalone_query": message,
                    "answer": f"answer {len(received_states)}",
                    "sources": [],
                    "graph_trace": {"status": "NOT_REQUESTED"},
                },
            ],
        }
        return {
            "answer": f"answer {len(received_states)}",
            "sources": [],
            "memory_state": next_state,
            "graph_trace": {"status": "NOT_REQUESTED"},
        }

    monkeypatch.setattr(app_module, "chat", fake_chat)

    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    with TestClient(app_module.app) as live_client:
        first = live_client.post("/chat", json={"question": "First question"})
        conversation_id = first.json()["conversation_id"]
        second = live_client.post(
            "/chat",
            json={
                "question": "What about its deadline?",
                "conversation_id": conversation_id,
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id
    assert second.json()["memory_state"]["turns"][-1]["user_message"] == (
        "What about its deadline?"
    )
    assert received_states[0]["turns"] == []
    assert received_states[1]["turns"][0]["user_message"] == "First question"
    assert store.load(conversation_id, user_id="test-user")["turns"][-1]["user_message"] == (
        "What about its deadline?"
    )


def test_chat_rejects_an_unknown_conversation_id(monkeypatch, tmp_path):
    class FakeGraphRuntime:
        retriever = None

        def close(self):
            return None

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setenv("BCT_ENABLE_GRAPH", "1")
    monkeypatch.setattr(
        app_module,
        "open_relationship_graph_runtime",
        lambda: FakeGraphRuntime(),
    )
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: store)
    monkeypatch.setattr(
        app_module,
        "chat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("chat must not run")
        ),
    )

    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    with TestClient(app_module.app) as live_client:
        response = live_client.post(
            "/chat",
            json={"question": "Follow-up", "conversation_id": "missing"},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_sub_questions_splits_only_on_question_marks():
    split = app_module._sub_questions
    assert split("Quel est le plafond ?") == ["Quel est le plafond ?"]
    assert split("Quel est le plafond ? Et quelle est la durée du crédit ?") == [
        "Quel est le plafond ?",
        "Et quelle est la durée du crédit ?",
    ]
    assert split("ما هو السقف؟ وما هي المدة القصوى للقرض؟") == [
        "ما هو السقف؟",
        "وما هي المدة القصوى للقرض؟",
    ]
    # Short fragments and "A et B ?" without a second "?" stay one query.
    assert split("Quel est le plafond et la durée ?") == ["Quel est le plafond et la durée ?"]
    assert split("Plafond ? Ok ? Et la durée maximale ?") == ["Et la durée maximale ?"]
    assert len(split(" ".join(f"Question numéro {i} sur le plafond ?" for i in range(5)))) == 3


def test_chat_runs_two_questions_as_two_turns_sharing_memory(monkeypatch, tmp_path):
    class FakeGraphRuntime:
        retriever = None

        def close(self):
            return None

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    calls = []
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setenv("BCT_ENABLE_GRAPH", "1")
    monkeypatch.setattr(app_module, "open_relationship_graph_runtime", lambda: FakeGraphRuntime())
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: store)

    def fake_chat(message, memory_state, *_args, **_kwargs):
        calls.append((message, list(memory_state.get("turns", []))))
        turn = {"user_message": message, "standalone_query": message, "answer": f"answer {len(calls)}",
                "sources": [], "graph_trace": {"status": "NOT_REQUESTED"}}
        return {
            "answer": f"answer {len(calls)}",
            "sources": [{"file": f"Cir_2019_0{len(calls)}_fr.pdf", "page": len(calls), "score": 1.0}],
            "status": "answered",
            "memory_state": {**memory_state, "turns": [*memory_state.get("turns", []), turn]},
            "graph_trace": {"status": "NOT_REQUESTED"},
        }

    monkeypatch.setattr(app_module, "chat", fake_chat)

    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    with TestClient(app_module.app) as live_client:
        response = live_client.post(
            "/chat",
            json={"question": "Quel est le plafond ? Quelle est la durée maximale ?"},
        )
        transcript = live_client.get(f"/conversations/{response.json()['conversation_id']}").json()

    assert response.status_code == 200
    assert [message for message, _turns in calls] == ["Quel est le plafond ?", "Quelle est la durée maximale ?"]
    # The second question sees the first as a previous turn (follow-up resolution).
    assert calls[1][1][0]["user_message"] == "Quel est le plafond ?"
    # Each question is its own transcript turn with its own answer and sources.
    assert [turn["question"] for turn in transcript["turns"]] == [
        "Quel est le plafond ?",
        "Quelle est la durée maximale ?",
    ]
    assert [turn["answer"] for turn in transcript["turns"]] == ["answer 1", "answer 2"]
    assert transcript["turns"][1]["sources"][0]["file"] == "Cir_2019_02_fr.pdf"
    # The HTTP body carries the last turn; the UI refetches the transcript anyway.
    assert response.json()["answer"] == "answer 2"


def test_conversation_rename_and_delete_endpoints(monkeypatch, tmp_path):
    import app as app_module
    from conversation_memory import ConversationStore, new_memory_state

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: store)
    monkeypatch.setattr(app_module, "create_runtime_profile_manager", lambda: None)
    monkeypatch.setattr(app_module, "_graph_enabled", lambda: False)
    conversation_id = store.create("test-user")
    store.save_with_turn(
        conversation_id,
        new_memory_state(),
        user_id="test-user",
        question="Question ?",
        answer="A",
    )

    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    with TestClient(app_module.app) as live_client:
        assert live_client.patch(f"/conversations/{conversation_id}", json={"title": "   "}).status_code == 400
        renamed = live_client.patch(f"/conversations/{conversation_id}", json={"title": " Mon titre "})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Mon titre"
        assert live_client.get("/conversations").json()[0]["title"] == "Mon titre"
        assert live_client.patch("/conversations/missing", json={"title": "x"}).status_code == 404

        assert live_client.delete(f"/conversations/{conversation_id}").status_code == 204
        assert live_client.get(f"/conversations/{conversation_id}").status_code == 404
        assert live_client.get("/conversations").json() == []
        assert live_client.delete(f"/conversations/{conversation_id}").status_code == 404


def _title_setup(monkeypatch, tmp_path):
    from conversation_memory import new_memory_state

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: store)
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setattr(app_module, "_graph_enabled", lambda: False)

    def fake_chat(message, memory_state, *_args, **_kwargs):
        state = {**new_memory_state(), "turns": [*memory_state.get("turns", []), {"user_message": message, "standalone_query": message, "answer": "a", "sources": []}]}
        return {"answer": "a", "sources": [], "memory_state": state, "graph_trace": {}}

    monkeypatch.setattr(app_module, "chat", fake_chat)
    return store


def test_new_conversation_title_comes_from_the_llm(monkeypatch, tmp_path):
    store = _title_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "create_llm", lambda provider="groq": _FakeTitleLLM(' "Dépôt à distance d’une demande de change." '))
    question = "Comment une personne ou une entreprise peut-elle déposer à distance une demande d’autorisation pour une opération de change ?"

    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    with TestClient(app_module.app) as live_client:
        first = live_client.post("/chat", json={"question": question}).json()
        live_client.post("/chat", json={"question": "Et pour une banque ?", "conversation_id": first["conversation_id"]})

    assert store.transcript(first["conversation_id"], user_id="test-user")["title"] == "Dépôt à distance d’une demande de change."
    assert store.list_conversations(user_id="test-user")[0]["title"] != question


def test_conversation_title_falls_back_when_llm_fails(monkeypatch, tmp_path):
    store = _title_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "create_llm", lambda provider="groq": _FakeTitleLLM(RuntimeError("down")))

    monkeypatch.setenv("BCT_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("BCT_SETTINGS_DB", str(tmp_path / "settings.sqlite3"))
    with TestClient(app_module.app) as live_client:
        first = live_client.post("/chat", json={"question": "Quelles sont les heures d'ouverture des guichets ?"}).json()

    assert store.transcript(first["conversation_id"], user_id="test-user")["title"] == "Heures d'ouverture des guichets"
