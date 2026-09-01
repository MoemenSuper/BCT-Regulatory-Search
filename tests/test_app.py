from fastapi.testclient import TestClient
import app as app_module
from app import app
from conversation_memory import ConversationStore


client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

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
    monkeypatch.setattr(app_module, "create_embedding_model", lambda: object())
    monkeypatch.setattr(app_module, "load_vector_store", lambda _model: object())
    monkeypatch.setattr(app_module, "create_reranker", lambda: object())
    monkeypatch.setattr(app_module, "load_documents_from_chroma", lambda _store: [])
    monkeypatch.setattr(app_module, "create_bm25", lambda _documents: object())
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

    with TestClient(app_module.app) as live_client:
        response = live_client.post("/chat", json={"question": "Bonjour"})

    assert response.status_code == 200
    assert received["graph_retriever"] is fake_runtime.retriever
    assert fake_runtime.closed is True


def test_chat_creates_and_resumes_a_persistent_conversation(monkeypatch, tmp_path):
    class FakeGraphRuntime:
        retriever = None

        def close(self):
            return None

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    received_states = []
    monkeypatch.setattr(app_module, "create_embedding_model", lambda: object())
    monkeypatch.setattr(app_module, "load_vector_store", lambda _model: object())
    monkeypatch.setattr(app_module, "create_reranker", lambda: object())
    monkeypatch.setattr(app_module, "load_documents_from_chroma", lambda _store: [])
    monkeypatch.setattr(app_module, "create_bm25", lambda _documents: object())
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
    assert received_states[0]["turns"] == []
    assert received_states[1]["turns"][0]["user_message"] == "First question"
    assert store.load(conversation_id)["turns"][-1]["user_message"] == (
        "What about its deadline?"
    )


def test_chat_rejects_an_unknown_conversation_id(monkeypatch, tmp_path):
    class FakeGraphRuntime:
        retriever = None

        def close(self):
            return None

    store = ConversationStore(tmp_path / "conversations.sqlite3")
    monkeypatch.setattr(app_module, "create_embedding_model", lambda: object())
    monkeypatch.setattr(app_module, "load_vector_store", lambda _model: object())
    monkeypatch.setattr(app_module, "create_reranker", lambda: object())
    monkeypatch.setattr(app_module, "load_documents_from_chroma", lambda _store: [])
    monkeypatch.setattr(app_module, "create_bm25", lambda _documents: object())
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

    with TestClient(app_module.app) as live_client:
        response = live_client.post(
            "/chat",
            json={"question": "Follow-up", "conversation_id": "missing"},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."
