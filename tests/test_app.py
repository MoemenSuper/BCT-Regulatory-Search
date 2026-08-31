from fastapi.testclient import TestClient
import app as app_module
from app import app


client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_rejects_empty_question():
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422


def test_lifespan_passes_relationship_graph_to_chat_and_closes_driver(monkeypatch):
    class FakeGraphRuntime:
        def __init__(self):
            self.closed = False
            self.retriever = object()

        def close(self):
            self.closed = True

    fake_runtime = FakeGraphRuntime()
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
