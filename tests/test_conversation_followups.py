import json

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import conversation
from regulatory_graph.runtime import (
    GraphRetrievalResult,
    GraphRetrievalStatus,
    GraphRetrievalTrace,
)


def _document(filename="Cir_2019_07_fr.pdf", page=2):
    return Document(
        page_content="verified regulatory evidence",
        metadata={"source": filename, "page": page},
    )


def _previous_state():
    return {
        "topics": ["Circular 2019-07"],
        "first_topic": "Circular 2019-07",
        "current_topic": "Circular 2019-07",
        "turns": [
            {
                "user_message": "What did Circular 2019-07 change?",
                "standalone_query": "changes made by Circular 2019-07",
                "answer": "It amended the exchange-office rules.",
                "sources": [{"file": "Cir_2019_07_fr.pdf", "page": 3}],
                "graph_trace": {"status": "EXPANDED"},
            }
        ],
    }


def test_route_message_validates_a_follow_up_rewrite_against_memory():
    response = {
        "intent": "FOLLOW_UP",
        "rewrite_query": "current deadline under Circular 2019-07",
        "new_topic": None,
        "current_topic": "Circular 2019-07",
    }
    llm = FakeListChatModel(responses=[json.dumps(response)])

    route = conversation.route_message(llm, "What about the deadline?", _previous_state())

    assert route == response


def test_route_message_fails_closed_on_malformed_output():
    llm = FakeListChatModel(responses=["not valid json"])

    route = conversation.route_message(llm, "What about that one?", _previous_state())

    assert route["intent"] == "AMBIGUOUS"
    assert route["rewrite_query"] is None


def test_route_message_fails_closed_when_multiple_topics_have_no_current_topic():
    response = {
        "intent": "FOLLOW_UP",
        "rewrite_query": "content of Circular 2019-07",
        "new_topic": None,
        "current_topic": "Circular 2019-07",
    }
    llm = FakeListChatModel(responses=[json.dumps(response)])
    memory = {
        **_previous_state(),
        "topics": ["Circular 2019-07", "Circular 2025-17"],
        "current_topic": None,
    }

    route = conversation.route_message(llm, "What about that one?", memory)

    assert route["intent"] == "AMBIGUOUS"


def test_follow_up_uses_standalone_query_for_dense_bm25_and_graph(monkeypatch):
    rewritten = "relationship between Circular 2019-07 and Circular 2018-07"
    ordinary = _document()
    calls = {}

    class GraphRetriever:
        def retrieve(self, query, seed_documents):
            calls["graph_query"] = query
            calls["seeds"] = tuple(seed_documents)
            return GraphRetrievalResult(
                documents=(),
                trace=GraphRetrievalTrace(status=GraphRetrievalStatus.NO_EVIDENCE),
            )

    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "FOLLOW_UP",
            "rewrite_query": rewritten,
            "new_topic": None,
            "current_topic": "Circular 2019-07",
        },
    )
    monkeypatch.setattr(
        conversation,
        "retrieve_relevant_chunks",
        lambda query, _store: calls.setdefault("dense_query", query) and [ordinary],
    )
    monkeypatch.setattr(
        conversation,
        "retrieve_bm25",
        lambda query, _bm25, _documents: calls.setdefault("bm25_query", query) and [],
    )
    monkeypatch.setattr(
        conversation,
        "score_documents",
        lambda _reranker, query, documents: (
            calls.setdefault("reranker_query", query)
            and [(document, 1) for document in documents]
        ),
    )

    def answer(_llm, message, _documents, memory):
        calls["answer_query"] = message
        calls["answer_memory"] = memory
        return "follow-up answer"

    monkeypatch.setattr(conversation, "generate_answer", answer)

    result = conversation.chat(
        "How is it related to the previous circular?",
        _previous_state(),
        vector_store=object(),
        reranker=object(),
        bm25=object(),
        bm25_documents=[],
        graph_retriever=GraphRetriever(),
    )

    assert calls["dense_query"] == rewritten
    assert calls["bm25_query"] == rewritten
    assert calls["reranker_query"] == rewritten
    assert calls["graph_query"] == rewritten
    assert calls["answer_query"] == rewritten
    assert "What did Circular 2019-07 change?" in calls["answer_memory"]
    assert result["memory_state"]["turns"][-1]["standalone_query"] == rewritten
    assert result["memory_state"]["turns"][-1]["answer"] == "follow-up answer"


def test_new_topic_does_not_leak_old_turns_into_answer_memory(monkeypatch):
    ordinary = _document("Cir_2025_17_fr.pdf", page=3)
    captured = {}
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "NEW_TOPIC",
            "rewrite_query": "reporting under Circular 2025-17",
            "new_topic": "Circular 2025-17",
            "current_topic": "Circular 2025-17",
        },
    )
    monkeypatch.setattr(conversation, "retrieve_relevant_chunks", lambda *_: [ordinary])
    monkeypatch.setattr(conversation, "retrieve_bm25", lambda *_: [])
    monkeypatch.setattr(
        conversation,
        "score_documents",
        lambda _reranker, _query, documents: [(document, 1) for document in documents],
    )

    def answer(_llm, _message, _documents, memory):
        captured["memory"] = memory
        return "new-topic answer"

    monkeypatch.setattr(conversation, "generate_answer", answer)

    result = conversation.chat(
        "Now tell me about Circular 2025-17.",
        _previous_state(),
        vector_store=object(),
        reranker=object(),
        bm25=object(),
        bm25_documents=[],
    )

    assert "What did Circular 2019-07 change?" not in captured["memory"]
    assert "Circular 2025-17" in captured["memory"]
    assert result["memory_state"]["current_topic"] == "Circular 2025-17"


def test_ambiguous_reference_asks_for_clarification_without_retrieval(monkeypatch):
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "AMBIGUOUS",
            "rewrite_query": None,
            "new_topic": None,
            "current_topic": None,
        },
    )
    monkeypatch.setattr(
        conversation,
        "retrieve_relevant_chunks",
        lambda *_: (_ for _ in ()).throw(AssertionError("retrieval must not run")),
    )

    result = conversation.chat(
        "What about that one?",
        {
            **_previous_state(),
            "topics": ["Circular 2019-07", "Circular 2025-17"],
            "current_topic": None,
        },
        vector_store=object(),
        reranker=object(),
        bm25=object(),
        bm25_documents=[],
    )

    assert "which" in result["answer"].casefold()
    assert result["sources"] == []
    assert result["memory_state"]["turns"] == _previous_state()["turns"]
