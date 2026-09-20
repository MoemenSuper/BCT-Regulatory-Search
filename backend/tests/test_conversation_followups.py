import json

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import conversation


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


def test_route_message_treats_standalone_deictic_as_ambiguous_not_general_chat():
    response = {
        "intent": "GENERAL_CHAT",
        "rewrite_query": None,
        "new_topic": None,
        "current_topic": None,
    }
    llm = FakeListChatModel(responses=[json.dumps(response)])

    route = conversation.route_message(
        llm,
        "Est-ce que les banques peuvent faire cette opération ?",
        {"turns": []},
    )

    assert route["intent"] == "AMBIGUOUS"
    assert route["rewrite_query"] is None


def test_follow_up_uses_standalone_query_for_dense_bm25(monkeypatch):
    rewritten = "relationship between Circular 2019-07 and Circular 2018-07"
    ordinary = _document()
    calls = {}

    class Backend:
        def retrieve(self, query):
            calls["retrieval_query"] = query
            return [(ordinary, 1.0)]

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

    def answer(_llm, message, _documents, memory):
        calls["answer_query"] = message
        calls["answer_memory"] = memory
        return "follow-up answer"

    monkeypatch.setattr(conversation, "generate_grounded_answer", lambda *args, **_kwargs: {"answer": answer(*args), "sources": []})

    result = conversation.chat(
        "How is it related to the previous circular?",
        _previous_state(),
        retrieval_backend=Backend(),
    )

    assert calls["retrieval_query"] == rewritten
    assert calls["answer_query"] == "How is it related to the previous circular?"
    assert rewritten in calls["answer_memory"]
    assert "What did Circular 2019-07 change?" in calls["answer_memory"]
    assert result["memory_state"]["turns"][-1]["standalone_query"] == rewritten
    assert result["memory_state"]["turns"][-1]["answer"] == "follow-up answer"


def test_new_topic_does_not_leak_old_turns_into_answer_memory(monkeypatch):
    ordinary = _document("Cir_2025_17_fr.pdf", page=3)
    captured = {}

    class Backend:
        def retrieve(self, _query):
            return [(ordinary, 1.0)]

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

    def answer(_llm, _message, _documents, memory):
        captured["memory"] = memory
        return "new-topic answer"

    monkeypatch.setattr(conversation, "generate_grounded_answer", lambda *args, **_kwargs: {"answer": answer(*args), "sources": []})

    result = conversation.chat(
        "Now tell me about Circular 2025-17.",
        _previous_state(),
        retrieval_backend=Backend(),
    )

    assert "What did Circular 2019-07 change?" not in captured["memory"]
    assert "Circular 2025-17" in captured["memory"]
    assert result["memory_state"]["current_topic"] == "Circular 2025-17"


def test_chat_uses_the_selected_profile_backend_and_answer_provider(monkeypatch):
    ordinary = _document("Cir_2025_17_fr.pdf", page=3)
    calls = {}

    class Backend:
        def retrieve(self, query):
            calls["retrieval_query"] = query
            return [(ordinary, 0.9)]

        def rank(self, _query, _documents):
            raise AssertionError("extra rerank is not needed")

    monkeypatch.setattr(
        conversation,
        "create_llm",
        lambda provider: calls.setdefault("answer_provider", provider) or object(),
    )
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "NEW_TOPIC",
            "rewrite_query": "Circular 2025-17 reporting",
            "new_topic": "Circular 2025-17",
            "current_topic": "Circular 2025-17",
        },
    )
    monkeypatch.setattr(
        conversation,
        "generate_grounded_answer",
        lambda _llm, _query, _documents, _memory, **_kwargs: {"answer": "answer", "sources": []},
    )

    result = conversation.chat(
        "What does Circular 2025-17 require?",
        {"topics": [], "turns": []},
        retrieval_backend=Backend(),
        llm_provider="ollama",
    )

    assert calls["answer_provider"] == "ollama"
    assert calls["retrieval_query"] == "Circular 2025-17 reporting"
    assert result["answer"] == "answer"


def test_new_topic_retrieval_uses_the_standalone_rewrite(monkeypatch):
    message = (
        "Notre établissement traverse un problème temporaire de liquidité. "
        "Dans quels cas peut-on demander une assistance financière exceptionnelle "
        "à la Banque Centrale ?"
    )
    rewritten_query = (
        "Dans quels cas une banque peut-elle demander une assistance financière "
        "exceptionnelle à la Banque Centrale de Tunisie ?"
    )
    calls = {}
    ordinary = _document("Cir_2016_07_fr.pdf", page=1)

    class Backend:
        def retrieve(self, query):
            calls["retrieval_query"] = query
            return [(ordinary, 1.0)]

    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "NEW_TOPIC",
            "rewrite_query": rewritten_query,
            "new_topic": "Assistance financière exceptionnelle",
            "current_topic": "Assistance financière exceptionnelle",
        },
    )
    monkeypatch.setattr(
        conversation,
        "generate_grounded_answer",
        lambda _llm, query, _documents, _memory, **_kwargs: calls.__setitem__("answer_query", query)
        or {"answer": "answer", "sources": []},
    )

    result = conversation.chat(
        message,
        _previous_state(),
        retrieval_backend=Backend(),
    )

    assert calls["retrieval_query"] == rewritten_query
    assert calls["answer_query"] == message
    assert result["memory_state"]["turns"][-1]["standalone_query"] == rewritten_query


def test_ambiguous_reference_asks_for_clarification_without_retrieval(monkeypatch):
    class Backend:
        def retrieve(self, _query):
            raise AssertionError("retrieval must not run")

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

    result = conversation.chat(
        "What about that one?",
        {
            **_previous_state(),
            "topics": ["Circular 2019-07", "Circular 2025-17"],
            "current_topic": None,
        },
        retrieval_backend=Backend(),
    )

    assert "which" in result["answer"].casefold()
    assert result["sources"] == []
    assert result["memory_state"]["turns"] == _previous_state()["turns"]


def test_general_chat_reply_uses_memory_without_json_wrapper():
    llm = FakeListChatModel(
        responses=[
            "D'après notre échange, nous avons parlé de la circulaire 2019-07. "
            "Je peux chercher d'autres notes BCT si vous précisez la question."
        ]
    )
    result = conversation.general_chat_reply(
        llm,
        "Peux-tu résumer notre conversation ?",
        _previous_state(),
    )
    assert result["status"] == "answered"
    assert result["sources"] == []
    assert "2019-07" in result["answer"]


def test_general_chat_replies_without_retrieval(monkeypatch):
    class Backend:
        def retrieve(self, _query):
            raise AssertionError("retrieval must not run")

    class FakeLLM:
        def invoke(self, _payload):
            class Msg:
                content = (
                    "Je suis l'assistant de recherche réglementaire BCT. "
                    "Posez une question précise sur une circulaire ou une note."
                )
            return Msg()

        def __or__(self, _other):
            return self

    monkeypatch.setattr(conversation, "create_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "GENERAL_CHAT",
            "rewrite_query": None,
            "new_topic": None,
            "current_topic": None,
        },
    )
    monkeypatch.setattr(
        conversation,
        "general_chat_reply",
        lambda _llm, message, _memory: {
            "status": "answered",
            "answer": "Bonjour — je recherche dans les circulaires et notes BCT.",
            "sources": [],
        },
    )

    result = conversation.chat(
        "Bonjour, comment peux-tu m'aider ?",
        {"topics": [], "turns": []},
        retrieval_backend=Backend(),
    )

    assert result["status"] == "answered"
    assert result["sources"] == []
    assert result["refusal_reason"] is None
    assert "circulaires" in result["answer"].casefold() or "bct" in result["answer"].casefold()


def test_general_chat_off_topic_still_skips_retrieval(monkeypatch):
    class Backend:
        def retrieve(self, _query):
            raise AssertionError("retrieval must not run")

    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "GENERAL_CHAT",
            "rewrite_query": None,
            "new_topic": None,
            "current_topic": None,
        },
    )
    monkeypatch.setattr(
        conversation,
        "general_chat_reply",
        lambda *_: {
            "status": "answered",
            "answer": "يمكنني مساعدتك في الوثائق التنظيمية للبنك المركزي، وليس في وصفات الطبخ.",
            "sources": [],
        },
    )

    result = conversation.chat(
        "كيف أعد طبق كسكسي تونسي في المنزل ؟",
        {"topics": [], "turns": []},
        retrieval_backend=Backend(),
    )

    assert result["status"] == "answered"
    assert result["sources"] == []
    assert result.get("refusal_reason") in (None, "")
