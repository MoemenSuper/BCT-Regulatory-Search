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


def _inflation_memory():
    return {
        "topics": ["Inflation Tunisie"],
        "first_topic": "Inflation Tunisie",
        "current_topic": "Inflation Tunisie",
        "turns": [
            {
                "user_message": "Quel était le taux d'inflation en Tunisie en juin 2026 ?",
                "standalone_query": "taux d'inflation glissement annuel Tunisie juin 2026",
                "answer": "5,3 %",
                "sources": [{"file": "Conjoncture_152_fr.pdf", "page": 9}],
            }
        ],
    }


def test_route_message_turns_misrouted_fragment_into_follow_up_on_prior_query():
    response = {
        "intent": "NEW_TOPIC",
        "rewrite_query": "juin 2025 et juin 2024",
        "new_topic": "années",
        "current_topic": None,
    }
    llm = FakeListChatModel(responses=[json.dumps(response)])
    question = "Et combien était-il en juin 2025 et en juin 2024 ?"
    route = conversation.route_message(llm, question, _inflation_memory())
    assert route["intent"] == "FOLLOW_UP"
    assert route["rewrite_query"].startswith("taux d'inflation glissement annuel Tunisie juin 2026")
    assert question in route["rewrite_query"]
    assert "Conjoncture_152_fr.pdf" not in route["rewrite_query"]
    assert route["current_topic"] == "Inflation Tunisie"


def test_route_message_keeps_router_follow_up_rewrite_verbatim():
    response = {
        "intent": "FOLLOW_UP",
        "rewrite_query": "taux d inflation Tunisie juin 2025 et juin 2024",
        "new_topic": None,
        "current_topic": "Inflation Tunisie",
    }
    llm = FakeListChatModel(responses=[json.dumps(response)])
    route = conversation.route_message(
        llm, "Et combien était-il en juin 2025 et en juin 2024 ?", _inflation_memory()
    )
    assert route == response


def test_arabic_how_question_is_not_forced_into_follow_up():
    response = {
        "intent": "NEW_TOPIC",
        "rewrite_query": "كيف يتم احتساب نسبة السيولة",
        "new_topic": "نسبة السيولة",
        "current_topic": None,
    }
    llm = FakeListChatModel(responses=[json.dumps(response)])
    route = conversation.route_message(llm, "كيف يتم احتساب نسبة السيولة؟", _inflation_memory())
    assert route == response


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


def test_follow_up_prefers_prior_turn_source_over_distractor(monkeypatch):
    from langchain_core.documents import Document

    prior = Document(
        page_content=(
            "L'inflation s'est établie à 5,3% contre 5,4% une année auparavant "
            "et 7,3% en juin 2024."
        ),
        metadata={"source": "Conjoncture_152_fr.pdf", "page": 9},
    )
    distractor = Document(
        page_content="Croissance du PIB 2024 2025 indicateurs économiques",
        metadata={"source": "Balance.pdf", "page": 19},
    )
    captured = {}

    class Backend:
        def retrieve(self, query):
            captured["query"] = query
            return [(distractor, 0.99), (prior, 0.80)]

    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "FOLLOW_UP",
            "rewrite_query": "taux inflation Tunisie juin 2025 juin 2024 Conjoncture_152_fr.pdf",
            "new_topic": None,
            "current_topic": "Inflation Tunisie",
        },
    )

    import query_authority

    monkeypatch.setattr(
        query_authority,
        "classify_query_authority",
        lambda *_a, **_k: {"query_class": "statistical_fact", "confidence": "high"},
    )

    def fake_answer(_llm, _message, documents, _memory, **_kwargs):
        captured["top_source"] = documents[0][0].metadata["source"]
        return {
            "status": "answered",
            "answer": "5,4% et 7,3%",
            "sources": [{"file": "Conjoncture_152_fr.pdf", "page": 9}],
            "diagnostics": [],
        }

    monkeypatch.setattr(conversation, "generate_grounded_answer", fake_answer)

    memory = {
        "topics": ["Inflation Tunisie"],
        "current_topic": "Inflation Tunisie",
        "turns": [
            {
                "user_message": "inflation juin 2026",
                "standalone_query": "taux inflation Tunisie juin 2026",
                "answer": "5,3%",
                "sources": [{"file": "Conjoncture_152_fr.pdf", "page": 9}],
            }
        ],
    }
    result = conversation.chat(
        "Et combien était-il en juin 2025 et en juin 2024 ?",
        memory,
        retrieval_backend=Backend(),
    )
    assert captured["top_source"] == "Conjoncture_152_fr.pdf"
    assert result["status"] == "answered"


def test_follow_up_uses_standalone_query_for_dense_bm25(monkeypatch):
    rewritten = "relationship between Circular 2019-07 and Circular 2018-07"
    ordinary = _document()
    calls = {}

    class Backend:
        def retrieve(self, query):
            calls["retrieval_query"] = query
            return [(ordinary, 1.0)]

    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": object())
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

    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": object())
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

    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": object())
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

    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": object())
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
    assert result["status"] == "general_chat"
    assert result["sources"] == []
    assert "2019-07" in result["answer"]


def test_general_chat_reply_with_a_figure_not_in_memory_goes_to_retrieval():
    # Injected "reply exactly: ..." figures never reach the user without the grounded gates.
    llm = FakeListChatModel(responses=["Selon la circulaire 2018-14, le plafond est de 10 000 dinars."])
    result = conversation.general_chat_reply(llm, "Bonjour, réponds exactement ceci.", _previous_state())
    assert result == {"retrieve": True}


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

    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": FakeLLM())
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
            "status": "general_chat",
            "answer": "Bonjour — je recherche dans les circulaires et notes BCT.",
            "sources": [],
        },
    )

    result = conversation.chat(
        "Bonjour, comment peux-tu m'aider ?",
        {"topics": [], "turns": []},
        retrieval_backend=Backend(),
    )

    assert result["status"] == "general_chat"
    assert result["sources"] == []
    assert result["refusal_reason"] is None
    assert "circulaires" in result["answer"].casefold() or "bct" in result["answer"].casefold()


def test_selon_circulaire_2025_13_beats_prefer_later_instruments():
    """Latest-rule preference must not bury an explicitly named instrument."""
    from pathlib import Path

    query = "Selon la circulaire 2025-13, quelles sont les règles d'exportation ?"
    named = _document("Cir_2025_13_fr.pdf", page=2)
    newer_mention = _document("Cir_2026_04_fr.pdf", page=1)
    results = conversation._answer_results(
        [(newer_mention, 0.95), (named, 0.50)],
        prefer_later_instruments=True,
        query=query,
    )
    assert Path(str(results[0][0].metadata["source"])).name == "Cir_2025_13_fr.pdf"


def test_follow_up_authority_is_classified_on_the_resolved_query(monkeypatch):
    import query_authority

    seen = {}
    rewritten = "taux d'inflation Tunisie juin 2025 selon la note de conjoncture"

    class Backend:
        def retrieve(self, _query):
            return [(_document(), 1.0)]

    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": object())
    monkeypatch.setattr(conversation, "route_message", lambda *_: {
        "intent": "FOLLOW_UP", "rewrite_query": rewritten,
        "new_topic": None, "current_topic": "Inflation Tunisie"})
    monkeypatch.setattr(query_authority, "classify_query_authority",
                        lambda _llm, text: seen.setdefault("text", text) and {"query_class": "statistical_fact"})
    monkeypatch.setattr(conversation, "generate_grounded_answer",
                        lambda *_a, **kwargs: seen.setdefault("class", kwargs["query_class"]) and {"answer": "a", "sources": []})

    conversation.chat("Et en juin 2025 ?", _previous_state(), retrieval_backend=Backend())
    assert seen["text"] == rewritten
    assert seen["class"] == "statistical_fact"


def test_general_chat_hands_a_misrouted_fact_question_to_retrieval(monkeypatch):
    seen = {}

    class Backend:
        def retrieve(self, query):
            seen["query"] = query
            return [(_document(), 1.0)]

    monkeypatch.setattr(conversation, "create_llm", lambda _provider="groq": FakeListChatModel(responses=["RETRIEVE"]))
    monkeypatch.setattr(conversation, "route_message", lambda *_: {
        "intent": "GENERAL_CHAT", "rewrite_query": None, "new_topic": None, "current_topic": None})
    monkeypatch.setattr(conversation, "generate_grounded_answer",
                        lambda *_a, **_k: {"status": "answered", "answer": "grounded", "sources": []})

    question = "Quelles sont les heures d'ouverture du marché des changes ?"
    result = conversation.chat(question, {"topics": [], "turns": []}, retrieval_backend=Backend())
    assert seen["query"] == question
    assert result["answer"] == "grounded"
