import conversation
from answer_contract import format_refusal_reason, generate_grounded_answer
from conversation_memory import ConversationStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


def test_format_refusal_reason_keeps_selection_detail():
    reason = format_refusal_reason(
        "search_results",
        ["selection:clarification_needed:Same scope, different ceilings"],
    )
    assert reason == "selection:clarification_needed:Same scope, different ceilings"


def test_named_instrument_absent_is_specific(monkeypatch):
    docs = [
        (
            Document(
                page_content="Le plafond est de 320 dinars.",
                metadata={"source": "Cir_2016_01_fr.pdf", "page": 2, "pages": [2]},
            ),
            0.9,
        )
    ]
    result = generate_grounded_answer(
        RunnableLambda(lambda _prompt: AIMessage(content="{}")),
        "Quel est l'objet de la circulaire 2024-52 ?",
        docs,
    )
    assert result["status"] == "search_results"
    assert result["diagnostics"] == ["named_instrument_absent:cir:2024-52"]


def test_chat_keeps_refusal_reason_without_user_facing_diagnostics(monkeypatch):
    doc = Document(
        page_content="Le plafond est de 320 dinars.",
        metadata={"source": "Cir_2017_08_fr.pdf", "page": 2, "pages": [2]},
    )

    def answer(*_args, **_kwargs):
        return {
            "status": "search_results",
            "answer": "fallback",
            "sources": [],
            "diagnostics": ["quote_not_found", "quote_not_found"],
        }

    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: dict(intent="NEW_TOPIC", rewrite_query="", new_topic="t", current_topic="t"),
    )
    monkeypatch.setattr(conversation, "generate_grounded_answer", answer)

    class Backend:
        def retrieve(self, _query):
            return [(doc, 0.9)]

    result = conversation.chat(
        "Quel est le plafond ?",
        {"topics": [], "turns": []},
        retrieval_backend=Backend(),
    )
    assert "diagnostics" not in result
    assert result["status"] == "search_results"
    assert result["refusal_reason"] == "quote_not_found"
    assert result["refusal_diagnostics"] == ["quote_not_found", "quote_not_found"]


def test_answer_refusals_persist_in_conversation_store(tmp_path):
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    store.record_answer_refusal(
        conversation_id="c1",
        user_id="u1",
        user_email="analyst@bct.gov.tn",
        question="Quel est le plafond ?",
        answer_status="search_results",
        reason="selection:clarification_needed:Same scope, different ceilings",
        diagnostics=["selection:clarification_needed:Same scope, different ceilings"],
        profile="cloud",
    )
    assert store.count_answer_refusals() == 1
    item = store.list_answer_refusals()[0]
    assert item["user_email"] == "analyst@bct.gov.tn"
    assert "clarification_needed" in item["reason"]
    assert item["diagnostics"][0].startswith("selection:clarification_needed:")
