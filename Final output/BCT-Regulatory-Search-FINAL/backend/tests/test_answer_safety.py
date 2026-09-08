"""Synthetic answer-layer regressions; no benchmark IDs or expected answers."""
import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from answer_contract import generate_grounded_answer, parse_answer


def record(source="Cir_2022_41_fr.pdf", text="Le plafond est de 320 dinars.", eid="E1"):
    return dict(evidence_id=eid, source=source, page=2, score=0.9, text=text)


def draft(text="Le plafond est de 320 dinars.", quote=None, eid="E1", **changes):
    return dict(status="answered", message="", claims=[dict(
        text=text, quotes=[dict(evidence_id=eid, quote=quote or text)]
    )], **changes)


def parse(value, evidence, question="Quel est le plafond ?", **kwargs):
    return parse_answer(json.dumps(value, ensure_ascii=False), question, evidence, **kwargs)


def test_requested_circular_cannot_be_replaced_by_a_similar_document():
    evidence = [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]
    result = parse(draft("Le plafond est de 640 dinars.", eid="E2"), evidence,
                   "Selon la circulaire 2022-41, quel est le plafond ?")
    assert result["status"] == "insufficient_evidence"


@pytest.mark.parametrize("claim,quote", [
    ("Le plafond est de 640 dinars.", "Le plafond est de 320 dinars."),
    ("Le code est 0912.", "Le code est 0192."),
    ("La durée est de deux mois (60 jours).", "La durée est de deux mois."),
])
def test_claim_numbers_must_be_supported_by_its_own_quote(claim, quote):
    result = parse(draft(claim, quote), [record(text=quote), record(text=claim, eid="E2")])
    assert result["status"] == "insufficient_evidence"


def test_typographic_quote_normalization_preserves_the_actual_source_excerpt():
    quote = "L’allocation est de 3\u202f200 dinars."
    result = parse(draft("L'allocation est de 3 200 dinars.", "L'allocation est de 3 200 dinars."),
                   [record(text=quote)])
    assert result["status"] == "answered"
    assert result["sources"][0]["excerpt"] == quote


def test_partial_message_cannot_smuggle_an_unquoted_legal_answer():
    value = draft()
    value.update(status="partial_answer", message="Le plafond actuel est de 990 dinars.")
    result = parse(value, [record()])
    assert "990" not in result["answer"]


def test_current_query_is_qualified_even_when_called_without_graph_router():
    result = parse(draft(), [record()], "What is the latest supported ceiling?")
    assert result["status"] == "partial_answer"


def test_long_contiguous_quote_and_empty_partial_message_are_not_rejected():
    text = "Conditions complémentaires. " * 65 + "Le plafond est de 320 dinars."
    value = draft(quote=text)
    value["status"] = "partial_answer"
    result = parse(value, [record(text=text)])
    assert result["status"] == "partial_answer"
    assert result["sources"]


def test_abbreviated_quote_recovers_omitted_conditions_without_fuzzy_matching():
    full = "Le prêt est remboursable en 7 ans, sous réserve des conditions du contrat, à compter du premier versement."
    quote = "Le prêt est remboursable en 7 ans ... à compter du premier versement."
    result = parse(draft("Le prêt est remboursable en 7 ans.", quote), [record(text=full)])
    assert result["sources"][0]["excerpt"] == full
    wrong = parse(draft("Le prêt est remboursable en 7 ans.", quote.replace("7 ans", "8 ans")), [record(text=full)])
    assert wrong["status"] == "insufficient_evidence"


def test_small_word_number_equivalence():
    result = parse(draft("La durée est de 7 ans.", "La durée est de sept ans."), [record(text="La durée est de sept ans.")])
    assert result["status"] == "answered"


def test_cited_document_name_is_metadata_not_an_unsupported_rule_number():
    result = parse(draft("Le document Cir_2022_41_fr fixe le plafond à 320 dinars.", "Le plafond est de 320 dinars."), [record()])
    assert result["status"] == "answered"
    wrong = parse(draft("Le document Cir_2022_41_fr fixe le plafond à 41 dinars.", "Le plafond est de 320 dinars."), [record()])
    assert wrong["status"] == "insufficient_evidence"


def test_arabic_corrupt_header_is_not_silently_repaired():
    text = "مذكرة إلى البنوك عدد 41 لسنة 2202\nالرمز هو 709."
    result = parse(draft("الرمز هو 709.", "الرمز هو 709."), [record(source="Note_2022_41_ar.pdf", text=text)], "ما الرمز؟")
    assert result["status"] == "insufficient_evidence"


def test_ordinary_invalid_quote_gets_a_bounded_repair_with_reason():
    responses = [draft(quote="invented"), draft()]
    seen = []
    def respond(prompt):
        seen.append(prompt.to_messages())
        if "Select evidence for" in seen[-1][0].content:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E1"])))
        return AIMessage(content=json.dumps(responses.pop(0)))
    doc = Document(page_content=record()["text"], metadata={"source": record()["source"], "page": 2, "pages": [2]})
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", [(doc, .9)])
    assert result["status"] == "answered"
    assert any("quote_not_found" in m[-1].content for m in seen)


def test_conflicting_scope_returns_search_results_without_drafting():
    def respond(prompt):
        assert "Select evidence for" in prompt.to_messages()[0].content
        return AIMessage(content=json.dumps(dict(decision="clarification_needed", reason="Same scope, different ceilings; no requested instrument or precedence.", evidence_ids=[])))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", docs)
    assert result["status"] == "search_results"
    assert "320" not in result["answer"]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["excerpt"] == "Le plafond est de 320 dinars."


def test_draft_cannot_cite_evidence_excluded_by_selection():
    calls = []
    def respond(prompt):
        calls.append(prompt.to_messages())
        if "Select evidence for" in calls[-1][0].content:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="E2 concerns another operation", evidence_ids=["E1"])))
        assert '"evidence_id": "E2"' not in calls[-1][-1].content
        return AIMessage(content=json.dumps(draft("Le plafond est de 640 dinars.", eid="E2")))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record(), record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars.", "E2")]]
    result = generate_grounded_answer(RunnableLambda(respond), "Quel est le plafond ?", docs)
    assert result["status"] == "search_results"
    assert len(calls) == 3


def test_named_document_is_the_only_candidate_for_direct_contents_question():
    def respond(prompt):
        messages = prompt.to_messages()
        assert "Cir_2024_52_fr.pdf" not in messages[-1].content
        if "Select evidence for" in messages[0].content:
            return AIMessage(content=json.dumps(dict(decision="answer", reason="", evidence_ids=["E2"])))
        return AIMessage(content=json.dumps(draft(eid="E2")))
    docs = [(Document(page_content=r["text"], metadata={"source": r["source"], "page": 2, "pages": [2]}), .9)
            for r in [record("Cir_2024_52_fr.pdf", "Le plafond est de 640 dinars."), record()]]
    result = generate_grounded_answer(RunnableLambda(respond), "Selon la circulaire 2022-41, quel est le plafond ?", docs)
    assert result["status"] == "answered"
    assert result["sources"][0]["file"] == "Cir_2022_41_fr.pdf"


@pytest.mark.parametrize("question", [
    "Quel est le dernier plafond applicable ?", "ما أحدث سقف؟", "What is the latest ceiling?",
])
def test_currentness_detection_in_both_answer_entrypoints(question):
    from graph_contract import is_temporal_rule_query
    assert is_temporal_rule_query(question)
    assert parse(draft(), [record()], question)["status"] == "partial_answer"


def test_historical_uncertainty_is_not_described_as_todays_latest_value():
    result = parse(draft(), [record()], "Au 1er janvier 2023, quel plafond était applicable ?")
    assert result["status"] == "partial_answer"
    assert "date demandée" in result["answer"]
    assert "dernière valeur" not in result["answer"]


def test_resolved_applicability_can_be_explicitly_passed_by_the_caller():
    result = parse(draft(), [record()], "What is currently in force?", temporal_unverified=False)
    assert result["status"] == "answered"


@pytest.mark.parametrize("claim", ["Le plafond actuel est de 320 dinars.", "Le plafond actuellement applicable est de 320 dinars.", "The current ceiling is 320 dinars.", "السقف الحالي هو 320 دينار."])
def test_a_disclaimer_does_not_validate_an_unverified_currentness_assertion(claim):
    result = parse(draft(claim, "Le plafond est de 320 dinars."), [record()],
                   "What is currently applicable?", temporal_unverified=True)
    assert result["status"] == "insufficient_evidence"
    assert not result["sources"]


def test_search_fallback_preserves_original_top5_not_currentness_answer_order(monkeypatch):
    import conversation
    docs = [(Document(page_content=f"Passage original {n}", metadata={
        "source": f"Cir_{2020+n}_41_fr.pdf", "page": 2, "pages": [2]}), 1 - n / 10)
        for n in range(6)]
    class Backend:
        def retrieve(self, query):
            return docs
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(conversation, "route_message", lambda *_: dict(
        intent="NEW_TOPIC", rewrite_query="", new_topic="plafond", current_topic="plafond"))
    def answer(_llm, _question, evidence, *_args, **_kwargs):
        assert evidence[0][0].metadata["source"] == "Cir_2024_41_fr.pdf"
        return dict(status="search_results", answer="fallback", sources=[])
    monkeypatch.setattr(conversation, "generate_grounded_answer", answer)
    result = conversation.chat("Quel est le plafond actuel ?", {"topics": [], "turns": []},
                               None, None, None, [], retrieval_backend=Backend())
    assert result["status"] == "search_results"
    assert [s["file"] for s in result["sources"]] == [d.metadata["source"] for d, _ in docs[:5]]
    assert [s["excerpt"] for s in result["sources"]] == [d.page_content for d, _ in docs[:5]]


def test_no_retrieval_does_not_invent_search_results():
    def unexpected(_):
        raise AssertionError("Model must not be called without evidence")
    result = generate_grounded_answer(RunnableLambda(unexpected), "Quel est le plafond ?", [])
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_search_results_survive_api_serialization_and_history(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module
    from conversation_memory import ConversationStore
    from answer_contract import search_response
    monkeypatch.setenv("BCT_ENABLE_GRAPH", "0")
    monkeypatch.setattr(app_module, "create_local_backend", lambda: object())
    monkeypatch.setattr(app_module, "create_voyage_backend_from_environment", lambda: object())
    monkeypatch.setattr(app_module, "open_relationship_graph_runtime", lambda: None)
    monkeypatch.setattr(app_module, "open_conversation_store", lambda: ConversationStore(tmp_path / "history.sqlite3"))
    result = search_response("Quel est le plafond ?", [record(eid=f"E{n}", source=f"Cir_2022_{n:02}_fr.pdf") for n in range(1, 6)])
    monkeypatch.setattr(app_module, "chat", lambda *args, **kwargs: {**result, "memory_state": {}, "graph_trace": {}})
    with TestClient(app_module.app) as client:
        response = client.post("/chat", json={"question": "Quel est le plafond ?"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "search_results"
        assert body["sources"] == result["sources"]
        saved = client.get(f"/conversations/{body['conversation_id']}").json()["turns"][-1]
        assert saved["answer_status"] == "search_results"
        assert saved["sources"] == result["sources"]
