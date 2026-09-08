import json
import pytest
from answer_contract import parse_answer


EVIDENCE = [{"evidence_id": "E1", "source": "Cir_2020_03_fr.pdf", "page": 1,
             "score": 0.8, "text": "Objet : Les allocations pour voyages d'affaires."}]


def draft(evidence_id="E1", quote="Les allocations pour voyages d'affaires."):
    return {"status": "answered", "message": "UNTRUSTED FREE TEXT MUST NOT BE DISPLAYED",
            "claims": [{"text": "Elle concerne les allocations pour voyages d'affaires.",
                        "quotes": [{"evidence_id": evidence_id, "quote": quote}]}]}


def test_only_claims_and_actual_cited_sources_are_displayed():
    result = parse_answer(json.dumps(draft()), "Quel est l'objet ?", EVIDENCE)
    assert result["status"] == "answered"
    assert "UNTRUSTED" not in result["answer"]
    assert result["sources"] == [{"file": "Cir_2020_03_fr.pdf", "page": 1, "score": 0.8,
                                 "excerpt": "Les allocations pour voyages d'affaires."}]


@pytest.mark.parametrize("value", [draft("E999"), draft(quote="invented quote"),
    {"status": "answered", "message": "A fabricated legal conclusion", "claims": []},
    {"status": "insufficient_evidence", "message": "A fabricated legal conclusion", "claims": []}])
def test_invalid_or_unsupported_output_has_no_answer_sources(value):
    result = parse_answer(json.dumps(value), "Quel est l'objet ?", EVIDENCE)
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []
    assert "fabricated" not in result["answer"]


def test_conflicting_native_header_does_not_become_an_affirmative_answer():
    conflicting = [{**EVIDENCE[0], "text": "CIRCULAIRE AUX INTERMEDIAIRES AGREES n° 2002-03 du 04 février 2020\n" + EVIDENCE[0]["text"]}]
    result = parse_answer(json.dumps(draft()), "Quel est l'objet ?", conflicting)
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_invalid_model_response_keeps_the_arabic_question_language():
    result = parse_answer("not JSON", "ما هو تاريخها؟", EVIDENCE)
    assert result["answer"].startswith("المقاطع")
    assert result["sources"] == []


def test_partial_answer_keeps_supported_facts_and_uses_a_trusted_gap_message():
    value = draft()
    value.update(status="partial_answer", message="Je n’ai pas trouvé la date demandée dans ces passages.")
    result = parse_answer(json.dumps(value), "Quel est l'objet et la date ?", EVIDENCE)
    assert result["status"] == "partial_answer"
    assert "voyages d'affaires. [1]" in result["answer"]
    assert "une partie de la demande" in result["answer"]
    assert result["sources"][0]["excerpt"] == value["claims"][0]["quotes"][0]["quote"]


@pytest.mark.parametrize("change", ["no_claims", "wrong_quote"])
def test_partial_answer_does_not_bypass_evidence_requirements(change):
    value = draft()
    value.update(status="partial_answer", message="La date n’est pas établie.")
    if change == "no_claims":
        value["claims"] = []
    elif change == "wrong_quote":
        value["claims"][0]["quotes"][0]["quote"] = "invented"
    result = parse_answer(json.dumps(value), "Quel est l'objet ?", EVIDENCE)
    assert result["status"] == "insufficient_evidence"
    assert result["sources"] == []


def test_unverified_temporal_scope_cannot_be_presented_as_a_complete_answer():
    result = parse_answer(json.dumps(draft()), "What does this say and is it still in force?",
                          EVIDENCE, temporal_unverified=True)
    assert result["status"] == "partial_answer"
    assert result["answer"].startswith("This is the latest value supported by the cited passages")
    assert "currently in force" in result["answer"]
    assert "could not be fully confirmed" in result["answer"]
    assert result["sources"]


def test_generation_binds_partial_answer_scope_and_literal_reference_context():
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from answer_contract import generate_grounded_answer

    def respond(prompt):
        messages = prompt.to_messages()
        if "Select evidence for" in messages[0].content:
            return AIMessage(content=json.dumps({"decision": "partial", "reason": "", "evidence_ids": ["E1"]}))
        assert "Unverified temporal scope: True" in messages[0].content
        assert '{"previous": "message"}' in messages[1].content
        return AIMessage(content=json.dumps(draft()))

    document = Document(page_content=EVIDENCE[0]["text"], metadata={
        "source": EVIDENCE[0]["source"], "page": 1, "pages": [1]})
    result = generate_grounded_answer(RunnableLambda(respond), "Is it still in force?",
        [(document, 0.8)], '{"previous": "message"}', temporal_unverified=True)
    assert result["status"] == "partial_answer"
    assert result["sources"][0]["page"] == 1


def test_all_supporting_quotes_on_a_shared_page_are_preserved():
    evidence = [{**EVIDENCE[0], "text": EVIDENCE[0]["text"] + "\nTunis, le 04 février 2020."}]
    value = draft()
    value["claims"].append({"text": "L’en-tête porte la date du 04 février 2020.",
        "quotes": [{"evidence_id": "E1", "quote": "Tunis, le 04 février 2020."}]})
    result = parse_answer(json.dumps(value), "Quel est l'objet et la date de l'en-tête ?", evidence)
    assert len(result["sources"]) == 1
    assert "voyages d'affaires" in result["sources"][0]["excerpt"]
    assert "Tunis, le 04 février 2020." in result["sources"][0]["excerpt"]
    assert result["answer"].count("[1]") == 2
