import json

import pytest

from experiments.structured_answer_experiment import (
    _contract,
    parse_structured_answer,
    structured_diagnostics,
)


def test_parse_structured_answer_enforces_claim_and_citation_shape():
    value = {
        "status": "answered",
        "answer": "Le taux est 6%.",
        "claims": [{"text": "Le taux est 6%.", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1", "source": "Cir.pdf", "page": 2}],
    }

    assert parse_structured_answer(json.dumps(value)) == value

    value["extra"] = True
    with pytest.raises(ValueError, match="top-level"):
        parse_structured_answer(json.dumps(value))


def test_non_answered_response_cannot_claim_or_cite():
    value = {
        "status": "clarification_needed",
        "answer": "Quelle allocation ?",
        "claims": [{"text": "claim", "evidence_ids": []}],
        "citations": [],
    }

    with pytest.raises(ValueError, match="must not contain"):
        parse_structured_answer(json.dumps(value))


def test_v2_parser_requires_nonempty_user_facing_answer():
    value = {
        "status": "insufficient_evidence",
        "answer": "",
        "claims": [],
        "citations": [],
    }

    assert parse_structured_answer(json.dumps(value)) == value
    with pytest.raises(ValueError, match="non-empty"):
        parse_structured_answer(json.dumps(value), require_nonempty_answer=True)


def test_diagnostics_require_exact_citation_and_expected_negative_status():
    relevant = {
        "relevant": True,
        "expected_source": "Cir_2026_01_fr.pdf",
        "expected_page": 2,
    }
    answer = {
        "status": "answered",
        "answer": "Réponse",
        "claims": [{"text": "Réponse", "evidence_ids": ["E1"]}],
        "citations": [
            {"evidence_id": "E1", "source": "Cir_2026_01_fr.pdf", "page": 2}
        ],
    }
    negative = {"relevant": False, "expected_behavior": "clarify"}
    clarification = {
        "status": "clarification_needed",
        "answer": "Quel transfert ?",
        "claims": [],
        "citations": [],
    }

    assert structured_diagnostics(relevant, answer)["exact_structured_citation"] is True
    assert structured_diagnostics(relevant, answer)["claim_evidence_links_valid"] is True
    assert structured_diagnostics(negative, clarification)["status_expected"] is True


def test_v3_contract_keeps_future_financial_requests_in_scope():
    version, prompt, _schema, require_nonempty = _contract(
        {"answer_experiment": {"prompt_version": "bct-claim-linked-answer-v3"}}
    )

    assert version == "bct-claim-linked-answer-v3"
    assert "future exchange rate" in prompt
    assert "insufficient_evidence" in prompt
    assert require_nonempty is True


def test_v4_contract_requires_row_identifiers_for_entity_confirmation():
    version, prompt, _schema, require_nonempty = _contract(
        {"answer_experiment": {"prompt_version": "bct-claim-linked-answer-v4"}}
    )

    assert version == "bct-claim-linked-answer-v4"
    assert "include the entity's row identifier" in prompt
    assert require_nonempty is True
