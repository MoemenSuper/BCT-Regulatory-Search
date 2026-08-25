import pytest

from experiments.retrieved_context_answer_experiment import (
    MAX_COMPLETION_TOKENS,
    PROMPT_VERSION,
    REASONING_EFFORT,
    _query_states,
    apply_post_retrieval_query_state,
    retrieved_structured_diagnostics,
)


def test_v2_budget_changes_only_reasoning_budget_not_evidence_count():
    assert PROMPT_VERSION == "bct-retrieved-context-answer-v2-low-reasoning"
    assert REASONING_EFFORT == "low"
    assert MAX_COMPLETION_TOKENS == 1024


def test_query_state_inputs_must_exactly_cover_suite():
    suite = {"cases": [{"id": "relevant"}, {"id": "negative"}]}
    relevant = {"records": [{"id": "relevant", "query_state": {}}]}
    negative = {"records": [{"id": "negative", "query_state": {}}]}

    assert set(_query_states(suite, negative, relevant)) == {"relevant", "negative"}
    with pytest.raises(ValueError, match="exactly cover"):
        _query_states(suite, {"records": []}, relevant)


def test_query_state_is_applied_only_after_generator_abstains():
    answered = {"status": "answered", "answer": "A", "claims": [], "citations": []}
    state = {
        "scope": "clearly_unrelated",
        "temporal_state": "not_current_or_future",
        "ambiguity": "sufficiently_specific",
        "missing_detail": "",
    }

    assert apply_post_retrieval_query_state(answered, state, "fr")[0] is answered
    abstention = {"status": "insufficient_evidence", "answer": "x", "claims": [], "citations": []}
    response, path = apply_post_retrieval_query_state(abstention, state, "fr")
    assert response["status"] == "out_of_scope"
    assert path == "retrieved_top5_abstention_then_query_state"


def test_retrieved_diagnostics_require_citation_to_matching_evidence():
    case = {
        "relevant": True,
        "expected_source": "Cir.pdf",
        "expected_page": 2,
    }
    response = {
        "status": "answered",
        "answer": "A",
        "claims": [{"text": "A", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1", "source": "Cir.pdf", "page": 2}],
    }
    evidence = [{"evidence_id": "E1", "source": "Cir.pdf", "page": 2}]

    diagnostics = retrieved_structured_diagnostics(case, response, evidence)
    assert diagnostics["exact_expected_citation"] is True
    assert diagnostics["citations_match_retrieved_evidence"] is True

    response["citations"][0]["page"] = 3
    assert retrieved_structured_diagnostics(case, response, evidence)[
        "citations_match_retrieved_evidence"
    ] is False
