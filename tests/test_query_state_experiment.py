import json

import pytest

from experiments.query_state_experiment import (
    _answer,
    derive_pre_retrieval_decision,
    derive_status,
    parse_query_state,
)


def _state(**overrides):
    return {
        "scope": "bct_regulatory_or_financial",
        "temporal_state": "not_current_or_future",
        "ambiguity": "sufficiently_specific",
        "missing_detail": "",
        **overrides,
    }


def test_status_mapping_has_conservative_deterministic_precedence():
    assert derive_status(_state(scope="clearly_unrelated")) == "out_of_scope"
    assert (
        derive_status(_state(temporal_state="current_or_future"))
        == "insufficient_evidence"
    )
    assert (
        derive_status(
            _state(
                ambiguity="missing_discriminating_detail",
                missing_detail="allowance type",
            )
        )
        == "clarification_needed"
    )
    assert derive_status(_state()) == "insufficient_evidence"


def test_pre_retrieval_decision_does_not_preempt_specific_in_scope_query():
    assert derive_pre_retrieval_decision(_state()) == "proceed_to_retrieval"
    assert (
        derive_pre_retrieval_decision(
            _state(
                ambiguity="missing_discriminating_detail",
                missing_detail="transfer type",
            )
        )
        == "clarification_needed"
    )


def test_query_state_requires_missing_detail_only_for_ambiguity():
    ambiguous = _state(
        ambiguity="missing_discriminating_detail", missing_detail="transfer type"
    )
    assert parse_query_state(json.dumps(ambiguous)) == ambiguous

    with pytest.raises(ValueError, match="requires a missing detail"):
        parse_query_state(
            json.dumps(
                _state(
                    ambiguity="missing_discriminating_detail", missing_detail=""
                )
            )
        )
    with pytest.raises(ValueError, match="must not contain"):
        parse_query_state(json.dumps(_state(missing_detail="unexpected")))


def test_clarification_text_is_deterministically_bilingual():
    state = _state(
        ambiguity="missing_discriminating_detail",
        missing_detail="type of allowance",
    )

    assert _answer("clarification_needed", state, "ar") == (
        "يرجى تحديد النوع أو الفئة المقصودة في طلبك."
    )
    assert _answer("clarification_needed", state, "fr") == (
        "Veuillez préciser le type ou la catégorie visée par votre demande."
    )
