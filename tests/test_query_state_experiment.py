import json

import pytest

from experiments.query_state_experiment import derive_status, parse_query_state


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
