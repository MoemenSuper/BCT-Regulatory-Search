import pytest

from experiments.selective_answer_candidate import compose_selective_answer_candidate


def _record(case_id, language, status):
    return {
        "id": case_id,
        "language": language,
        "relevant": case_id != "negative",
        "response": {"status": status},
        "structured_diagnostics": {
            "exact_structured_citation": True,
            "claim_evidence_links_valid": True,
            "status_expected": True,
        },
        "latency_seconds": 1.0,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _inputs():
    suite = {
        "cases": [
            {"id": "answered", "language": "fr", "relevant": True},
            {"id": "retry", "language": "ar", "relevant": True},
            {"id": "negative", "language": "ar", "relevant": False},
        ]
    }
    base = {
        "records": [
            _record("answered", "fr", "answered"),
            _record("retry", "ar", "insufficient_evidence"),
            _record("negative", "ar", "insufficient_evidence"),
        ]
    }
    retry_suite = {"cases": [{"id": "retry"}]}
    retry = {"records": [_record("retry", "ar", "answered")]}
    negative = {"records": [_record("negative", "ar", "clarification_needed")]}
    return suite, base, retry_suite, retry, negative


def test_composition_uses_only_insufficient_relevant_retries_and_query_state_negatives():
    result = compose_selective_answer_candidate(
        suite=_inputs()[0],
        base_result=_inputs()[1],
        retry_suite=_inputs()[2],
        retry_result=_inputs()[3],
        negative_result=_inputs()[4],
    )

    assert [record["answer_path"] for record in result["records"]] == [
        "verified_excerpt_claim_linked_answer",
        "single_full_page_context_retry",
        "query_state_then_deterministic_status",
    ]
    assert result["policy_diagnostics"]["retried_case_ids"] == ["retry"]


def test_composition_fails_closed_when_retry_policy_does_not_match_base_status():
    suite, base, retry_suite, retry, negative = _inputs()
    base["records"][1]["response"]["status"] = "answered"

    with pytest.raises(ValueError, match="unexpectedly selected"):
        compose_selective_answer_candidate(
            suite=suite,
            base_result=base,
            retry_suite=retry_suite,
            retry_result=retry,
            negative_result=negative,
        )
