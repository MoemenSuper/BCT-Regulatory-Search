import pytest

from experiments.answer_status_suite import build_answer_status_suite


def test_status_suite_selects_every_negative_and_preserves_order():
    cases = [
        {"id": "relevant", "relevant": True, "language": "fr"},
        {
            "id": "clarify-ar",
            "relevant": False,
            "language": "ar",
            "expected_behavior": "clarify",
        },
        {
            "id": "abstain-fr",
            "relevant": False,
            "language": "fr",
            "expected_behavior": "abstain",
        },
        {
            "id": "reject-fr",
            "relevant": False,
            "language": "fr",
            "expected_behavior": "reject_out_of_scope",
        },
    ]

    result = build_answer_status_suite(
        base_suite={"cases": cases}, base_suite_sha256="A" * 64
    )

    assert [case["id"] for case in result["cases"]] == [
        "clarify-ar",
        "abstain-fr",
        "reject-fr",
    ]
    assert result["counts"]["total"] == 3
    assert result["counts"]["by_expected_behavior"] == {
        "abstain": 1,
        "clarify": 1,
        "reject_out_of_scope": 1,
    }
    assert result["answer_experiment"]["prompt_version"] == (
        "bct-claim-linked-answer-v2"
    )


def test_status_suite_requires_negative_cases():
    with pytest.raises(ValueError, match="no negative"):
        build_answer_status_suite(
            base_suite={"cases": [{"id": "only", "relevant": True}]},
            base_suite_sha256="A" * 64,
        )


def test_status_suite_can_freeze_v3_without_changing_case_selection():
    case = {
        "id": "future-fr",
        "relevant": False,
        "language": "fr",
        "expected_behavior": "abstain",
    }

    result = build_answer_status_suite(
        base_suite={"cases": [case]},
        base_suite_sha256="A" * 64,
        prompt_version="bct-claim-linked-answer-v3",
    )

    assert result["cases"] == [case]
    assert result["answer_experiment"] == {
        "experiment_id": "claim-linked-status-policy-development-v3",
        "candidate_and_evidence": "all frozen negative and ambiguous cases with no evidence",
        "prompt_version": "bct-claim-linked-answer-v3",
    }
