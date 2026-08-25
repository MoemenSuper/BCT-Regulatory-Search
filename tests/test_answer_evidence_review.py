import pytest

from experiments.answer_evidence_review import (
    build_reviewed_answer_result,
    expand_review_labels,
)


def _suite():
    return {
        "cases": [
            {"id": "relevant", "relevant": True, "language": "fr"},
            {"id": "negative", "relevant": False, "language": "ar"},
        ]
    }


def _review():
    return {
        "reviewer_type": "agent",
        "reviewed_against": "evidence",
        "relevant_default": {
            "answer_correct": True,
            "citation_correct": True,
            "grounded": True,
            "note": "supported",
        },
        "relevant_overrides": {},
        "negative_labels": {
            "negative": {
                "abstained_or_refused": True,
                "clarification_requested": False,
                "safe_response": True,
                "expected_behavior_met": True,
                "note": "safe",
            }
        },
    }


def test_review_expands_labels_and_aggregates_language_metrics():
    generated = {
        "experiment_id": "experiment",
        "configuration": {},
        "metrics": {},
        "latency_seconds": {},
        "limitations": [],
        "records": [
            {"id": "relevant", "relevant": True, "language": "fr"},
            {"id": "negative", "relevant": False, "language": "ar"},
        ],
    }

    result = build_reviewed_answer_result(_suite(), generated, _review())

    assert result["reviewed_metrics"]["overall"]["answer_correct"]["rate"] == 1.0
    assert result["reviewed_metrics"]["overall"]["negative"]["safe_response"]["rate"] == 1.0
    assert result["review_status"]["independent_human_confirmation"] is False


def test_review_requires_exact_negative_coverage():
    review = _review()
    review["negative_labels"] = {}

    with pytest.raises(ValueError, match="exactly cover"):
        expand_review_labels(_suite(), review)


def test_review_accepts_structured_experiment_automatic_metrics_key():
    generated = {
        "experiment_id": "structured",
        "configuration": {},
        "automatic_metrics": {"overall": {"status_rate": 1.0}},
        "latency_seconds": {},
        "limitations": [],
        "records": [
            {"id": "relevant", "relevant": True, "language": "fr"},
            {"id": "negative", "relevant": False, "language": "ar"},
        ],
    }

    result = build_reviewed_answer_result(_suite(), generated, _review())

    assert result["automatic_metrics"]["overall"]["status_rate"] == 1.0


def test_explicit_relevant_labels_must_cover_every_relevant_case():
    suite = {
        "cases": [
            {"id": "r1", "relevant": True},
            {"id": "r2", "relevant": True},
            {"id": "n1", "relevant": False},
        ]
    }
    label = {
        "answer_correct": True,
        "citation_correct": True,
        "grounded": True,
        "note": "Reviewed.",
    }
    negative = {
        "abstained_or_refused": True,
        "clarification_requested": False,
        "safe_response": True,
        "expected_behavior_met": True,
        "note": "Safe.",
    }
    review = {
        "relevant_labels": {"r1": label},
        "negative_labels": {"n1": negative},
    }

    with pytest.raises(ValueError, match="exactly cover"):
        expand_review_labels(suite, review)

    review["relevant_labels"]["r2"] = label
    labels = expand_review_labels(suite, review)
    assert set(labels) == {"r1", "r2", "n1"}
