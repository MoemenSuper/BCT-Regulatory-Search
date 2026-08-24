import pytest

from experiments.stress_catalog_benchmark import benchmark_stress_catalog


def _catalog():
    return {
        "inputs": {"evaluation_sha256": "eval", "result_sha256": "base"},
        "suites": {
            "risk": {
                "cases": [
                    {"id": "repair", "relevant": True, "language": "ar", "role": "failure"},
                    {"id": "regress", "relevant": True, "language": "fr", "role": "control"},
                ]
            },
            "negative": {
                "cases": [
                    {"id": "negative", "relevant": False, "language": "ar"},
                ]
            },
        },
    }


def _retrieval():
    return {
        "inputs": {"evaluation_sha256": "eval", "current_result_sha256": "base"},
        "rank_records": {
            "repair": {"current_page": 6, "fusion_page": 5},
            "regress": {"current_page": 5, "fusion_page": 6},
            "negative": {"current_page": None, "fusion_page": None},
        },
    }


def test_benchmark_reports_language_roles_and_cutoff_changes():
    result = benchmark_stress_catalog(_catalog(), _retrieval())

    overall = result["suites"]["risk"]["metrics"]["overall"]
    assert overall["current_page_at_5"] == 0.5
    assert overall["fusion_page_at_5"] == 0.5
    assert overall["repairs_at_5"] == 1
    assert overall["regressions_at_5"] == 1
    assert result["suites"]["risk"]["metrics"]["ar"]["net_at_5"] == 1
    assert result["suites"]["risk"]["metrics"]["fr"]["net_at_5"] == -1
    assert result["suites"]["risk"]["metrics"]["by_role"]["failure"]["n"] == 1
    assert result["suites"]["negative"]["status"] == "not_evaluated"


def test_benchmark_rejects_hash_mismatch():
    retrieval = _retrieval()
    retrieval["inputs"]["evaluation_sha256"] = "other"

    with pytest.raises(ValueError, match="different evaluation hashes"):
        benchmark_stress_catalog(_catalog(), retrieval)
