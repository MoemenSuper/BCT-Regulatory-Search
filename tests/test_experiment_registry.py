import json

import pytest

from experiments.experiment_registry import append_experiment


def _entry(experiment_id="retrieval-001"):
    return {
        "experiment_id": experiment_id,
        "timestamp": "2026-08-24T00:00:00+00:00",
        "hypothesis": "Sequential structured chunks improve exact-page retrieval.",
        "changed_variable": "chunk_representation",
        "configuration": {"chunk_size": 1000, "overlap": 200},
        "model_identifiers": {
            "embedding": "intfloat/multilingual-e5-small",
            "reranker": "BAAI/bge-reranker-v2-m3",
        },
        "providers": {"embedding": "local", "reranker": "local"},
        "code_commit": "1778c51",
        "dataset_hashes": {"development": "A" * 64},
        "development_metrics": {"exact_page_at_5": 0.8795},
        "validation_metrics": {"status": "not_available"},
        "language_metrics": {
            "fr": {"exact_page_at_5": 0.90},
            "ar": {"exact_page_at_5": 0.85},
        },
        "repairs": 22,
        "regressions": 9,
        "failure_distribution": {"extraction": 25},
        "latency": {"mean_seconds": 1.055},
        "approximate_cost": {"currency": "USD", "amount": 0.0},
        "conclusion": "The controlled development result improved significantly.",
        "decision": "KEEP",
    }


def test_registry_appends_valid_unique_entries(tmp_path):
    registry = tmp_path / "registry.jsonl"

    append_experiment(registry, _entry())
    append_experiment(registry, _entry("retrieval-002"))

    rows = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
    assert [row["experiment_id"] for row in rows] == ["retrieval-001", "retrieval-002"]


def test_registry_rejects_duplicates_and_incomplete_language_metrics(tmp_path):
    registry = tmp_path / "registry.jsonl"
    entry = _entry()
    append_experiment(registry, entry)

    with pytest.raises(ValueError, match="already exists"):
        append_experiment(registry, entry)

    incomplete = _entry("retrieval-002")
    incomplete["language_metrics"]["ar"] = {}
    with pytest.raises(ValueError, match="French and Arabic"):
        append_experiment(registry, incomplete)

    invalid_hash = _entry("retrieval-003")
    invalid_hash["dataset_hashes"]["development"] = "ABC123"
    with pytest.raises(ValueError, match="SHA-256"):
        append_experiment(registry, invalid_hash)
