import json

import pytest

from experiments.evaluation_protocol import freeze_protocol
from experiments.frozen_evaluation import evaluate_frozen_split


def _case(case_id, source, language, *, relevant=True):
    case = {
        "id": case_id,
        "query": f"Question {case_id}",
        "language": language,
        "category": "definition" if relevant else "not_in_corpus",
        "relevant": relevant,
    }
    if relevant:
        case.update(
            {
                "expected_source": source,
                "expected_page": 1,
                "evidence_quote": "Independently verified evidence.",
            }
        )
    return case


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _split(prefix, number):
    return [
        _case(f"{prefix}-fr", f"Cir_2026_{number}_fr.pdf", "fr"),
        _case(f"{prefix}-ar", f"Note_2026_{number + 1}_ar.pdf", "ar"),
        _case(f"{prefix}-negative", None, "ar", relevant=False),
    ]


def _result(path, prefix):
    return _write(
        path,
        {
            "records": [
                {
                    "id": f"{prefix}-fr",
                    "result": {"source_rank": 1, "exact_page_rank": 1},
                    "answer_evaluation": {
                        "answer_correct": True,
                        "citation_correct": True,
                        "grounded": True,
                    },
                },
                {
                    "id": f"{prefix}-ar",
                    "result": {"source_rank": 2, "exact_page_rank": 6},
                    "answer_evaluation": {
                        "answer_correct": False,
                        "citation_correct": False,
                        "grounded": True,
                    },
                },
                {
                    "id": f"{prefix}-negative",
                    "result": {"source_rank": None, "exact_page_rank": None},
                    "answer_evaluation": {
                        "abstained": True,
                        "safe_response": True,
                    },
                },
            ]
        },
    )


def _protocol(tmp_path):
    development = _write(tmp_path / "development.json", _split("dev", 10))
    validation = _write(tmp_path / "validation.json", _split("val", 20))
    holdout = _write(tmp_path / "holdout.json", _split("holdout", 30))
    protocol_path = tmp_path / "protocol.json"
    freeze_protocol(
        development=development,
        validation=validation,
        holdout=holdout,
        output=protocol_path,
        frozen_at="2026-08-24T00:00:00+00:00",
    )
    return protocol_path, validation, holdout


def test_final_holdout_emits_aggregate_metrics_and_logs_access(tmp_path):
    protocol, _, _ = _protocol(tmp_path)
    result = _result(tmp_path / "holdout-result.json", "holdout")
    output = tmp_path / "holdout-aggregate.json"
    ledger = tmp_path / "access.jsonl"

    artifact = evaluate_frozen_split(
        protocol_path=protocol,
        role="final_holdout",
        result_path=result,
        output_path=output,
        ledger_path=ledger,
        purpose="Final release gate",
        accessed_by="test-suite",
        code_commit="abc123",
        accessed_at="2026-08-24T01:00:00+00:00",
    )

    assert artifact["role"] == "final_holdout"
    assert artifact["detail_policy"] == "aggregate_only"
    assert artifact["metrics"]["overall"]["relevant_count"] == 2
    assert artifact["metrics"]["overall"]["exact_page_at_5"] == 0.5
    assert artifact["metrics"]["fr"]["answer_correct"]["rate"] == 1.0
    assert artifact["metrics"]["ar"]["negative"]["safe_response"]["rate"] == 1.0
    serialized = output.read_text(encoding="utf-8")
    assert "holdout-fr" not in serialized
    assert "Cir_2026_30_fr.pdf" not in serialized

    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["role"] == "final_holdout"
    assert events[0]["detail_policy"] == "aggregate_only"
    assert events[0]["dataset_sha256"] == artifact["dataset_sha256"]


def test_final_holdout_refuses_case_details(tmp_path):
    protocol, _, _ = _protocol(tmp_path)
    result = _result(tmp_path / "holdout-result.json", "holdout")

    with pytest.raises(ValueError, match="aggregate-only"):
        evaluate_frozen_split(
            protocol_path=protocol,
            role="final_holdout",
            result_path=result,
            output_path=tmp_path / "output.json",
            ledger_path=tmp_path / "access.jsonl",
            purpose="Inspect failures",
            accessed_by="test-suite",
            code_commit="abc123",
            include_case_details=True,
        )

    assert not (tmp_path / "access.jsonl").exists()


def test_validation_can_emit_logged_case_details(tmp_path):
    protocol, _, _ = _protocol(tmp_path)
    result = _result(tmp_path / "validation-result.json", "val")

    artifact = evaluate_frozen_split(
        protocol_path=protocol,
        role="validation",
        result_path=result,
        output_path=tmp_path / "validation-output.json",
        ledger_path=tmp_path / "access.jsonl",
        purpose="Architecture selection checkpoint",
        accessed_by="test-suite",
        code_commit="abc123",
        include_case_details=True,
    )

    assert artifact["detail_policy"] == "case_details_explicitly_requested"
    assert [case["id"] for case in artifact["case_details"]] == [
        "val-fr",
        "val-ar",
        "val-negative",
    ]


def test_evaluation_fails_closed_on_dataset_drift_or_result_identity_errors(tmp_path):
    protocol, validation, _ = _protocol(tmp_path)
    result = _result(tmp_path / "validation-result.json", "val")
    common = {
        "protocol_path": protocol,
        "role": "validation",
        "result_path": result,
        "output_path": tmp_path / "output.json",
        "ledger_path": tmp_path / "access.jsonl",
        "purpose": "Controlled validation",
        "accessed_by": "test-suite",
        "code_commit": "abc123",
    }

    validation.write_text(validation.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        evaluate_frozen_split(**common)

    protocol, _, _ = _protocol(tmp_path)
    common["protocol_path"] = protocol
    value = json.loads(result.read_text(encoding="utf-8"))
    value["records"].append(value["records"][0])
    _write(result, value)
    with pytest.raises(ValueError, match="duplicate case IDs"):
        evaluate_frozen_split(**common)
