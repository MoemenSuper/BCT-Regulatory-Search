import json

import pytest

from experiments.vlm_numeric_experiment import (
    evaluate_vlm_numeric_cases,
    parse_vlm_payload,
)


def test_parse_vlm_payload_requires_exact_typed_inventory():
    value = parse_vlm_payload(
        json.dumps(
            {
                "items": [
                    {
                        "literal": "١٥",
                        "kind": "number",
                        "context": "في أجل ١٥ يوما",
                        "uncertain": False,
                    }
                ]
            }
        )
    )

    assert value["items"][0]["literal"] == "١٥"

    with pytest.raises(ValueError, match="items"):
        parse_vlm_payload('{"items": "not-a-list"}')
    with pytest.raises(ValueError, match="literal"):
        parse_vlm_payload('{"items": [{"kind": "number", "context": "x", "uncertain": false}]}')


def test_vlm_evaluation_scores_frozen_expected_literals_by_page():
    suite = {
        "cases": [
            {
                "id": "case-1",
                "expected_source": "Note_2026_01_ar.pdf",
                "expected_page": 1,
                "expected_numbers": ["15", "2026"],
                "expected_identifiers": ["abc-123"],
                "native": {"number_recall": 0.0, "identifier_recall": 1.0},
                "ocr": {"number_recall": 0.5, "identifier_recall": 0.0},
            }
        ]
    }
    pages = {
        ("note_2026_01_ar.pdf", 1): {
            "items": [
                {
                    "literal": "15",
                    "kind": "number",
                    "context": "الأجل 15 يوما في 2026",
                    "uncertain": False,
                },
                {
                    "literal": "2026",
                    "kind": "date",
                    "context": "في سنة 2026",
                    "uncertain": False,
                },
                {
                    "literal": "ABC-123",
                    "kind": "identifier",
                    "context": "الرمز ABC-123",
                    "uncertain": False,
                },
            ],
            "latency_seconds": 1.0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    }

    result = evaluate_vlm_numeric_cases(suite, pages)

    assert result["metrics"]["vlm"]["mean_number_recall"] == 1.0
    assert result["metrics"]["vlm"]["mean_identifier_recall"] == 1.0
    assert result["metrics"]["usage"]["total_tokens"] == 30
    assert result["vlm_number_improvements"] == ["case-1"]
    assert result["vlm_number_regressions"] == []
    assert result["vlm_number_unchanged_count"] == 0
    assert result["decision"] == "KEEP_FOR_FULL_TEXT_VLM_ABLATION"
