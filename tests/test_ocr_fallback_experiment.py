from experiments.ocr_fallback_experiment import (
    evaluate_variants,
    select_triggered_arabic_cases,
)


def _case(case_id, role, *, quality=1.0, latin=0.0, fragments=0.0, page=1):
    return {
        "id": case_id,
        "role": role,
        "language": "ar",
        "expected_source": "Note_2026_01_ar.pdf",
        "expected_page": page,
        "diagnostics": {
            "quality_score": quality,
            "latin_character_ratio": latin,
            "single_arabic_token_ratio": fragments,
        },
    }


def test_selects_only_arabic_cases_triggered_by_the_frozen_gate():
    suite = {
        "cases": [
            _case("low-quality", "failure", quality=0.4),
            _case("latin-heavy", "failure", latin=0.3, page=2),
            _case("fragmented", "control", fragments=0.1, page=3),
            _case("clean", "control", page=4),
            {
                **_case("french", "failure", quality=0.1, page=5),
                "language": "fr",
            },
        ]
    }

    selected = select_triggered_arabic_cases(suite)

    assert [case["id"] for case in selected] == [
        "low-quality",
        "latin-heavy",
        "fragmented",
    ]


def test_evaluation_scores_evidence_tokens_numbers_and_predeclared_gate():
    cases = [
        {**_case("failure", "failure"), "evidence_quote": "الأجل 15 يوما سنة 2026"},
        {**_case("control", "control", page=2), "evidence_quote": "قيمة 100 دينار"},
    ]
    variants = {
        ("Note_2026_01_ar.pdf", 1): {
            "native": {"text": "الأجل معكوس 51 يوما سنة 6202", "elapsed_seconds": 0.0},
            "auto_ocr": {"text": "deadline 15 days", "elapsed_seconds": 1.0},
            "arabic_rapidocr": {
                "text": "الأجل 15 يوما سنة 2026",
                "elapsed_seconds": 2.0,
            },
        },
        ("Note_2026_01_ar.pdf", 2): {
            "native": {"text": "قيمة 100 دينار", "elapsed_seconds": 0.0},
            "auto_ocr": {"text": "value 100", "elapsed_seconds": 1.0},
            "arabic_rapidocr": {
                "text": "قيمة 100 دينار",
                "elapsed_seconds": 2.0,
            },
        },
    }

    result = evaluate_variants(cases, variants)

    explicit = result["aggregates"]["failure"]["arabic_rapidocr"]
    assert explicit["mean_evidence_token_coverage"] == 1.0
    assert explicit["mean_critical_number_recall"] == 1.0
    assert result["comparisons"]["arabic_vs_native_failure_coverage_delta"] > 0.15
    assert result["decision"] == "KEEP_FOR_CACHED_RETRIEVAL_ABLATION"
    assert result["limitations"]

