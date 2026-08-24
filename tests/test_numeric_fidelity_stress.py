from experiments.numeric_fidelity_stress import (
    critical_identifiers,
    critical_numbers,
    evaluate_numeric_fidelity,
)


def test_critical_numbers_normalize_western_and_arabic_indic_digits():
    assert critical_numbers("الأجل ١٥ يوما إلى 2026/07/15 وبنسبة ١٢٫٥٪") == {
        "12.5",
        "15",
        "2026",
        "07",
    }


def test_identifiers_require_letters_and_digits():
    assert critical_identifiers("الرمز ABC-123 والمرجع 2026") == {"abc-123"}


def test_fidelity_keeps_native_and_ocr_separate_and_scores_union():
    result = evaluate_numeric_fidelity(
        expected="الأجل 15 يوما في 2026 والرمز ABC-123",
        native="الأجل 51 يوما في 6202 والرمز ABC-123",
        ocr="الأجل 15 يوما في 2026 والرمز ABC-132",
    )

    assert result["native"]["number_recall"] == 0.0
    assert result["ocr"]["number_recall"] == 1.0
    assert result["native"]["identifier_recall"] == 1.0
    assert result["ocr"]["identifier_recall"] == 0.0
    assert result["union"]["number_recall"] == 1.0
    assert result["union"]["identifier_recall"] == 1.0

