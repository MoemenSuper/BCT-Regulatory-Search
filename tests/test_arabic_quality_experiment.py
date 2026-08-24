from experiments.arabic_quality_experiment import evaluate_arabic_gate


def _case(role, quality, latin, single):
    return {
        "role": role,
        "language": "ar",
        "diagnostics": {
            "quality_score": quality,
            "latin_character_ratio": latin,
            "single_arabic_token_ratio": single,
        },
    }


def test_arabic_gate_reports_current_and_proposed_confusion_matrices():
    suite = {
        "cases": [
            _case("failure", 1.0, 0.0, 0.20),
            _case("failure", 0.45, 0.0, 0.00),
            _case("control", 1.0, 0.0, 0.02),
            _case("control", 1.0, 0.3, 0.02),
        ]
        + [_case("control", 1.0, 0.0, 0.02) for _ in range(8)]
    }

    result = evaluate_arabic_gate(suite)

    assert result["current_gate"] == {
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 0,
        "true_negative": 10,
        "recall": 0.5,
        "false_positive_rate": 0.0,
    }
    assert result["proposed_gate"]["true_positive"] == 2
    assert result["proposed_gate"]["false_positive"] == 1
    assert result["proposed_gate"]["recall"] == 1.0
    assert result["decision"] == "KEEP_FOR_CONTROLLED_FALLBACK_COMPARISON"
