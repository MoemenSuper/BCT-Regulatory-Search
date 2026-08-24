from experiments.stress_suite import build_extraction_stress_suite


def _case(case_id, language, source, category):
    return {
        "id": case_id,
        "language": language,
        "category": category,
        "relevant": True,
        "expected_source": source,
        "expected_page": 1,
        "evidence_quote": "نسبة 6 بالمائة" if language == "ar" else "taux de 6 pour cent",
    }


def _record(case_id, language, rank, failures):
    return {
        "id": case_id,
        "language": language,
        "relevant": True,
        "result": {"exact_page_rank": rank},
        "failure_categories": failures,
    }


def test_extraction_suite_pairs_failures_with_language_category_controls():
    evaluation = [
        _case("ar-failure", "ar", "Note_2026_01_ar.pdf", "amount_or_rate"),
        _case("ar-control", "ar", "Note_2026_02_ar.pdf", "amount_or_rate"),
        _case("fr-failure", "fr", "Cir_2026_01_fr.pdf", "definition"),
        _case("fr-control", "fr", "Cir_2026_02_fr.pdf", "definition"),
    ]
    result = {
        "records": [
            _record("ar-failure", "ar", None, ["evidence_missing_because_of_extraction"]),
            _record("ar-control", "ar", 1, []),
            _record("fr-failure", "fr", None, ["evidence_missing_because_of_extraction"]),
            _record("fr-control", "fr", 2, []),
        ]
    }
    pages = {
        ("note_2026_01_ar.pdf", 1): {
            "raw_text": "ن س ب ة 6 بالمائة",
            "quality_score": 1.0,
            "quality_flags": [],
            "extraction_method": "native",
        },
        ("note_2026_02_ar.pdf", 1): {
            "raw_text": "نسبة 6 بالمائة",
            "quality_score": 1.0,
            "quality_flags": [],
            "extraction_method": "native",
        },
        ("cir_2026_01_fr.pdf", 1): {
            "raw_text": "texte corrompu",
            "quality_score": 1.0,
            "quality_flags": [],
            "extraction_method": "native",
        },
        ("cir_2026_02_fr.pdf", 1): {
            "raw_text": "taux de 6 pour cent",
            "quality_score": 1.0,
            "quality_flags": [],
            "extraction_method": "native",
        },
    }

    suite = build_extraction_stress_suite(evaluation, result, pages)

    assert suite["counts"] == {
        "ar": {"failure": 1, "control": 1},
        "fr": {"failure": 1, "control": 1},
    }
    by_id = {case["id"]: case for case in suite["cases"]}
    assert by_id["ar-failure"]["role"] == "failure"
    assert by_id["ar-control"]["role"] == "control"
    assert by_id["ar-failure"]["diagnostics"]["single_arabic_token_ratio"] > 0
    assert by_id["ar-control"]["diagnostics"]["evidence_token_coverage"] == 1.0
