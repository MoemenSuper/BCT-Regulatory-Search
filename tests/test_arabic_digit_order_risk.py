from experiments.arabic_digit_order_risk import (
    contextual_implausible_years,
    document_digit_order_risk,
    source_year,
)


def test_source_year_is_parsed_only_for_arabic_bct_filename():
    assert source_year("Note_2018_06_ar.pdf") == 2018
    assert source_year("Note_2018_06_fr.pdf") is None


def test_contextual_impossible_years_are_flagged_without_treating_amounts_as_years():
    text = "تفتح يوم 07 جويلية 6102، ويبلغ السقف 5000 دينار لسنة 2016."

    assert contextual_implausible_years(text, document_year=2016) == ["6102"]


def test_historical_years_are_allowed():
    text = "المراجع: منشور لسنة 1989 ومذكرة سنة 2000."

    assert contextual_implausible_years(text, document_year=2018) == []


def test_document_level_risk_propagates_a_header_failure_to_the_document():
    document = {
        "filename": "Note_2017_18_ar.pdf",
        "language": "ar",
        "pages": [
            {
                "page_number": 1,
                "raw_text": "مذكرة عدد 18 لسنة 7112",
                "metadata": {},
            },
            {
                "page_number": 2,
                "raw_text": "وذلك على الأقل 74 ساعة قبل بداية مدة النيابة",
                "metadata": {},
            },
        ],
    }

    assert document_digit_order_risk(document) == {
        "requires_visual_fallback": True,
        "document_year": 2017,
        "page_hits": [{"page": 1, "suspicious_tokens": ["7112"]}],
    }
