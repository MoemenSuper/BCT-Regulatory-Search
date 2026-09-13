from ingestion.quality import assess_page_quality, contains_sensitive_literals


def test_empty_native_page_requires_visual_fallback():
    quality = assess_page_quality("", 0)
    assert quality.requires_fallback is True
    assert quality.score == 0.0
    assert "no_native_text" in quality.flags


def test_normal_regulatory_text_does_not_trigger_obvious_corruption_gate():
    text = (
        "Article 12. La Banque Centrale de Tunisie peut retirer l'agrément "
        "lorsque l'établissement ne remplit plus les conditions prévues."
    )
    quality = assess_page_quality(text, 2)
    assert quality.requires_fallback is False
    assert quality.score >= 0.55


def test_sensitive_literal_detector_catches_dates_amounts_and_percentages():
    assert contains_sensitive_literals("La date est fixée au 11/10/2026.")
    assert contains_sensitive_literals("Le taux applicable est de 7%.")
    assert contains_sensitive_literals("المبلغ يساوي 1000 دينار")
    assert not contains_sensitive_literals("Disposition générale sans valeur chiffrée.")
