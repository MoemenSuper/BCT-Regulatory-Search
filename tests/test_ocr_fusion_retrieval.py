from experiments.ocr_fusion_retrieval import (
    is_arabic_query,
    page_risk_diagnostics,
    secondary_ocr_chunks,
)


def test_page_risk_diagnostics_match_the_frozen_gate_features():
    fragmented = page_risk_diagnostics("ا ل ب ن ك المركزي", 1.0)
    clean = page_risk_diagnostics("البنك المركزي التونسي", 1.0)
    latin_heavy = page_risk_diagnostics("abc def نص", 1.0)

    assert fragmented["single_arabic_token_ratio"] >= 0.10
    assert fragmented["requires_fallback"] is True
    assert clean["requires_fallback"] is False
    assert latin_heavy["latin_character_ratio"] > 0.20
    assert latin_heavy["requires_fallback"] is True


def test_secondary_chunks_are_page_local_and_keep_provenance():
    chunks = secondary_ocr_chunks(
        source="Note_2026_01_ar.pdf",
        page_number=3,
        text="كلمة " * 300,
        max_chars=1000,
        overlap=200,
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "Note_2026_01_ar.pdf" for chunk in chunks)
    assert all(chunk.metadata["pages"] == [3] for chunk in chunks)
    assert all(chunk.metadata["representation"] == "arabic_ocr_secondary" for chunk in chunks)
    assert all(chunk.metadata["extraction_methods"] == "ocr" for chunk in chunks)


def test_arabic_query_detection_uses_runtime_text_not_evaluation_labels():
    assert is_arabic_query("ما هو الأجل القانوني؟") is True
    assert is_arabic_query("Quel est le délai légal ?") is False

