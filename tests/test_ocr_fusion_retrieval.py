from experiments.ocr_fusion_retrieval import (
    build_slim_result,
    candidate_pool_outcome,
    is_arabic_query,
    page_risk_diagnostics,
    secondary_ocr_chunks,
)
from langchain_core.documents import Document


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


def test_candidate_pool_outcome_separates_source_and_exact_page_hits():
    candidates = [
        {
            "document": Document(
                page_content="text",
                metadata={"source": "Note_2026_01_ar.pdf", "pages": [2, 3]},
            )
        }
    ]

    assert candidate_pool_outcome(candidates, "note_2026_01_ar.pdf", 3) == {
        "source_hit": True,
        "exact_page_hit": True,
    }
    assert candidate_pool_outcome(candidates, "Note_2026_01_ar.pdf", 4) == {
        "source_hit": True,
        "exact_page_hit": False,
    }


def test_slim_result_keeps_all_ranks_without_retrieved_text():
    full = {
        "status": "complete",
        "timestamp": "2026-08-24T00:00:00+00:00",
        "configuration": {"name": "fusion"},
        "inputs": {"evaluation_sha256": "A" * 64},
        "summary": {"repairs": ["case-1"], "regressions": []},
        "limitations": ["development only"],
        "records": [
            {
                "id": "case-1",
                "language": "ar",
                "relevant": True,
                "baseline": {"source_rank": None, "exact_page_rank": None},
                "result": {
                    "source_rank": 1,
                    "exact_page_rank": 1,
                    "top5": [{"chunk_text": "large evidence text"}],
                },
                "repaired": True,
                "regressed": False,
            }
        ],
    }
    candidates = {
        "status": "complete",
        "metrics": {"ar": {"union_exact_page_hit": 1.0}},
        "exact_page_candidate_rescues": ["case-1"],
        "rescue_failure_categories": {"extraction": 1},
    }

    slim = build_slim_result(
        full,
        candidates,
        full_result_sha256="B" * 64,
        candidate_analysis_sha256="C" * 64,
    )

    assert slim["rank_records"]["case-1"]["fusion_page"] == 1
    assert slim["rank_records"]["case-1"]["change"] == "repair"
    assert "large evidence text" not in str(slim)
    assert slim["deployment_status"] == "PROHIBITED_PENDING_UNSEEN_VALIDATION"
