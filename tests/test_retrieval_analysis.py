import pytest

from experiments.retrieval_analysis import analyze_retrieval


def _candidate(source, page, *, dense_rank=None, bm25_rank=None):
    ranks = {"structured": {}}
    if dense_rank is not None:
        ranks["structured"]["dense"] = dense_rank
    if bm25_rank is not None:
        ranks["structured"]["bm25"] = bm25_rank
    return {
        "document": {
            "page_content": "evidence",
            "metadata": {"source": source, "pages": [page]},
        },
        "representations": ["structured"],
        "ranks": ranks,
    }


def test_analysis_separates_candidate_recall_from_ranking_by_language():
    evaluation = [
        {
            "id": "fr-hit",
            "language": "fr",
            "relevant": True,
            "expected_source": "Cir_2026_01_fr.pdf",
            "expected_page": 2,
        },
        {
            "id": "ar-ranked-low",
            "language": "ar",
            "relevant": True,
            "expected_source": "Note_2026_02_ar.pdf",
            "expected_page": 3,
        },
        {
            "id": "ar-missing",
            "language": "ar",
            "relevant": True,
            "expected_source": "Note_2026_03_ar.pdf",
            "expected_page": 1,
        },
        {"id": "negative", "language": "fr", "relevant": False},
    ]
    result = {
        "records": [
            {
                "id": "fr-hit",
                "language": "fr",
                "relevant": True,
                "result": {"source_rank": 1, "exact_page_rank": 1},
                "failure_categories": [],
            },
            {
                "id": "ar-ranked-low",
                "language": "ar",
                "relevant": True,
                "result": {"source_rank": 2, "exact_page_rank": 6},
                "failure_categories": ["correct_evidence_retrieved_but_ranked_too_low"],
                "primary_failure_category": "correct_evidence_retrieved_but_ranked_too_low",
            },
            {
                "id": "ar-missing",
                "language": "ar",
                "relevant": True,
                "result": {"source_rank": None, "exact_page_rank": None},
                "failure_categories": ["correct_document_missing_from_candidate_set"],
                "primary_failure_category": "correct_document_missing_from_candidate_set",
            },
            {
                "id": "negative",
                "language": "fr",
                "relevant": False,
                "result": {"source_rank": None, "exact_page_rank": None},
                "failure_categories": [],
                "primary_failure_category": "ambiguous_or_insufficient_query",
            },
        ]
    }
    candidate_cache = {
        "fr-hit": {
            "candidates": [
                _candidate("C:/corpus/Cir_2026_01_fr.pdf", 2, dense_rank=1, bm25_rank=4)
            ]
        },
        "ar-ranked-low": {
            "candidates": [
                _candidate("Note_2026_02_ar.pdf", 3, dense_rank=8, bm25_rank=2)
            ]
        },
        "ar-missing": {
            "candidates": [_candidate("Note_2025_03_ar.pdf", 1, dense_rank=1)],
        },
        "negative": {"candidates": []},
    }

    analysis = analyze_retrieval(evaluation, result, candidate_cache)

    overall = analysis["groups"]["overall"]
    assert overall["candidate"]["n"] == 3
    assert overall["candidate"]["exact_page_pool_recall"] == 2 / 3
    assert overall["candidate"]["dense_exact_page_recall_at_5"] == 1 / 3
    assert overall["candidate"]["dense_exact_page_recall_at_10"] == 2 / 3
    assert overall["candidate"]["bm25_exact_page_recall_at_5"] == 2 / 3
    assert overall["ranking"]["exact_page_at_5"] == 1 / 3
    assert overall["ranking"]["exact_page_at_10"] == 2 / 3

    assert analysis["groups"]["fr"]["candidate"]["exact_page_pool_recall"] == 1.0
    assert analysis["groups"]["ar"]["candidate"]["exact_page_pool_recall"] == 0.5
    assert analysis["groups"]["ar"]["primary_failures"] == {
        "correct_document_missing_from_candidate_set": 1,
        "correct_evidence_retrieved_but_ranked_too_low": 1,
    }
    assert analysis["groups"]["overall"]["negative_or_ambiguous"] == {
        "case_count": 1,
        "categories": {"unknown": 1},
        "primary_categories": {"ambiguous_or_insufficient_query": 1},
    }

    incomplete_result = {"records": result["records"][:-1]}
    with pytest.raises(ValueError, match="missing 1 evaluation cases"):
        analyze_retrieval(evaluation, incomplete_result, candidate_cache)

    duplicate_result = {"records": [*result["records"], result["records"][0]]}
    with pytest.raises(ValueError, match="duplicate case IDs"):
        analyze_retrieval(evaluation, duplicate_result, candidate_cache)

    extra_result = {
        "records": [
            *result["records"],
            {
                "id": "unexpected",
                "result": {"source_rank": None, "exact_page_rank": None},
            },
        ]
    }
    with pytest.raises(ValueError, match="1 unexpected evaluation cases"):
        analyze_retrieval(evaluation, extra_result, candidate_cache)
