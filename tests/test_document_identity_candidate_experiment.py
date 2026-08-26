import pytest

from langchain_core.documents import Document

from experiments.document_identity_candidate_experiment import (
    _score_records,
    build_identity_reranker_documents,
    parse_query_identity,
    parse_source_identity,
    ranked_signature,
)


def _candidate(source: str, page: int, text: str = "original chunk"):
    return {
        "document": Document(
            page_content=text,
            metadata={"source": source, "page": page, "page_end": page},
        ),
        "representations": {"structured_baseline_chunking"},
        "ranks": {},
    }


def test_source_identity_parses_filename_metadata_without_gold():
    assert parse_source_identity(r"C:\public\Cir_2018_03_ar.pdf") == {
        "kind": "cir",
        "year": 2018,
        "number": 3,
        "language": "ar",
    }
    assert parse_source_identity("Note-2021-005-fr.pdf") == {
        "kind": "note",
        "year": 2021,
        "number": 5,
        "language": "fr",
    }
    assert parse_source_identity("unrecognized.pdf") is None


def test_full_instrument_identity_is_parsed_in_french_arabic_and_filename_form():
    assert parse_query_identity("Que prévoit la circulaire n° 2018-03 ?") == {
        "kind": "cir",
        "year": 2018,
        "number": 3,
        "route_reason": "explicit_instrument_identity",
    }
    assert parse_query_identity("ما أجل المنشور عدد 3 لسنة 2018؟") == {
        "kind": "cir",
        "year": 2018,
        "number": 3,
        "route_reason": "explicit_instrument_identity",
    }
    assert parse_query_identity("Consulter Note_2021_05_fr.pdf") == {
        "kind": "note",
        "year": 2021,
        "number": 5,
        "route_reason": "explicit_instrument_identity",
    }


def test_narrow_year_phrases_route_but_legal_dates_seasons_and_future_do_not():
    assert parse_query_identity("Quelles règles s'appliquaient en 2018 ?") == {
        "kind": None,
        "year": 2018,
        "number": None,
        "route_reason": "explicit_source_year",
    }
    assert parse_query_identity("Le billet type 2017 avait quelles dimensions ?")[
        "year"
    ] == 2017
    assert parse_query_identity("ما هي القواعد لسنة 2018؟")["year"] == 2018

    assert parse_query_identity("personnes visées par la loi n°2016-48") is None
    assert parse_query_identity("mesures adoptées le 19 mars 2020") is None
    assert parse_query_identity("campagne agricole 2017-2018") is None
    assert parse_query_identity("bénéfice de l'exercice 2023") is None
    assert parse_query_identity("fin 2017 puis fin 2018") is None
    assert parse_query_identity("le prochain texte sera publié en 2027") is None


def test_unrouted_queries_are_an_exact_reranker_document_noop():
    candidates = [
        _candidate("Cir_2018_03_ar.pdf", 3),
        _candidate("Cir_2019_03_ar.pdf", 3),
    ]

    documents = build_identity_reranker_documents(candidates, None)

    assert documents == [candidate["document"] for candidate in candidates]
    assert all(
        document is candidate["document"]
        for document, candidate in zip(documents, candidates, strict=True)
    )


def test_routed_prefix_preserves_candidate_budget_metadata_and_original_text():
    candidates = [
        _candidate("Cir_2018_03_ar.pdf", 3, "answer-bearing text"),
        _candidate("unknown.pdf", 1, "unparsed text"),
    ]
    identity = parse_query_identity("ما هي القواعد لسنة 2018؟")

    documents = build_identity_reranker_documents(candidates, identity)

    assert len(documents) == len(candidates)
    assert documents[0].page_content == (
        "[document kind=cir; year=2018; number=3; language=ar]\n"
        "answer-bearing text"
    )
    assert documents[0].metadata == candidates[0]["document"].metadata
    assert documents[1] is candidates[1]["document"]
    assert candidates[0]["document"].page_content == "answer-bearing text"


def test_ranked_signature_uses_only_runtime_source_page_and_score():
    ranked = [(_candidate("Cir_2018_03_ar.pdf", 3), 0.75)]
    assert ranked_signature(ranked) == [
        {"source": "Cir_2018_03_ar.pdf", "page": 3, "score": 0.75}
    ]


def test_gold_is_applied_only_after_runtime_rankings_are_frozen():
    runtime = [
        {
            "id": "case",
            "query_identity": {"route_reason": "explicit_source_year"},
            "control_undiversified_ranked": [
                {"source": "Cir_2019_03_ar.pdf", "page": 3, "score": 0.8}
            ],
            "control_ranked": [
                {"source": "Cir_2019_03_ar.pdf", "page": 3, "score": 0.8}
            ],
            "candidate_ranked": [
                {"source": "Cir_2018_03_ar.pdf", "page": 3, "score": 0.9}
            ],
        }
    ]
    evaluation = [
        {
            "id": "case",
            "language": "ar",
            "relevant": True,
            "expected_source": "Cir_2018_03_ar.pdf",
            "expected_page": 3,
        }
    ]
    frozen = {"rank_records": {"case": {"fusion_page": None, "diverse_page": None}}}

    scored = _score_records(runtime, evaluation, {"case": "failure"}, frozen)

    assert scored[0]["control_page_rank"] is None
    assert scored[0]["candidate_page_rank"] == 1


def test_french_control_validates_frozen_undiversified_rank_before_real_diversity():
    duplicate = {"source": "Other.pdf", "page": 1, "score": 0.85}
    expected = {"source": "Cir_2018_03_fr.pdf", "page": 3, "score": 0.8}
    runtime = [
        {
            "id": "case",
            "query_identity": None,
            "control_undiversified_ranked": [
                {"source": "Other.pdf", "page": 1, "score": 0.9},
                duplicate,
                expected,
            ],
            "control_ranked": [
                {"source": "Other.pdf", "page": 1, "score": 0.9},
                expected,
            ],
            "candidate_ranked": [
                {"source": "Other.pdf", "page": 1, "score": 0.9},
                expected,
            ],
        }
    ]
    evaluation = [
        {
            "id": "case",
            "language": "fr",
            "relevant": True,
            "expected_source": "Cir_2018_03_fr.pdf",
            "expected_page": 3,
        }
    ]
    frozen = {
        "rank_records": {"case": {"fusion_page": 3, "diverse_page": 3}}
    }

    scored = _score_records(runtime, evaluation, {}, frozen)

    assert scored[0]["control_fusion_page_rank"] == 3
    assert scored[0]["control_page_rank"] == 2

    runtime[0]["control_undiversified_ranked"] = runtime[0]["control_ranked"]
    with pytest.raises(ValueError, match="Undiversified control rank drift"):
        _score_records(runtime, evaluation, {}, frozen)
