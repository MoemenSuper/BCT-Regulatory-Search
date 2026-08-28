import pytest

from experiments.graph_temporal_benchmark import (
    apply_audited_corrections,
    graph_readiness,
    score_case,
    select_graph_temporal_cases,
)


def _case(case_id="graph_case", *, language="fr"):
    return {
        "id": case_id,
        "query": "question",
        "language": language,
        "category": "multi_document_synthesis",
        "evaluation_slice": "graph_temporal",
        "relevant": True,
        "requires_graph": True,
        "expected_source": "new.pdf",
        "expected_page": 4,
        "expected_sources": [
            {"source": "old.pdf", "pages": [2]},
            {"source": "new.pdf", "pages": [4]},
        ],
        "evidence_quotes": [],
    }


def test_selects_only_graph_temporal_cases_and_rejects_duplicate_ids():
    selected = select_graph_temporal_cases(
        [
            _case(),
            {**_case("old"), "requires_graph": False},
            {**_case("other"), "evaluation_slice": "other"},
        ]
    )

    assert [case["id"] for case in selected] == ["graph_case"]

    with pytest.raises(ValueError, match="duplicate"):
        select_graph_temporal_cases([_case(), _case()])


def test_audited_corrections_change_only_the_known_old_source_page_and_quote():
    cases = [
        {
            **_case("graph_fr_2019_07_multi_office_02"),
            "expected_sources": [
                {"source": "Cir_2018_07_fr.pdf", "pages": [2]},
                {"source": "Cir_2019_07_fr.pdf", "pages": [3]},
            ],
            "evidence_quote": "wrong page quote",
            "evidence_quotes": [
                {
                    "source": "Cir_2018_07_fr.pdf",
                    "page": 2,
                    "quote": "wrong page quote",
                },
                {
                    "source": "Cir_2019_07_fr.pdf",
                    "page": 3,
                    "quote": "new rule",
                },
            ],
        },
        _case("untouched"),
    ]

    corrected, receipt = apply_audited_corrections(cases)

    changed = corrected[0]
    assert changed["expected_sources"][0]["pages"] == [3]
    assert changed["evidence_quotes"][0]["page"] == 3
    assert "Article 3" in changed["evidence_quotes"][0]["quote"]
    assert changed["evidence_quote"] == changed["evidence_quotes"][0]["quote"]
    assert corrected[1] == cases[1]
    assert receipt[0]["before_pages"] == [2]
    assert receipt[0]["after_pages"] == [3]


def test_corrections_fail_if_the_supplied_gold_no_longer_matches_audited_input():
    case = {
        **_case("graph_fr_2019_07_start_deadline_03"),
        "expected_sources": [
            {"source": "Cir_2018_07_fr.pdf", "pages": [9]},
            {"source": "Cir_2019_07_fr.pdf", "pages": [3]},
        ],
        "evidence_quotes": [
            {"source": "Cir_2018_07_fr.pdf", "page": 9, "quote": "drift"}
        ],
    }

    with pytest.raises(ValueError, match="audited page 2"):
        apply_audited_corrections([case])


def test_score_case_reports_primary_and_complete_multi_source_coverage():
    case = _case()
    candidate_pages = [
        {"source": "old.pdf", "page": 2},
        {"source": "new.pdf", "page": 4},
    ]
    ranked_pages = [
        {"source": "new.pdf", "page": 4, "score": 0.9},
        {"source": "old.pdf", "page": 3, "score": 0.8},
        {"source": "old.pdf", "page": 2, "score": 0.7},
    ]

    scored = score_case(case, candidate_pages, ranked_pages)

    assert scored["primary_page_rank"] == 1
    assert scored["required_source_recall_at_5"] == 1.0
    assert scored["complete_required_sources_at_5"] is True
    assert scored["required_page_pair_recall_at_1"] == 0.5
    assert scored["required_page_pair_recall_at_5"] == 1.0
    assert scored["complete_required_page_pairs_at_5"] is True
    assert scored["missing_required_page_pairs_from_candidates"] == []


def test_graph_readiness_requires_coverage_retriever_and_context_assembly():
    cases = [_case()]

    partial = graph_readiness(
        cases,
        local_graph_sources={"old.pdf"},
        runtime_retriever_exists=False,
        answer_context_assembler_exists=False,
    )
    ready = graph_readiness(
        cases,
        local_graph_sources={"old.pdf", "new.pdf"},
        runtime_retriever_exists=True,
        answer_context_assembler_exists=True,
    )

    assert partial["cases_with_any_required_source"] == 1
    assert partial["cases_with_all_required_sources"] == 0
    assert partial["decision"] == "GRAPH_ARM_NOT_READY"
    assert ready["decision"] == "GRAPH_ARM_READY"

