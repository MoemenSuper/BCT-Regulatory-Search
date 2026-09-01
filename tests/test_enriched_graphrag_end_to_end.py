from langchain_core.documents import Document

from experiments.enriched_graphrag_end_to_end import (
    graph_candidate,
    runtime_inputs,
    seed_documents,
    structured_diagnostics,
)


def test_runtime_inputs_remove_all_gold_fields():
    inputs = runtime_inputs(
        [
            {
                "id": "case",
                "query": "question",
                "language": "fr",
                "category": "relationship",
                "expected_answer": "secret",
                "expected_sources": [{"source": "secret.pdf", "pages": [2]}],
            }
        ]
    )

    assert inputs == [
        {
            "id": "case",
            "query": "question",
            "language": "fr",
            "category": "relationship",
        }
    ]


def test_graph_candidate_normalizes_zero_based_runtime_page_to_evaluation_page():
    candidate = graph_candidate(
        Document(
            page_content="verified graph evidence",
            metadata={
                "source": "Circular.pdf",
                "page": 2,
                "page_label": 3,
                "retrieval_source": "neo4j_relationship",
            },
        )
    )

    assert candidate["document"].metadata["page"] == 3
    assert candidate["document"].metadata["page_label"] == 3
    assert candidate["representations"] == {"neo4j_relationship"}


def test_seed_documents_preserve_one_based_page_label_for_graph_runtime():
    candidate = {
        "document": Document(
            page_content="ordinary evidence",
            metadata={"source": "Circular.pdf", "page": 3},
        )
    }

    seed = seed_documents([candidate])[0]

    assert seed.metadata["page"] == 2
    assert seed.metadata["page_label"] == 3


def test_structured_diagnostics_require_every_expected_source_page_citation():
    evidence = [
        {"evidence_id": "E1", "source": "A.pdf", "page": 2},
        {"evidence_id": "E2", "source": "B.pdf", "page": 4},
    ]
    case = {
        "expected_sources": [
            {"source": "A.pdf", "pages": [2]},
            {"source": "B.pdf", "pages": [4]},
        ]
    }
    response = {
        "status": "answered",
        "answer": "answer",
        "claims": [{"text": "claim", "evidence_ids": ["E1"]}],
        "citations": [
            {"evidence_id": "E1", "source": "A.pdf", "page": 2},
        ],
    }

    diagnostics = structured_diagnostics(case, response, evidence)

    assert diagnostics["citations_match_evidence"] is True
    assert diagnostics["claim_evidence_links_valid"] is True
    assert diagnostics["complete_required_page_citations"] is False
