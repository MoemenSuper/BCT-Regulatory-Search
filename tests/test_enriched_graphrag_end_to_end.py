from types import SimpleNamespace

import httpx
from groq import RateLimitError
from langchain_core.documents import Document

from experiments.enriched_graphrag_end_to_end import (
    GroqClientPool,
    _load_cases,
    _rate_limit_wait_seconds,
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


def test_load_cases_accepts_frozen_list_or_wrapped_suite():
    cases = [{"id": "case"}]

    assert _load_cases(cases) == cases
    assert _load_cases({"cases": cases}) == cases


def test_structured_diagnostics_falls_back_to_primary_expected_page():
    evidence = [{"evidence_id": "E1", "source": "A.pdf", "page": 2}]
    case = {
        "relevant": True,
        "expected_source": "A.pdf",
        "expected_page": 2,
        "expected_behavior": "answer",
    }
    response = {
        "status": "answered",
        "answer": "answer",
        "claims": [{"text": "claim", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1", "source": "A.pdf", "page": 2}],
    }

    diagnostics = structured_diagnostics(case, response, evidence)

    assert diagnostics["complete_required_page_citations"] is True
    assert diagnostics["required_page_citation_recall"] == 1.0
    assert diagnostics["status_expected"] is True


def test_structured_diagnostics_scores_negative_expected_status_without_gold_page():
    case = {
        "relevant": False,
        "expected_source": None,
        "expected_page": None,
        "expected_behavior": "reject_out_of_scope",
    }
    response = {
        "status": "out_of_scope",
        "answer": "not a BCT question",
        "claims": [],
        "citations": [],
    }

    diagnostics = structured_diagnostics(case, response, [])

    assert diagnostics["status_expected"] is True
    assert diagnostics["complete_required_page_citations"] is True


def test_rate_limit_wait_honors_provider_reset_longer_than_one_minute():
    error = SimpleNamespace(
        response=SimpleNamespace(headers={"retry-after": "812.592"})
    )

    assert _rate_limit_wait_seconds(error) == 812.592


def _rate_limit_error(wait_seconds):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        headers={"retry-after": str(wait_seconds)},
    )
    return RateLimitError("rate limited", response=response, body={})


class _StubCompletions:
    def __init__(self, events):
        self.events = list(events)
        self.call_count = 0

    def create(self, **_kwargs):
        self.call_count += 1
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


def _stub_client(events):
    completions = _StubCompletions(events)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions_stub=completions,
    )


def test_groq_client_pool_rotates_on_429_then_stays_on_healthy_slot():
    first = _stub_client([_rate_limit_error(600)])
    second = _stub_client(["second-1", "second-2"])
    pool = GroqClientPool(
        [("GROQ_API_KEY", first), ("GROQ_API_KEY_2", second)],
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )

    assert pool.create(model="model") == ("second-1", "GROQ_API_KEY_2")
    assert pool.create(model="model") == ("second-2", "GROQ_API_KEY_2")
    assert first.completions_stub.call_count == 1
    assert second.completions_stub.call_count == 2


def test_groq_client_pool_waits_for_earliest_slot_when_all_are_limited():
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    first = _stub_client([_rate_limit_error(10), "first-reset"])
    second = _stub_client([_rate_limit_error(20)])
    pool = GroqClientPool(
        [("GROQ_API_KEY", first), ("GROQ_API_KEY_2", second)],
        clock=lambda: now[0],
        sleeper=sleep,
    )

    assert pool.create(model="model") == ("first-reset", "GROQ_API_KEY")
    assert sleeps == [10.0]
