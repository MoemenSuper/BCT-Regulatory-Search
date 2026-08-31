from datetime import date

from langchain_core.documents import Document
import pytest

import conversation
from regulatory_graph.fixtures import circular_2016_03_fr_bundle
from regulatory_graph.neo4j_store import (
    AffectedProvision,
    LineageEntry,
    RelationshipEvidence,
    TemporalResolution,
)
from regulatory_graph.runtime import (
    GraphRetrievalResult,
    GraphRetrievalStatus,
    GraphRetrievalTrace,
    RelationshipGraphRetriever,
    TemporalRetrievalStatus,
    is_relationship_query,
    is_temporal_rule_query,
)


class FakeRelationshipGraph:
    def __init__(self, evidence=(), error=None):
        self.evidence = evidence
        self.error = error
        self.calls = []

    def relationship_evidence(self, seed_filenames, *, limit):
        self.calls.append((seed_filenames, limit))
        if self.error is not None:
            raise self.error
        return self.evidence


def _document(filename, page=0, text="ordinary evidence"):
    return Document(
        page_content=text,
        metadata={"source": f"documents/{filename}", "page": page},
    )


@pytest.mark.parametrize(
    "query",
    (
        "Quels documents citent la circulaire 2013-15 ?",
        "Quelle circulaire remplace ce document ?",
        "Which documents cite Circular 2013-15?",
        "Does this note amend Circular 2013-15?",
        "Which documents are cited by Circular 2017-08?",
        "Which circular was replaced by Circular 2016-03?",
        "How are Circular 2017-08 and Circular 2013-15 related?",
        "Comment la circulaire 2017-08 est-elle liée à la circulaire 2013-15 ?",
        "ما العلاقة بين المنشورين؟",
        "كيف يرتبط هذا المنشور بالمذكرة؟",
        "ما هي المذكرة التي تلغي هذا المنشور؟",
    ),
)
def test_relationship_intent_covers_incoming_and_outgoing_multilingual_queries(query):
    assert is_relationship_query(query) is True


@pytest.mark.parametrize(
    "query",
    (
        "Comment modifier mes informations de contact ?",
        "Replace the email address in my profile.",
        "Replace the document attached to my profile.",
        "What is the relationship between interest rates and inflation?",
        "Quel est le ratio minimum requis ?",
        "كيف أعدل معلومات الاتصال الخاصة بي؟",
    ),
)
def test_non_document_actions_do_not_trigger_relationship_retrieval(query):
    assert is_relationship_query(query) is False


@pytest.mark.parametrize(
    "query",
    (
        "What rule is currently in force under Circular 2016-03?",
        "Quelle règle est en vigueur dans la circulaire 2016-03 ?",
        "ما هي القاعدة السارية في المنشور 2016-03؟",
        "How has Article 4 changed across years?",
    ),
)
def test_temporal_rule_intent_is_multilingual_and_provision_scoped(query):
    assert is_temporal_rule_query(query) is True


@pytest.mark.parametrize(
    "query",
    (
        "What is the current account deficit?",
        "Update my current contact details.",
        "What was inflation in 2017?",
    ),
)
def test_non_regulatory_current_or_historical_questions_are_not_temporal_rule_queries(query):
    assert is_temporal_rule_query(query) is False


def test_non_relationship_query_never_calls_neo4j():
    graph = FakeRelationshipGraph()
    retriever = RelationshipGraphRetriever(graph)

    result = retriever.retrieve(
        "Quel est le ratio minimum requis ?",
        (_document("Cir_2016_03_fr.pdf"),),
    )

    assert is_relationship_query("Quel est le ratio minimum requis ?") is False
    assert result.documents == ()
    assert result.trace.status == GraphRetrievalStatus.NOT_REQUESTED
    assert graph.calls == []


@pytest.mark.parametrize(
    "query",
    (
        "What did it replace?",
        "What is its predecessor?",
        "Que remplace-t-elle ?",
        "Quel est son prédécesseur ?",
        "ما الذي يلغيه؟",
        "ما الذي يعدله؟",
    ),
)
def test_anaphoric_relationship_follow_up_uses_available_regulatory_seed(query):
    graph = FakeRelationshipGraph()
    retriever = RelationshipGraphRetriever(graph)

    result = retriever.retrieve(
        query,
        (_document("Cir_2016_03_fr.pdf"),),
    )

    assert is_relationship_query(query) is False
    assert result.trace.status == GraphRetrievalStatus.NO_EVIDENCE
    assert graph.calls == [(('Cir_2016_03_fr.pdf',), 10)]


def test_anaphoric_relationship_follow_up_without_a_seed_makes_no_graph_call():
    graph = FakeRelationshipGraph()

    result = RelationshipGraphRetriever(graph).retrieve(
        "What did it replace?",
        (),
    )

    assert result.trace.status == GraphRetrievalStatus.NOT_REQUESTED
    assert graph.calls == []


def test_relationship_query_returns_bounded_source_linked_graph_documents():
    evidence = RelationshipEvidence(
        seed_filename="CB_2017_08_FR.pdf",
        seed_instrument_uid="BCT:CIRCULAR:2017:08",
        related_instrument_uid="BCT:CIRCULAR:2013:15",
        relation_kind="CITES",
        direction="OUTGOING",
        fact_uid="reference:2017-08:2013-15",
        evidence_uid="evidence:2017-08:p28:2013-15",
        filename="CB_2017_08_FR.pdf",
        page_number=28,
        quote="La presente circulaire cite la circulaire n 2013-15.",
    )
    graph = FakeRelationshipGraph((evidence,))
    retriever = RelationshipGraphRetriever(graph, max_seeds=2, max_evidence=4)

    result = retriever.retrieve(
        "Quelle relation entre ces circulaires ?",
        (
            _document("CB_2017_08_FR.pdf"),
            _document("CB_2017_08_FR.pdf", page=1),
            _document("Cir_2016_03_fr.pdf"),
            _document("ignored.pdf"),
        ),
    )

    assert is_relationship_query("ما العلاقة بين المنشورين؟") is True
    assert graph.calls == [
        (("CB_2017_08_FR.pdf", "Cir_2016_03_fr.pdf"), 4)
    ]
    assert result.trace.status == GraphRetrievalStatus.EXPANDED
    assert result.trace.evidence_count == 1
    assert result.trace.paths == (evidence.path,)
    assert len(result.documents) == 1
    graph_document = result.documents[0]
    assert evidence.quote in graph_document.page_content
    assert evidence.path in graph_document.page_content
    assert graph_document.metadata == {
        "source": "CB_2017_08_FR.pdf",
        "page": 27,
        "page_label": 28,
        "retrieval_source": "neo4j_relationship",
        "graph_direction": "OUTGOING",
        "graph_fact_uid": "reference:2017-08:2013-15",
        "graph_path": evidence.path,
        "graph_relation": "CITES",
    }


def test_graph_unavailability_falls_closed_without_candidate_documents():
    graph = FakeRelationshipGraph(error=RuntimeError("database offline"))

    result = RelationshipGraphRetriever(graph).retrieve(
        "Which circular replaces this one?",
        (_document("Cir_2016_03_fr.pdf"),),
    )

    assert result.documents == ()
    assert result.trace.status == GraphRetrievalStatus.UNAVAILABLE
    assert result.trace.error_type == "RuntimeError"


def test_graph_unavailability_requires_abstention_for_temporal_relationship_query():
    graph = FakeRelationshipGraph(error=RuntimeError("database offline"))

    result = RelationshipGraphRetriever(
        graph,
        current_date=date(2026, 8, 31),
    ).retrieve(
        "Which current provision replaces Article 4 in Circular 1991-24?",
        (_document("Cir_2016_03_fr.pdf", page=1),),
    )

    assert result.documents == ()
    assert result.trace.status == GraphRetrievalStatus.UNAVAILABLE
    assert result.trace.temporal_status == TemporalRetrievalStatus.UNAVAILABLE
    assert result.requires_temporal_abstention is True


def test_current_rule_query_returns_one_complete_verified_provision_version():
    version = circular_2016_03_fr_bundle().provision_versions[0]

    class CompleteTemporalGraph(FakeRelationshipGraph):
        def __init__(self):
            super().__init__()
            self.temporal_calls = []

        def affected_provisions(self, seeds, *, limit):
            self.temporal_calls.append(("affected", seeds, limit))
            return (
                AffectedProvision(
                    uid=version.provision_uid,
                    instrument_uid="BCT:CIRCULAR:1991:24",
                    label="Article 4",
                    canonical_path="article/4",
                ),
            )

        def resolve_provision_as_of(self, provision_uid, as_of):
            self.temporal_calls.append(("resolve", provision_uid, as_of))
            return TemporalResolution(
                provision_uid=provision_uid,
                as_of=as_of,
                version=version,
                reason="resolved",
            )

        def lineage(self, provision_uid):
            self.temporal_calls.append(("lineage", provision_uid))
            return (
                LineageEntry(
                    event_uid="event:bct:2016:03:article-2",
                    source_instrument_uid="BCT:CIRCULAR:2016:03",
                    action="REPLACE",
                    effective_from=date(2016, 12, 30),
                    introduced_version_uids=(version.uid,),
                    retired_version_uids=("version:bct:1991:24:article-4:old",),
                    evidence=(
                        {
                            "uid": "evidence:cir-2016-03:p2:article-2",
                            "filename": "Cir_2016_03_fr.pdf",
                            "page_number": 2,
                            "quote": "Article 2 replaces Article 4.",
                        },
                    ),
                    predecessor_complete=True,
                ),
            )

    graph = CompleteTemporalGraph()
    result = RelationshipGraphRetriever(
        graph,
        current_date=date(2026, 8, 31),
    ).retrieve(
        "What rule in Article 4 is currently in force?",
        (_document("Cir_2016_03_fr.pdf", page=1),),
    )

    assert result.trace.temporal_status == TemporalRetrievalStatus.RESOLVED
    assert result.trace.as_of == date(2026, 8, 31)
    assert result.requires_temporal_abstention is False
    assert len(result.documents) == 1
    assert version.text in result.documents[0].page_content
    assert result.documents[0].metadata["temporal_resolution"] == "VERIFIED"
    assert result.documents[0].metadata["provision_uid"] == version.provision_uid
    affected_call = graph.temporal_calls[0]
    assert affected_call[0] == "affected"
    assert affected_call[1][0].filename == "Cir_2016_03_fr.pdf"
    assert affected_call[1][0].page_numbers == (2,)


def test_current_rule_query_abstains_when_replacement_predecessor_is_incomplete():
    version = circular_2016_03_fr_bundle().provision_versions[0]

    class IncompleteTemporalGraph(FakeRelationshipGraph):
        def affected_provisions(self, _seeds, *, limit):
            assert limit == 10
            return (
                AffectedProvision(
                    uid=version.provision_uid,
                    instrument_uid="BCT:CIRCULAR:1991:24",
                    label="Article 4",
                    canonical_path="article/4",
                ),
            )

        def resolve_provision_as_of(self, provision_uid, as_of):
            return TemporalResolution(
                provision_uid=provision_uid,
                as_of=as_of,
                version=version,
                reason="resolved",
            )

        def lineage(self, _provision_uid):
            return (
                LineageEntry(
                    event_uid="event:bct:2016:03:article-2",
                    source_instrument_uid="BCT:CIRCULAR:2016:03",
                    action="REPLACE",
                    effective_from=date(2016, 12, 30),
                    introduced_version_uids=(version.uid,),
                    retired_version_uids=(),
                    evidence=(
                        {
                            "uid": "evidence:cir-2016-03:p2:article-2",
                            "filename": "Cir_2016_03_fr.pdf",
                            "page_number": 2,
                            "quote": "Article 2 replaces Article 4.",
                        },
                    ),
                    predecessor_complete=False,
                ),
            )

    result = RelationshipGraphRetriever(
        IncompleteTemporalGraph(),
        current_date=date(2026, 8, 31),
    ).retrieve(
        "What rule in Article 4 is currently in force?",
        (_document("Cir_2016_03_fr.pdf", page=1),),
    )

    assert result.documents == ()
    assert result.trace.temporal_status == TemporalRetrievalStatus.INCOMPLETE
    assert result.trace.temporal_reason == "replacement_predecessor_incomplete"
    assert result.requires_temporal_abstention is True


def test_current_rule_query_does_not_guess_between_multiple_affected_parts():
    class AmbiguousTemporalGraph(FakeRelationshipGraph):
        def __init__(self):
            super().__init__()
            self.resolution_calls = []

        def affected_provisions(self, _seeds, *, limit):
            assert limit == 10
            return (
                AffectedProvision(
                    uid="BCT:CIRCULAR:1991:24:ARTICLE:4",
                    instrument_uid="BCT:CIRCULAR:1991:24",
                    label="Article 4",
                    canonical_path="article/4",
                ),
                AffectedProvision(
                    uid="BCT:CIRCULAR:1993:08:ANNEX:13",
                    instrument_uid="BCT:CIRCULAR:1993:08",
                    label="Annexe 13",
                    canonical_path="annex/13",
                ),
            )

        def resolve_provision_as_of(self, provision_uid, as_of):
            self.resolution_calls.append((provision_uid, as_of))
            raise AssertionError("an ambiguous provision must not be resolved")

        def lineage(self, _provision_uid):
            raise AssertionError("an ambiguous provision must not load lineage")

    graph = AmbiguousTemporalGraph()
    result = RelationshipGraphRetriever(
        graph,
        current_date=date(2026, 8, 31),
    ).retrieve(
        "Which rule is currently in force across these circulars?",
        (_document("Cir_2016_03_fr.pdf", page=2),),
    )

    assert result.documents == ()
    assert result.trace.temporal_status == TemporalRetrievalStatus.AMBIGUOUS
    assert result.trace.temporal_reason == "multiple_affected_provisions"
    assert result.trace.provision_uids == (
        "BCT:CIRCULAR:1991:24:ARTICLE:4",
        "BCT:CIRCULAR:1993:08:ANNEX:13",
    )
    assert graph.resolution_calls == []


@pytest.mark.parametrize(
    ("query", "expected_date"),
    (
        ("Which rule was in force as of 2017-01-01?", date(2017, 1, 1)),
        ("Quelle règle était en vigueur au 31/12/2016 ?", date(2016, 12, 31)),
    ),
)
def test_temporal_rule_query_uses_the_explicit_supported_date(query, expected_date):
    class DateCapturingGraph(FakeRelationshipGraph):
        def __init__(self):
            super().__init__()
            self.as_of = None

        def affected_provisions(self, _seeds, *, limit):
            assert limit == 10
            return (
                AffectedProvision(
                    uid="BCT:CIRCULAR:1991:24:ARTICLE:4",
                    instrument_uid="BCT:CIRCULAR:1991:24",
                    label="Article 4",
                    canonical_path="article/4",
                ),
            )

        def resolve_provision_as_of(self, provision_uid, as_of):
            self.as_of = as_of
            return TemporalResolution(
                provision_uid=provision_uid,
                as_of=as_of,
                version=None,
                reason="no_verified_version",
            )

        def lineage(self, _provision_uid):
            return ()

    graph = DateCapturingGraph()
    result = RelationshipGraphRetriever(
        graph,
        current_date=date(2026, 8, 31),
    ).retrieve(
        query,
        (_document("Cir_2016_03_fr.pdf", page=1),),
    )

    assert graph.as_of == expected_date
    assert result.trace.as_of == expected_date


def test_year_only_historical_rule_query_abstains_without_querying_the_graph():
    class MustNotBeCalledGraph(FakeRelationshipGraph):
        def affected_provisions(self, _seeds, *, limit):
            raise AssertionError(f"graph must not be queried with limit {limit}")

    result = RelationshipGraphRetriever(
        MustNotBeCalledGraph(),
        current_date=date(2026, 8, 31),
    ).retrieve(
        "Which rule was in force in 2017?",
        (_document("Cir_2016_03_fr.pdf", page=1),),
    )

    assert result.documents == ()
    assert result.trace.temporal_status == TemporalRetrievalStatus.DATE_AMBIGUOUS
    assert result.trace.temporal_reason == "exact_historical_date_required"
    assert result.requires_temporal_abstention is True


def test_invalid_explicit_temporal_date_abstains_instead_of_raising():
    result = RelationshipGraphRetriever(
        FakeRelationshipGraph(),
        current_date=date(2026, 8, 31),
    ).retrieve(
        "Which rule was in force as of 2020-13-01?",
        (_document("Cir_2016_03_fr.pdf", page=1),),
    )

    assert result.documents == ()
    assert result.trace.temporal_status == TemporalRetrievalStatus.DATE_AMBIGUOUS
    assert result.trace.temporal_reason == "invalid_explicit_date"


def test_future_temporal_date_abstains_without_querying_the_graph():
    class MustNotBeCalledGraph(FakeRelationshipGraph):
        def affected_provisions(self, _seeds, *, limit):
            raise AssertionError(f"future graph query must not run with {limit}")

    result = RelationshipGraphRetriever(
        MustNotBeCalledGraph(),
        current_date=date(2026, 8, 31),
    ).retrieve(
        "Which rule was in force as of 2027-01-01?",
        (_document("Cir_2016_03_fr.pdf", page=1),),
    )

    assert result.documents == ()
    assert result.trace.temporal_status == TemporalRetrievalStatus.DATE_AMBIGUOUS
    assert result.trace.temporal_reason == "future_as_of_not_supported"


def test_chat_reranks_graph_evidence_with_the_existing_rag_candidates(monkeypatch):
    ordinary = _document("CB_2017_08_FR.pdf", text="ordinary evidence")
    graph_document = _document(
        "CB_2017_08_FR.pdf",
        page=27,
        text="verified graph relationship evidence",
    )
    graph_document.metadata.update(
        page_label=28,
        retrieval_source="neo4j_relationship",
        graph_path="BCT:2017:08 -[CITES]-> BCT:2013:15",
    )

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, seed_documents):
            self.calls.append((query, tuple(seed_documents)))
            return GraphRetrievalResult(
                documents=(graph_document,),
                trace=GraphRetrievalTrace(
                    status=GraphRetrievalStatus.EXPANDED,
                    seed_filenames=("CB_2017_08_FR.pdf",),
                    evidence_count=1,
                    paths=(graph_document.metadata["graph_path"],),
                ),
            )

    fake_retriever = FakeRetriever()
    answer_documents = []
    score_calls = []
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "NEW_TOPIC",
            "rewrite_query": "relation entre circulaires",
            "new_topic": "circulaires",
            "current_topic": "circulaires",
        },
    )
    monkeypatch.setattr(
        conversation,
        "retrieve_relevant_chunks",
        lambda *_: [ordinary],
    )
    monkeypatch.setattr(conversation, "retrieve_bm25", lambda *_: [])

    def fake_score(_reranker, _query, documents):
        score_calls.append(tuple(documents))
        return [
            (document, 10 if document is graph_document else 1)
            for document in documents
        ]

    monkeypatch.setattr(conversation, "score_documents", fake_score)

    def fake_answer(_llm, _message, documents, _memory):
        answer_documents.extend(documents)
        return "answer"

    monkeypatch.setattr(conversation, "generate_answer", fake_answer)

    result = conversation.chat(
        "Quelle relation entre ces circulaires ?",
        {"topics": [], "first_topic": None, "current_topic": None},
        vector_store=object(),
        reranker=object(),
        bm25=object(),
        bm25_documents=[],
        graph_retriever=fake_retriever,
    )

    assert len(score_calls) == 2
    assert fake_retriever.calls[0][0] == "Quelle relation entre ces circulaires ?"
    assert fake_retriever.calls[0][1] == (ordinary,)
    assert answer_documents[0] is graph_document
    assert result["sources"][0]["page"] == 28
    assert result["graph_trace"]["status"] == "EXPANDED"
    assert result["graph_trace"]["evidence_count"] == 1


def test_chat_keeps_baseline_candidates_when_graph_is_unavailable(monkeypatch):
    ordinary = _document("CB_2017_08_FR.pdf")

    class UnavailableRetriever:
        def retrieve(self, _query, _seed_documents):
            return GraphRetrievalResult(
                documents=(),
                trace=GraphRetrievalTrace(
                    status=GraphRetrievalStatus.UNAVAILABLE,
                    error_type="ServiceUnavailable",
                ),
            )

    score_calls = []
    answer_documents = []
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "NEW_TOPIC",
            "rewrite_query": "circulaires 2017-08 2013-15",
            "new_topic": "circulaires",
            "current_topic": "circulaires",
        },
    )
    monkeypatch.setattr(
        conversation,
        "retrieve_relevant_chunks",
        lambda *_: [ordinary],
    )
    monkeypatch.setattr(conversation, "retrieve_bm25", lambda *_: [])

    def fake_score(_reranker, _query, documents):
        score_calls.append(tuple(documents))
        return [(document, 1) for document in documents]

    monkeypatch.setattr(conversation, "score_documents", fake_score)
    monkeypatch.setattr(
        conversation,
        "generate_answer",
        lambda _llm, _message, documents, _memory: (
            answer_documents.extend(documents) or "baseline answer"
        ),
    )

    result = conversation.chat(
        "Quelle relation entre ces circulaires ?",
        {"topics": [], "first_topic": None, "current_topic": None},
        vector_store=object(),
        reranker=object(),
        bm25=object(),
        bm25_documents=[],
        graph_retriever=UnavailableRetriever(),
    )

    assert score_calls == [(ordinary,)]
    assert answer_documents == [ordinary]
    assert result["answer"] == "baseline answer"
    assert result["graph_trace"]["status"] == "UNAVAILABLE"


def test_chat_abstains_from_current_rule_answer_when_graph_runtime_is_absent(monkeypatch):
    ordinary = _document("Cir_2016_03_fr.pdf", text="a possibly outdated rule")
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "NEW_TOPIC",
            "rewrite_query": "current rule Article 4",
            "new_topic": "Article 4",
            "current_topic": "Article 4",
        },
    )
    monkeypatch.setattr(
        conversation,
        "retrieve_relevant_chunks",
        lambda *_: [ordinary],
    )
    monkeypatch.setattr(conversation, "retrieve_bm25", lambda *_: [])
    monkeypatch.setattr(
        conversation,
        "score_documents",
        lambda _reranker, _query, documents: [
            (document, 1) for document in documents
        ],
    )
    monkeypatch.setattr(
        conversation,
        "generate_answer",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("unsafe temporal answer generation must not run")
        ),
    )

    result = conversation.chat(
        "What rule in Article 4 is currently in force?",
        {"topics": [], "first_topic": None, "current_topic": None},
        vector_store=object(),
        reranker=object(),
        bm25=object(),
        bm25_documents=[],
        graph_retriever=None,
    )

    assert "cannot determine" in result["answer"].casefold()
    assert result["sources"] == []
    assert result["graph_trace"]["status"] == "UNAVAILABLE"
    assert result["graph_trace"]["temporal_status"] == "UNAVAILABLE"
    assert result["graph_trace"]["temporal_reason"] == "temporal_graph_unavailable"


def test_chat_abstains_when_the_selected_provision_lineage_is_incomplete(monkeypatch):
    ordinary = _document("Cir_2016_03_fr.pdf", text="a possibly outdated rule")

    class IncompleteRetriever:
        def retrieve(self, _query, _seed_documents):
            return GraphRetrievalResult(
                documents=(),
                trace=GraphRetrievalTrace(
                    status=GraphRetrievalStatus.NO_EVIDENCE,
                    temporal_status=TemporalRetrievalStatus.INCOMPLETE,
                    as_of=date(2026, 8, 31),
                    provision_uids=("BCT:CIRCULAR:1991:24:ARTICLE:4",),
                    temporal_reason="replacement_predecessor_incomplete",
                ),
            )

    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "NEW_TOPIC",
            "rewrite_query": "current rule Article 4",
            "new_topic": "Article 4",
            "current_topic": "Article 4",
        },
    )
    monkeypatch.setattr(
        conversation,
        "retrieve_relevant_chunks",
        lambda *_: [ordinary],
    )
    monkeypatch.setattr(conversation, "retrieve_bm25", lambda *_: [])
    monkeypatch.setattr(
        conversation,
        "score_documents",
        lambda _reranker, _query, documents: [
            (document, 1) for document in documents
        ],
    )
    monkeypatch.setattr(
        conversation,
        "generate_answer",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("unsafe temporal answer generation must not run")
        ),
    )

    result = conversation.chat(
        "What rule in Article 4 is currently in force?",
        {"topics": [], "first_topic": None, "current_topic": None},
        vector_store=object(),
        reranker=object(),
        bm25=object(),
        bm25_documents=[],
        graph_retriever=IncompleteRetriever(),
    )

    assert "cannot determine" in result["answer"].casefold()
    assert result["sources"] == []
    assert result["graph_trace"]["temporal_status"] == "INCOMPLETE"
    assert result["graph_trace"]["temporal_reason"] == (
        "replacement_predecessor_incomplete"
    )


def test_chat_keeps_verified_temporal_context_mandatory_after_reranking(monkeypatch):
    ordinary_documents = [
        _document(f"ordinary-{index}.pdf", text=f"ordinary {index}")
        for index in range(6)
    ]
    temporal_document = _document(
        "Cir_2016_03_fr.pdf",
        page=1,
        text="verified controlling Article 4 text",
    )
    temporal_document.metadata.update(
        page_label=2,
        retrieval_source="neo4j_temporal_provision",
        temporal_resolution="VERIFIED",
        provision_uid="BCT:CIRCULAR:1991:24:ARTICLE:4",
    )

    class ResolvedRetriever:
        def retrieve(self, _query, _seed_documents):
            return GraphRetrievalResult(
                documents=(temporal_document,),
                trace=GraphRetrievalTrace(
                    status=GraphRetrievalStatus.EXPANDED,
                    temporal_status=TemporalRetrievalStatus.RESOLVED,
                    as_of=date(2026, 8, 31),
                    provision_uids=("BCT:CIRCULAR:1991:24:ARTICLE:4",),
                ),
            )

    answer_documents = []
    monkeypatch.setattr(conversation, "create_llm", lambda: object())
    monkeypatch.setattr(
        conversation,
        "route_message",
        lambda *_: {
            "intent": "NEW_TOPIC",
            "rewrite_query": "current rule Article 4",
            "new_topic": "Article 4",
            "current_topic": "Article 4",
        },
    )
    monkeypatch.setattr(
        conversation,
        "retrieve_relevant_chunks",
        lambda *_: ordinary_documents,
    )
    monkeypatch.setattr(conversation, "retrieve_bm25", lambda *_: [])
    monkeypatch.setattr(
        conversation,
        "score_documents",
        lambda _reranker, _query, documents: [
            (
                document,
                -1 if document is temporal_document else 10,
            )
            for document in documents
        ],
    )
    monkeypatch.setattr(
        conversation,
        "generate_answer",
        lambda _llm, _message, documents, _memory: (
            answer_documents.extend(documents) or "resolved answer"
        ),
    )

    result = conversation.chat(
        "What rule in Article 4 is currently in force?",
        {"topics": [], "first_topic": None, "current_topic": None},
        vector_store=object(),
        reranker=object(),
        bm25=object(),
        bm25_documents=[],
        graph_retriever=ResolvedRetriever(),
    )

    assert answer_documents[0] is temporal_document
    assert len(answer_documents) == 5
    assert result["graph_trace"]["temporal_status"] == "RESOLVED"
    assert result["sources"][0]["file"] == "Cir_2016_03_fr.pdf"
