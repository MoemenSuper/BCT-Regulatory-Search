from langchain_core.documents import Document

import conversation
from regulatory_graph.neo4j_store import RelationshipEvidence
from regulatory_graph.runtime import (
    GraphRetrievalResult,
    GraphRetrievalStatus,
    GraphRetrievalTrace,
    RelationshipGraphRetriever,
    is_relationship_query,
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
