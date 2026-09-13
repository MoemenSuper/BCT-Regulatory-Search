from langchain_core.documents import Document

from graph_contract import GraphRetrievalStatus, TemporalRetrievalStatus
from regulatory_graph_lite.runtime import RelationshipGraphLiteRetriever


class FakeStore:
    def verified_edges_for_instruments(self, ids, *, limit, relationship_types=None):
        if "cir:2017:8" not in ids:
            return []
        assert limit == 8
        return [
            {
                "relation": "REPLACES",
                "source_id": "cir:2025:17",
                "target_id": "cir:2017:8",
                "source_provision": None,
                "target_provision": None,
                "candidate_id": "abc",
                "evidence_file": "Cir_2025_17_fr.pdf",
                "evidence_page": 3,
                "evidence_quote": "La présente circulaire remplace la circulaire n° 2017-08.",
                "proposed_effective_date": None,
            }
        ]


def seed():
    return Document(
        page_content="old content",
        metadata={"source": "Cir_2017_08_fr.pdf", "page_label": 1},
    )


def test_relationship_question_gets_verified_graph_evidence():
    result = RelationshipGraphLiteRetriever(FakeStore()).retrieve(
        "Quelle circulaire remplace la circulaire 2017-08 ?",
        [seed()],
    )
    assert result.trace.status is GraphRetrievalStatus.EXPANDED
    assert result.trace.temporal_status is TemporalRetrievalStatus.NOT_REQUESTED
    assert len(result.documents) == 1
    assert result.documents[0].metadata["temporal_verification"] == "VERIFIED_RELATIONSHIP_ONLY"
    assert result.documents[0].metadata["page"] == 3
    assert result.documents[0].metadata["page_label"] == 3
    assert "temporal_proposed_effective_date" not in result.documents[0].metadata


def test_currentness_question_keeps_relationship_evidence_but_marks_lineage_incomplete():
    result = RelationshipGraphLiteRetriever(FakeStore()).retrieve(
        "Quelle disposition de la circulaire 2017-08 est actuellement en vigueur ?",
        [seed()],
    )
    assert result.trace.status is GraphRetrievalStatus.EXPANDED
    assert result.trace.temporal_status is TemporalRetrievalStatus.INCOMPLETE
    assert result.requires_temporal_abstention is True
    assert len(result.documents) == 1


def test_explicit_query_identity_can_seed_graph_when_rag_seed_is_missing():
    result = RelationshipGraphLiteRetriever(FakeStore()).retrieve(
        "Quelle circulaire remplace la circulaire 2017-08 ?",
        [],
    )
    assert result.trace.status is GraphRetrievalStatus.EXPANDED
    assert len(result.documents) == 1


class RecordingStore(FakeStore):
    def __init__(self):
        self.calls = []

    def verified_edges_for_instruments(self, ids, *, limit, relationship_types=None):
        self.calls.append((tuple(ids), tuple(relationship_types or ())))
        return super().verified_edges_for_instruments(
            ids,
            limit=limit,
            relationship_types=relationship_types,
        )


def test_temporal_query_excludes_citation_only_edges_from_graph_lookup():
    store = RecordingStore()
    RelationshipGraphLiteRetriever(store).retrieve(
        "Quelle disposition de la circulaire 2017-08 est actuellement en vigueur ?",
        [seed()],
    )
    assert store.calls
    assert "CITES" not in store.calls[0][1]
    assert set(store.calls[0][1]) == {"AMENDS", "REPLACES", "ABROGATES"}


def test_business_travel_plafond_currentness_uses_amendment_edges_not_cites():
    store = RecordingStore()
    result = RelationshipGraphLiteRetriever(store).retrieve(
        "Quel est aujourd'hui le plafond en vigueur pour une allocation de voyage d'affaires ?",
        [seed()],
    )
    assert result.trace.temporal_status is TemporalRetrievalStatus.INCOMPLETE
    assert store.calls
    assert "CITES" not in store.calls[0][1]
    assert set(store.calls[0][1]) == {"AMENDS", "REPLACES", "ABROGATES"}


class ChainStore:
    def verified_edges_for_instruments(self, ids, *, limit, relationship_types=None):
        ids = set(ids)
        if "cir:2017:8" in ids:
            return [{
                "relation": "REPLACES",
                "source_id": "cir:2020:3",
                "target_id": "cir:2017:8",
                "source_provision": None,
                "target_provision": None,
                "candidate_id": "edge-1",
                "evidence_file": "Cir_2020_03_fr.pdf",
                "evidence_page": 2,
                "evidence_quote": "La présente circulaire remplace la circulaire n° 2017-08.",
                "proposed_effective_date": None,
            }]
        if "cir:2020:3" in ids:
            return [{
                "relation": "REPLACES",
                "source_id": "cir:2025:17",
                "target_id": "cir:2020:3",
                "source_provision": None,
                "target_provision": None,
                "candidate_id": "edge-2",
                "evidence_file": "Cir_2025_17_fr.pdf",
                "evidence_page": 3,
                "evidence_quote": "La présente circulaire remplace la circulaire n° 2020-03.",
                "proposed_effective_date": None,
            }]
        return []


def test_graph_lite_can_follow_one_additional_verified_hop():
    result = RelationshipGraphLiteRetriever(ChainStore()).retrieve(
        "Quelle circulaire remplace la circulaire 2017-08 ?",
        [seed()],
    )
    assert result.trace.status is GraphRetrievalStatus.EXPANDED
    assert len(result.documents) == 2
    assert any("2020-03" in document.page_content for document in result.documents)
