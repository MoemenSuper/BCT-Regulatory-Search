from datetime import date
import os
from urllib.parse import urlsplit

from langchain_core.documents import Document
import pytest

from regulatory_graph.fixtures import circular_2016_03_fr_bundle
from regulatory_graph.models import (
    GraphChunk,
    InstrumentReference,
    ProvisionVersion,
    VerificationStatus,
    VersionStatus,
)
from regulatory_graph.neo4j_store import (
    Neo4jGraphWriter,
    Neo4jRegulatoryGraph,
    Neo4jStructuralWriter,
    SourcePageSeed,
)
from regulatory_graph.runtime import RelationshipGraphRetriever, TemporalRetrievalStatus
from regulatory_graph.source_verification import verify_bundle_source


def _disposable_live_uri():
    uri = os.environ.get("BCT_NEO4J_TEST_URI")
    confirmed = os.environ.get("BCT_NEO4J_TEST_DISPOSABLE") == "YES"
    if not uri or not confirmed:
        return None
    parsed = urlsplit(uri)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("disposable Neo4j test URI must use a loopback host")
    if parsed.port in {None, 7687}:
        raise RuntimeError("disposable Neo4j test URI must use an explicit non-default port")
    if parsed.port == 17687:
        raise RuntimeError("disposable Neo4j test URI cannot use the persistent graph port")
    return uri


DISPOSABLE_LIVE_URI = _disposable_live_uri()


class FakeDriver:
    def __init__(self, responses=()):
        self.calls = []
        self.responses = list(responses)

    def execute_query(self, statement, **parameters):
        self.calls.append((statement, parameters))
        if self.responses:
            return self.responses.pop(0)
        return []


def _chunk(bundle, *, uid="chunk:cir-2016-03:fr:2:0", chunk_index=0):
    return GraphChunk(
        uid=uid,
        page_uid=bundle.pages[0].uid,
        chunk_index=chunk_index,
        text=f"Chunk {chunk_index}",
        content_hash="a" * 64,
        source_sha256=bundle.source_editions[0].sha256,
        extraction_artifact_hash="b" * 64,
        extraction_method="native",
    )


def test_live_cleanup_guard_requires_explicit_local_disposable_target(monkeypatch):
    monkeypatch.setenv("BCT_NEO4J_TEST_URI", "bolt://localhost:17687")
    monkeypatch.delenv("BCT_NEO4J_TEST_DISPOSABLE", raising=False)
    assert _disposable_live_uri() is None

    monkeypatch.setenv("BCT_NEO4J_TEST_DISPOSABLE", "YES")
    monkeypatch.setenv("BCT_NEO4J_TEST_URI", "bolt://database.example:17687")
    with pytest.raises(RuntimeError, match="loopback"):
        _disposable_live_uri()

    monkeypatch.setenv("BCT_NEO4J_TEST_URI", "bolt://localhost:7687")
    with pytest.raises(RuntimeError, match="non-default"):
        _disposable_live_uri()

    monkeypatch.setenv("BCT_NEO4J_TEST_URI", "bolt://localhost:17687")
    with pytest.raises(RuntimeError, match="persistent"):
        _disposable_live_uri()


def test_real_fixture_is_bound_to_the_frozen_pdf_and_three_changes():
    bundle = circular_2016_03_fr_bundle()

    assert bundle.source_editions[0].filename == "Cir_2016_03_fr.pdf"
    assert bundle.source_editions[0].sha256 == (
        "E463736EBB98BE5DBC6E02635E1D635734B9CDF4ACFD598F4F1BA8351DC43078"
    )
    assert tuple(page.page_number for page in bundle.pages) == (2, 3, 4)
    assert len(bundle.change_events) == 3
    assert all(
        event.verification_status == VerificationStatus.VERIFIED
        for event in bundle.change_events
    )
    assert {
        event.effective_from for event in bundle.change_events
    } == {date(2016, 8, 8), date(2016, 12, 30)}
    assert all(event.introduces_version_uids for event in bundle.change_events)
    assert all(not event.retires_version_uids for event in bundle.change_events)
    assert bundle.provision_versions[0].uid == (
        "version:bct:1991:24:article-4:2016-12-30"
    )


@pytest.mark.skipif(
    not os.environ.get("BCT_GRAPH_TRACER_SOURCE_PDF"),
    reason="set BCT_GRAPH_TRACER_SOURCE_PDF to verify the frozen source PDF",
)
def test_real_fixture_quotes_and_versions_are_exact_source_text():
    receipt = verify_bundle_source(
        circular_2016_03_fr_bundle(),
        os.environ["BCT_GRAPH_TRACER_SOURCE_PDF"],
    )

    assert receipt.source_sha256_verified is True
    assert receipt.page_count_verified is True
    assert receipt.exact_evidence_count == 4
    assert receipt.exact_verified_version_count == 2


def test_writer_uses_parameterized_merge_and_only_allowlisted_relationships():
    driver = FakeDriver()
    writer = Neo4jGraphWriter(driver, database="neo4j")

    receipt = writer.write_bundle(circular_2016_03_fr_bundle())

    statements = tuple(call[0] for call in driver.calls)
    assert receipt.change_event_count == 3
    assert receipt.bundle_sha256
    assert any("MERGE (node:Instrument {uid: row.uid})" in query for query in statements)
    assert any("MERGE (source)-[:DECLARES_CHANGE]->(target)" in query for query in statements)
    assert any("MERGE (source)-[:INTRODUCES_VERSION]->(target)" in query for query in statements)
    assert all("$rows" in query or "IF NOT EXISTS" in query for query in statements)
    assert all(
        call[1].get("database_") == "neo4j"
        for call in driver.calls
    )


def test_writer_persists_only_verified_reference_lineage():
    bundle = circular_2016_03_fr_bundle()
    reference = InstrumentReference(
        uid="reference:cir-2016-03:p2:cir-1991-24",
        source_instrument_uid="BCT:CIRCULAR:2016:03",
        target_instrument_uid="BCT:CIRCULAR:1991:24",
        evidence_uid=bundle.evidence_spans[0].uid,
        raw_citation="circulaire n°91-24",
        extraction_method="native",
        resolver_rule="french_bct_instrument_reference_v1",
        verification_status=VerificationStatus.VERIFIED,
        verification_method="manual_rendered_pdf_v1",
        rendered_image_sha256="d" * 64,
        verified_by="manual:test-reviewer",
    )
    driver = FakeDriver()

    Neo4jGraphWriter(driver).write_bundle(
        bundle.model_copy(update={"instrument_references": (reference,)})
    )

    statements = tuple(call[0] for call in driver.calls)
    assert any(
        "MERGE (node:InstrumentReference {uid: row.uid})" in query
        for query in statements
    )
    assert any(
        "MERGE (source)-[:DECLARES_REFERENCE]->(target)" in query
        for query in statements
    )
    assert any(
        "MERGE (source)-[:TARGETS]->(target)" in query
        for query in statements
    )
    assert any(
        "MERGE (source)-[:EVIDENCED_BY]->(target)" in query
        for query in statements
    )

    candidate = reference.model_copy(
        update={"verification_status": VerificationStatus.NEEDS_REVIEW}
    )
    rejected_driver = FakeDriver()
    with pytest.raises(ValueError, match="VERIFIED"):
        Neo4jGraphWriter(rejected_driver).write_bundle(
            bundle.model_copy(update={"instrument_references": (candidate,)})
        )
    assert rejected_driver.calls == []


def test_writer_rejects_relationships_outside_the_allowlist_before_querying():
    driver = FakeDriver()
    writer = Neo4jGraphWriter(driver)

    with pytest.raises(ValueError, match="not allowlisted"):
        writer._relationship(
            "Instrument",
            "RELATED_TO",
            "Instrument",
            (("instrument:1", "instrument:2"),),
        )

    assert driver.calls == []


def test_writer_does_not_erase_unspecified_optional_properties():
    driver = FakeDriver()

    Neo4jGraphWriter(driver).write_bundle(circular_2016_03_fr_bundle())

    edition_call = next(
        call for call in driver.calls if "MERGE (node:SourceEdition" in call[0]
    )
    assert all(value is not None for value in edition_call[1]["rows"][0].values())


def test_structural_writer_skips_complete_hash_identical_edition():
    bundle = circular_2016_03_fr_bundle().model_copy(
        update={
            "provisions": (),
            "provision_versions": (),
            "evidence_spans": (),
            "change_events": (),
        }
    )
    edition = bundle.source_editions[0]
    driver = FakeDriver(
        responses=[
            [
                {
                    "uid": edition.uid,
                    "logical_edition_uid": edition.uid,
                    "relative_path": edition.filename,
                    "sha256": edition.sha256,
                    "extraction_artifact_hash": None,
                    "page_uids": [page.uid for page in bundle.pages],
                    "chunk_uids": [],
                    "pages": [
                        {
                            "uid": page.uid,
                            "source_sha256": page.source_sha256,
                            "extraction_artifact_hash": page.extraction_artifact_hash,
                            "text_hash": page.text_hash,
                        }
                        for page in bundle.pages
                    ],
                    "chunks": [],
                    "lifecycle_status": edition.lifecycle_status,
                    "identity_verification_status": None,
                    "identity_evidence": None,
                    "identity_rule": None,
                    "identity_evidence_text": None,
                }
            ]
        ]
    )

    receipt = Neo4jStructuralWriter(driver).sync_bundle(bundle)

    assert receipt.skipped_edition_count == 1
    assert receipt.written_edition_count == 0
    assert receipt.candidate_edition_count == 0
    assert len(driver.calls) == 1
    assert "logical_edition_uid" in driver.calls[0][0]


def test_writer_persists_page_chunks_and_deterministic_chunk_order():
    bundle = circular_2016_03_fr_bundle()
    chunks = (
        _chunk(bundle),
        _chunk(bundle, uid="chunk:cir-2016-03:fr:2:1", chunk_index=1),
    )
    driver = FakeDriver()

    Neo4jGraphWriter(driver).write_bundle(
        bundle.model_copy(update={"chunks": chunks})
    )

    statements = tuple(call[0] for call in driver.calls)
    assert any("MERGE (node:Chunk {uid: row.uid})" in query for query in statements)
    assert any("MERGE (source)-[:HAS_CHUNK]->(target)" in query for query in statements)
    assert any("MERGE (source)-[:NEXT_CHUNK]->(target)" in query for query in statements)


def test_writer_persists_declared_provision_version_and_evidence_relationships():
    bundle = circular_2016_03_fr_bundle()
    parent = bundle.provisions[0]
    child = parent.model_copy(
        update={
            "uid": f"{parent.uid}:PARAGRAPH:1",
            "label": "Paragraph 1",
            "canonical_path": "article/4/paragraph/1",
            "parent_provision_uid": parent.uid,
        }
    )
    current = bundle.provision_versions[0].model_copy(
        update={
            "version_number": 2,
            "status": VersionStatus.ACTIVE,
            "verification_status": VerificationStatus.VERIFIED,
            "supersedes_version_uid": "version:bct:1991:24:article-4:prior",
        }
    )
    prior = current.model_copy(
        update={
            "uid": "version:bct:1991:24:article-4:prior",
            "version_number": 1,
            "valid_from": date(2010, 1, 1),
            "valid_to": current.valid_from,
            "status": VersionStatus.SUPERSEDED,
            "supersedes_version_uid": None,
        }
    )
    chunk = _chunk(bundle)
    evidence = bundle.evidence_spans[0].model_copy(
        update={"chunk_uid": chunk.uid}
    )
    driver = FakeDriver()

    Neo4jGraphWriter(driver).write_bundle(
        bundle.model_copy(
            update={
                "chunks": (chunk,),
                "provisions": (*bundle.provisions, child),
                "provision_versions": (
                    prior,
                    current,
                    *bundle.provision_versions[1:],
                ),
                "evidence_spans": (evidence, *bundle.evidence_spans[1:]),
            }
        )
    )

    statements = tuple(call[0] for call in driver.calls)
    assert any("MERGE (source)-[:CONTAINS_PROVISION]->(target)" in query for query in statements)
    assert any("MERGE (source)-[:CURRENT_VERSION]->(target)" in query for query in statements)
    assert any("DELETE relationship" in query for query in statements)
    assert any("MERGE (source)-[:SUPERSEDES_VERSION]->(target)" in query for query in statements)
    assert any("MERGE (source)-[:IN_CHUNK]->(target)" in query for query in statements)


def test_writer_validates_all_change_references_before_writing(monkeypatch):
    bundle = circular_2016_03_fr_bundle()
    bad_event = bundle.change_events[0].model_copy(
        update={"introduces_version_uids": ("missing-version",)}
    )
    invalid = bundle.model_copy(
        update={"change_events": (bad_event, *bundle.change_events[1:])}
    )
    driver = FakeDriver()

    with pytest.raises(ValueError, match="missing-version"):
        Neo4jGraphWriter(driver).write_bundle(invalid)

    assert driver.calls == []


def test_writer_revalidates_structural_references_before_writing():
    bundle = circular_2016_03_fr_bundle()
    invalid_provision = bundle.provisions[0].model_copy(
        update={"parent_provision_uid": "missing-parent"}
    )
    invalid = bundle.model_copy(
        update={
            "provisions": (invalid_provision, *bundle.provisions[1:]),
        }
    )
    driver = FakeDriver()

    with pytest.raises(ValueError, match="missing-parent"):
        Neo4jGraphWriter(driver).write_bundle(invalid)

    assert driver.calls == []


def test_writer_rejects_overlapping_verified_version_intervals():
    bundle = circular_2016_03_fr_bundle()
    version = bundle.provision_versions[0]
    overlapping = version.model_copy(
        update={
            "uid": f"{version.uid}:overlap",
            "version_number": 2,
            "valid_from": date(2017, 1, 1),
        }
    )
    invalid = bundle.model_copy(
        update={
            "provision_versions": (version, overlapping, *bundle.provision_versions[1:]),
        }
    )
    driver = FakeDriver()

    with pytest.raises(ValueError, match="overlapping verified versions"):
        Neo4jGraphWriter(driver).write_bundle(invalid)

    assert driver.calls == []


def test_writer_rejects_duplicate_chunk_positions():
    bundle = circular_2016_03_fr_bundle()
    duplicate_position = (
        _chunk(bundle),
        _chunk(bundle, uid="chunk:cir-2016-03:fr:2:duplicate"),
    )
    driver = FakeDriver()

    with pytest.raises(ValueError, match="duplicate chunk position"):
        Neo4jGraphWriter(driver).write_bundle(
            bundle.model_copy(update={"chunks": duplicate_position})
        )

    assert driver.calls == []


def test_writer_rejects_cross_provision_supersession():
    bundle = circular_2016_03_fr_bundle()
    version = bundle.provision_versions[0].model_copy(
        update={"supersedes_version_uid": bundle.provision_versions[1].uid}
    )
    driver = FakeDriver()

    with pytest.raises(ValueError, match="same provision"):
        Neo4jGraphWriter(driver).write_bundle(
            bundle.model_copy(
                update={
                    "provision_versions": (version, *bundle.provision_versions[1:]),
                }
            )
        )

    assert driver.calls == []


def test_as_of_query_returns_one_verified_half_open_version():
    version = circular_2016_03_fr_bundle().provision_versions[0]
    driver = FakeDriver(
        responses=[
            [{"version": version.model_dump(mode="python")}],
            [],
        ]
    )
    graph = Neo4jRegulatoryGraph(driver)

    resolved = graph.resolve_provision_as_of(
        version.provision_uid,
        date(2016, 12, 30),
    )
    missing = graph.resolve_provision_as_of(
        version.provision_uid,
        date(2016, 12, 29),
    )

    assert resolved.version == version
    assert resolved.reason == "resolved"
    assert missing.version is None
    assert missing.reason == "no_verified_version"
    query, parameters = driver.calls[0]
    assert "version.verification_status = 'VERIFIED'" in query
    assert "version.valid_from <= $as_of" in query
    assert "version.valid_to IS NULL OR $as_of < version.valid_to" in query
    assert parameters["as_of"] == date(2016, 12, 30)


def test_affected_provisions_use_verified_source_pages_and_incoming_targets():
    driver = FakeDriver(
        responses=[
            [
                {
                    "uid": "BCT:CIRCULAR:1991:24:ARTICLE:4",
                    "instrument_uid": "BCT:CIRCULAR:1991:24",
                    "label": "Article 4",
                    "canonical_path": "article/4",
                }
            ]
        ]
    )

    provisions = Neo4jRegulatoryGraph(driver).affected_provisions(
        (
            SourcePageSeed(
                filename="Cir_2016_03_fr.pdf",
                page_numbers=(2, 3),
            ),
        ),
        limit=4,
    )

    assert len(provisions) == 1
    assert provisions[0].uid == "BCT:CIRCULAR:1991:24:ARTICLE:4"
    assert provisions[0].label == "Article 4"
    query, parameters = driver.calls[0]
    assert "event.verification_status = 'VERIFIED'" in query
    assert "page.page_number IN seed.page_numbers" in query
    assert "(seed_instrument)-[:HAS_PROVISION]->(provision:Provision)" in query
    assert "TargetSpan" in query
    assert "WITHIN" in query
    assert parameters["seeds"] == [
        {"filename": "Cir_2016_03_fr.pdf", "page_numbers": [2, 3]}
    ]
    assert parameters["limit"] == 4


def test_as_of_query_rejects_overlapping_verified_versions():
    version = circular_2016_03_fr_bundle().provision_versions[0]
    driver = FakeDriver(
        responses=[
            [
                {"version": version.model_dump(mode="python")},
                {
                    "version": version.model_copy(
                        update={"uid": "overlapping-version", "version_number": 2}
                    ).model_dump(mode="python")
                },
            ]
        ]
    )

    with pytest.raises(ValueError, match="multiple verified versions"):
        Neo4jRegulatoryGraph(driver).resolve_provision_as_of(
            version.provision_uid,
            date(2016, 12, 30),
        )


def test_lineage_discloses_missing_predecessor_and_exact_evidence():
    driver = FakeDriver(
        responses=[
            [
                {
                    "event_uid": "event:bct:2016:03:article-2",
                    "source_instrument_uid": "BCT:CIRCULAR:2016:03",
                    "action": "REPLACE",
                    "effective_from": date(2016, 12, 30),
                    "introduced_version_uids": [
                        "version:bct:1991:24:article-4:2016-12-30"
                    ],
                    "retired_version_uids": [],
                    "evidence": [
                        {
                            "uid": "evidence:cir-2016-03:p2:article-2",
                            "filename": "Cir_2016_03_fr.pdf",
                            "page_number": 2,
                            "quote": "Article 2 : Les dispositions de l’article 4 ...",
                        },
                        {
                            "uid": "evidence:cir-2016-03:p4:effective-date",
                            "filename": "Cir_2016_03_fr.pdf",
                            "page_number": 4,
                            "quote": "Article 7 : ... 30 décembre 2016.",
                        },
                    ],
                }
            ]
        ]
    )

    entries = Neo4jRegulatoryGraph(driver).lineage(
        "BCT:CIRCULAR:1991:24:ARTICLE:4"
    )

    assert len(entries) == 1
    assert entries[0].predecessor_complete is False
    assert entries[0].evidence_pages == (2, 4)
    assert entries[0].source_filenames == ("Cir_2016_03_fr.pdf",)
    assert entries[0].evidence[0].page_number == 2
    assert entries[0].evidence[0].quote.startswith("Article 2")
    query, _parameters = driver.calls[0]
    assert "event.verification_status = 'VERIFIED'" in query
    assert "TargetSpan" in query
    assert "WITHIN" in query


def test_relationship_evidence_is_verified_bounded_and_source_linked():
    driver = FakeDriver(
        responses=[
            [
                {
                    "seed_filename": "CB_2017_08_FR.pdf",
                    "seed_instrument_uid": "BCT:CIRCULAR:2017:08",
                    "related_instrument_uid": "BCT:CIRCULAR:2013:15",
                    "relation_kind": "CITES",
                    "direction": "OUTGOING",
                    "fact_uid": "reference:2017-08:2013-15",
                    "evidence_uid": "evidence:2017-08:p28:2013-15",
                    "filename": "CB_2017_08_FR.pdf",
                    "page_number": 28,
                    "quote": "La presente circulaire cite la circulaire n 2013-15.",
                }
            ]
        ]
    )

    evidence = Neo4jRegulatoryGraph(driver).relationship_evidence(
        ("CB_2017_08_FR.pdf", "CB_2017_08_FR.pdf"),
        limit=4,
    )

    assert len(evidence) == 1
    assert evidence[0].filename == "CB_2017_08_FR.pdf"
    assert evidence[0].page_number == 28
    assert evidence[0].relation_kind == "CITES"
    assert evidence[0].path == (
        "BCT:CIRCULAR:2017:08 -[CITES]-> BCT:CIRCULAR:2013:15"
    )
    query, parameters = driver.calls[0]
    assert "fact.verification_status = 'VERIFIED'" in query
    assert "type(declares) IN ['DECLARES_REFERENCE', 'DECLARES_CHANGE']" in query
    assert "EVIDENCED_BY" in query
    assert "ON_PAGE" in query
    assert parameters["seed_filenames"] == ["CB_2017_08_FR.pdf"]
    assert parameters["limit"] == 4

    invalid = evidence[0].model_dump(mode="python")
    invalid["relation_kind"] = "RELATED_TO"
    with pytest.raises(ValueError):
        type(evidence[0]).model_validate(invalid)


def test_graph_snapshot_hash_is_stable_across_database_row_order():
    nodes = [
        {"labels": ["Instrument"], "properties": {"uid": "instrument:2"}},
        {"labels": ["Instrument"], "properties": {"uid": "instrument:1"}},
    ]
    relationships = [
        {
            "source_uid": "instrument:1",
            "relationship_type": "TARGETS",
            "target_uid": "instrument:2",
            "properties": {},
        }
    ]
    first = Neo4jRegulatoryGraph(
        FakeDriver(responses=[nodes, relationships])
    ).snapshot()
    second = Neo4jRegulatoryGraph(
        FakeDriver(responses=[list(reversed(nodes)), relationships])
    ).snapshot()

    assert first == second
    assert first.nodes == 2
    assert first.relationships == 1
    assert len(first.content_sha256) == 64


@pytest.mark.skipif(
    DISPOSABLE_LIVE_URI is None,
    reason="set the disposable Neo4j URI and explicit disposable confirmation",
)
def test_live_neo4j_write_is_idempotent_and_temporal_queries_are_exact():
    neo4j = pytest.importorskip("neo4j")
    base_bundle = circular_2016_03_fr_bundle()
    current_article_4 = base_bundle.provision_versions[0]
    predecessor_article_4 = ProvisionVersion(
        uid="version:bct:1991:24:article-4:predecessor-fixture",
        provision_uid=current_article_4.provision_uid,
        version_number=1,
        text="Synthetic verified predecessor text for disposable testing.",
        language="fr",
        valid_from=date(1991, 12, 1),
        valid_to=current_article_4.valid_from,
        status=VersionStatus.SUPERSEDED,
        content_hash="f" * 64,
        verification_status=VerificationStatus.VERIFIED,
    )
    completed_current_article_4 = current_article_4.model_copy(
        update={
            "version_number": 2,
            "supersedes_version_uid": predecessor_article_4.uid,
        }
    )
    completed_versions = (
        predecessor_article_4,
        completed_current_article_4,
        *base_bundle.provision_versions[1:],
    )
    completed_events = (
        base_bundle.change_events[0].model_copy(
            update={"retires_version_uids": (predecessor_article_4.uid,)}
        ),
        *base_bundle.change_events[1:],
    )
    reference = InstrumentReference(
        uid="reference:cir-2016-03:p2:cir-1991-24",
        source_instrument_uid="BCT:CIRCULAR:2016:03",
        target_instrument_uid="BCT:CIRCULAR:1991:24",
        evidence_uid=base_bundle.evidence_spans[0].uid,
        raw_citation="circulaire n°91-24",
        extraction_method="native",
        resolver_rule="french_bct_instrument_reference_v1",
        verification_status=VerificationStatus.VERIFIED,
        verification_method="manual_rendered_pdf_v1",
        rendered_image_sha256="d" * 64,
        verified_by="manual:test-reviewer",
    )
    bundle = base_bundle.model_copy(
        update={
            "provision_versions": completed_versions,
            "change_events": completed_events,
            "instrument_references": (reference,),
        }
    )
    fixture_uids = [
        item.uid
        for collection in (
            bundle.instruments,
            bundle.source_editions,
            bundle.pages,
            bundle.provisions,
            bundle.provision_versions,
            bundle.target_spans,
            bundle.evidence_spans,
            bundle.change_events,
            bundle.instrument_references,
        )
        for item in collection
    ]
    driver = neo4j.GraphDatabase.driver(DISPOSABLE_LIVE_URI, auth=None)
    try:
        existing = driver.execute_query(
            "MATCH (node) RETURN count(node) AS nodes",
            database_="neo4j",
        ).records[0]["nodes"]
        if existing:
            raise RuntimeError("disposable Neo4j test database must start empty")
        writer = Neo4jGraphWriter(driver)
        graph = Neo4jRegulatoryGraph(driver)

        first = writer.write_bundle(bundle)
        first_counts = graph.counts()
        second = writer.write_bundle(bundle)
        second_counts = graph.counts()

        assert first.bundle_sha256 == second.bundle_sha256
        assert first_counts == second_counts
        assert first_counts.nodes == bundle.node_count
        assert first_counts.relationships == 35
        assert graph.resolve_provision_as_of(
            "BCT:CIRCULAR:1991:24:ARTICLE:4", date(2016, 12, 29)
        ).version.uid == predecessor_article_4.uid
        assert graph.resolve_provision_as_of(
            "BCT:CIRCULAR:1991:24:ARTICLE:4", date(2016, 12, 30)
        ).version.uid == "version:bct:1991:24:article-4:2016-12-30"
        assert graph.resolve_provision_as_of(
            "BCT:CIRCULAR:1991:24:ARTICLE:16", date(2016, 8, 7)
        ).version is None
        assert graph.resolve_provision_as_of(
            "BCT:CIRCULAR:1991:24:ARTICLE:16", date(2016, 8, 8)
        ).version.uid == "version:bct:1991:24:article-16:2016-08-08"

        for provision_uid in (
            "BCT:CIRCULAR:1991:24:ARTICLE:4",
            "BCT:CIRCULAR:1993:08:ANNEX:13",
            "BCT:CIRCULAR:1991:24:ARTICLE:16",
        ):
            entries = graph.lineage(provision_uid)
            assert len(entries) == 1
            assert entries[0].action.value == "REPLACE"
            assert entries[0].predecessor_complete is (
                provision_uid == "BCT:CIRCULAR:1991:24:ARTICLE:4"
            )
            assert entries[0].source_filenames == ("Cir_2016_03_fr.pdf",)
            expected_event = next(
                event
                for event in bundle.change_events
                if provision_uid in event.target_provision_uids
            )
            expected_evidence = {
                evidence.uid: evidence
                for evidence in bundle.evidence_spans
                if evidence.uid in expected_event.evidence_uids
            }
            assert entries[0].evidence_uids == tuple(
                evidence.uid for evidence in entries[0].evidence
            )
            assert set(entries[0].evidence_uids) == set(expected_event.evidence_uids)
            assert all(
                evidence.page_number == expected_evidence[evidence.uid].page_number
                and evidence.quote == expected_evidence[evidence.uid].quote
                for evidence in entries[0].evidence
            )

        relationship_evidence = graph.relationship_evidence(
            (bundle.source_editions[0].filename,),
            limit=10,
        )
        assert any(
            item.relation_kind == "CITES"
            and item.related_instrument_uid == "BCT:CIRCULAR:1991:24"
            and item.filename == bundle.source_editions[0].filename
            and item.page_number == 2
            for item in relationship_evidence
        )
        affected = graph.affected_provisions(
            (
                SourcePageSeed(
                    filename=bundle.source_editions[0].filename,
                    page_numbers=(2,),
                ),
            ),
            limit=10,
        )
        assert tuple(item.uid for item in affected) == (
            "BCT:CIRCULAR:1991:24:ARTICLE:4",
        )
        temporal = RelationshipGraphRetriever(
            graph,
            current_date=date(2026, 8, 31),
        ).retrieve(
            "What rule in Article 4 is currently in force?",
            (
                Document(
                    page_content="ordinary seed",
                    metadata={
                        "source": bundle.source_editions[0].filename,
                        "page": 1,
                    },
                ),
            ),
        )
        assert temporal.trace.temporal_status == TemporalRetrievalStatus.RESOLVED
        assert temporal.requires_temporal_abstention is False
        assert completed_current_article_4.text in temporal.documents[0].page_content
    finally:
        driver.execute_query(
            "MATCH (node) WHERE node.uid IN $fixture_uids DETACH DELETE node",
            fixture_uids=fixture_uids,
            database_="neo4j",
        )
        driver.close()
