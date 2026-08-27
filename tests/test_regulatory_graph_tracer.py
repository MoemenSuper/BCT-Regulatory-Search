from datetime import date
import os

import pytest

from regulatory_graph.fixtures import circular_2016_03_fr_bundle
from regulatory_graph.models import VerificationStatus
from regulatory_graph.neo4j_store import Neo4jGraphWriter, Neo4jRegulatoryGraph


class FakeDriver:
    def __init__(self, responses=()):
        self.calls = []
        self.responses = list(responses)

    def execute_query(self, statement, **parameters):
        self.calls.append((statement, parameters))
        if self.responses:
            return self.responses.pop(0)
        return []


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


@pytest.mark.skipif(
    not os.environ.get("BCT_NEO4J_TEST_URI"),
    reason="set BCT_NEO4J_TEST_URI to run disposable Neo4j integration",
)
def test_live_neo4j_write_is_idempotent_and_temporal_queries_are_exact():
    neo4j = pytest.importorskip("neo4j")
    uri = os.environ["BCT_NEO4J_TEST_URI"]
    driver = neo4j.GraphDatabase.driver(uri, auth=None)
    try:
        driver.execute_query("MATCH (node) DETACH DELETE node", database_="neo4j")
        bundle = circular_2016_03_fr_bundle()
        writer = Neo4jGraphWriter(driver)
        graph = Neo4jRegulatoryGraph(driver)

        first = writer.write_bundle(bundle)
        first_counts = graph.counts()
        second = writer.write_bundle(bundle)
        second_counts = graph.counts()

        assert first.bundle_sha256 == second.bundle_sha256
        assert first_counts == second_counts
        assert first_counts.nodes == bundle.node_count
        assert graph.resolve_provision_as_of(
            "BCT:CIRCULAR:1991:24:ARTICLE:4", date(2016, 12, 29)
        ).version is None
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
            assert entries[0].predecessor_complete is False
            assert entries[0].source_filenames == ("Cir_2016_03_fr.pdf",)
    finally:
        driver.execute_query("MATCH (node) DETACH DELETE node", database_="neo4j")
        driver.close()
