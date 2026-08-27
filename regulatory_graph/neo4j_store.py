from dataclasses import dataclass
from datetime import date
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict

from regulatory_graph.models import (
    ChangeEvent,
    EvidenceSpan,
    GraphPage,
    Instrument,
    LegalAction,
    Provision,
    ProvisionVersion,
    RegulatoryGraphBundle,
    SourceEdition,
    TargetSpan,
)
from regulatory_graph.schema import install_schema
from regulatory_graph.validation import validate_change_event_for_write


_NODE_COLLECTIONS = (
    ("Instrument", "instruments"),
    ("SourceEdition", "source_editions"),
    ("Page", "pages"),
    ("Provision", "provisions"),
    ("ProvisionVersion", "provision_versions"),
    ("TargetSpan", "target_spans"),
    ("EvidenceSpan", "evidence_spans"),
    ("ChangeEvent", "change_events"),
)


_NODE_QUERIES = {
    label: f"UNWIND $rows AS row MERGE (node:{label} {{uid: row.uid}}) SET node += row"
    for label, _ in _NODE_COLLECTIONS
}


@dataclass(frozen=True)
class WriteReceipt:
    bundle_sha256: str
    change_event_count: int


@dataclass(frozen=True)
class GraphCounts:
    nodes: int
    relationships: int


class TemporalResolution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    provision_uid: str
    as_of: date
    version: ProvisionVersion | None
    reason: Literal["resolved", "no_verified_version"]


class LineageEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_uid: str
    source_instrument_uid: str
    action: LegalAction
    effective_from: date | None
    introduced_version_uids: tuple[str, ...]
    retired_version_uids: tuple[str, ...]
    evidence_uids: tuple[str, ...]
    evidence_pages: tuple[int, ...]
    evidence_quotes: tuple[str, ...]
    source_filenames: tuple[str, ...]
    predecessor_complete: bool


class _BundleReferences:
    def __init__(self, bundle: RegulatoryGraphBundle):
        self._ids = {
            label: {item.uid for item in getattr(bundle, attribute)}
            for label, attribute in _NODE_COLLECTIONS
        }

    def exists(self, label: str, uid: str) -> bool:
        return uid in self._ids[label]


class Neo4jGraphWriter:
    def __init__(self, driver: Any, *, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def write_bundle(self, bundle: RegulatoryGraphBundle) -> WriteReceipt:
        references = _BundleReferences(bundle)
        for event in bundle.change_events:
            validate_change_event_for_write(event, references)

        install_schema(self._driver, database=self._database)
        for label, attribute in _NODE_COLLECTIONS:
            items = getattr(bundle, attribute)
            if items:
                self._execute(_NODE_QUERIES[label], rows=[_properties(item) for item in items])

        self._write_relationships(bundle)
        payload = bundle.model_dump(mode="json")
        digest = sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return WriteReceipt(
            bundle_sha256=digest,
            change_event_count=len(bundle.change_events),
        )

    def _write_relationships(self, bundle: RegulatoryGraphBundle) -> None:
        page_uids = {
            (page.source_edition_uid, page.page_number): page.uid for page in bundle.pages
        }
        self._relationship(
            "Instrument", "HAS_EDITION", "SourceEdition",
            ((item.instrument_uid, item.uid) for item in bundle.source_editions),
        )
        self._relationship(
            "SourceEdition", "HAS_PAGE", "Page",
            ((item.source_edition_uid, item.uid) for item in bundle.pages),
        )
        self._relationship(
            "Instrument", "HAS_PROVISION", "Provision",
            ((item.instrument_uid, item.uid) for item in bundle.provisions),
        )
        self._relationship(
            "Provision", "HAS_VERSION", "ProvisionVersion",
            ((item.provision_uid, item.uid) for item in bundle.provision_versions),
        )
        self._relationship(
            "TargetSpan", "WITHIN", "ProvisionVersion",
            ((item.uid, item.provision_version_uid) for item in bundle.target_spans),
        )
        self._relationship(
            "EvidenceSpan", "ON_PAGE", "Page",
            (
                (item.uid, page_uids[(item.source_edition_uid, item.page_number)])
                for item in bundle.evidence_spans
            ),
        )
        self._relationship(
            "Instrument", "DECLARES_CHANGE", "ChangeEvent",
            ((event.source_instrument_uid, event.uid) for event in bundle.change_events),
        )
        self._event_relationships(bundle.change_events)

    def _event_relationships(self, events: tuple[ChangeEvent, ...]) -> None:
        self._relationship(
            "ChangeEvent", "TARGETS", "Instrument",
            ((event.uid, uid) for event in events for uid in event.target_instrument_uids),
        )
        self._relationship(
            "ChangeEvent", "TARGETS", "Provision",
            ((event.uid, uid) for event in events for uid in event.target_provision_uids),
        )
        self._relationship(
            "ChangeEvent", "TARGETS", "TargetSpan",
            ((event.uid, uid) for event in events for uid in event.target_span_uids),
        )
        self._relationship(
            "ChangeEvent", "RETIRES_VERSION", "ProvisionVersion",
            ((event.uid, uid) for event in events for uid in event.retires_version_uids),
        )
        self._relationship(
            "ChangeEvent", "INTRODUCES_VERSION", "ProvisionVersion",
            ((event.uid, uid) for event in events for uid in event.introduces_version_uids),
        )
        self._relationship(
            "ChangeEvent", "EVIDENCED_BY", "EvidenceSpan",
            ((event.uid, uid) for event in events for uid in event.evidence_uids),
        )

    def _relationship(
        self,
        source_label: str,
        relationship: str,
        target_label: str,
        pairs: Iterable[tuple[str, str]],
    ) -> None:
        rows = [{"source_uid": source, "target_uid": target} for source, target in pairs]
        if not rows:
            return
        query = (
            f"UNWIND $rows AS row MATCH (source:{source_label} {{uid: row.source_uid}}) "
            f"MATCH (target:{target_label} {{uid: row.target_uid}}) "
            f"MERGE (source)-[:{relationship}]->(target)"
        )
        self._execute(query, rows=rows)

    def _execute(self, statement: str, **parameters: Any) -> Any:
        return self._driver.execute_query(
            statement,
            **parameters,
            database_=self._database,
        )


class Neo4jRegulatoryGraph:
    def __init__(self, driver: Any, *, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def resolve_provision_as_of(
        self,
        provision_uid: str,
        as_of: date,
    ) -> TemporalResolution:
        result = self._execute(
            "MATCH (:Provision {uid: $provision_uid})-[:HAS_VERSION]->"
            "(version:ProvisionVersion) "
            "WHERE version.verification_status = 'VERIFIED' "
            "AND version.valid_from <= $as_of "
            "AND (version.valid_to IS NULL OR $as_of < version.valid_to) "
            "RETURN properties(version) AS version ORDER BY version.version_number",
            provision_uid=provision_uid,
            as_of=as_of,
        )
        rows = _rows(result)
        if len(rows) > 1:
            raise ValueError(
                f"multiple verified versions overlap for {provision_uid} at {as_of}"
            )
        version = ProvisionVersion(**rows[0]["version"]) if rows else None
        return TemporalResolution(
            provision_uid=provision_uid,
            as_of=as_of,
            version=version,
            reason="resolved" if version else "no_verified_version",
        )

    def lineage(self, provision_uid: str) -> tuple[LineageEntry, ...]:
        result = self._execute(
            "MATCH (target:Provision {uid: $provision_uid})<-[:TARGETS]-"
            "(event:ChangeEvent)<-[:DECLARES_CHANGE]-(source:Instrument) "
            "OPTIONAL MATCH (event)-[:INTRODUCES_VERSION]->(introduced:ProvisionVersion) "
            "WITH event, source, collect(DISTINCT introduced.uid) AS introduced_uids "
            "OPTIONAL MATCH (event)-[:RETIRES_VERSION]->(retired:ProvisionVersion) "
            "WITH event, source, introduced_uids, collect(DISTINCT retired.uid) AS retired_uids "
            "MATCH (event)-[:EVIDENCED_BY]->(evidence:EvidenceSpan)-[:ON_PAGE]->"
            "(page:Page)<-[:HAS_PAGE]-(edition:SourceEdition) "
            "RETURN event.uid AS event_uid, source.uid AS source_instrument_uid, "
            "event.action AS action, event.effective_from AS effective_from, "
            "introduced_uids AS introduced_version_uids, retired_uids AS retired_version_uids, "
            "collect(DISTINCT {uid: evidence.uid, filename: edition.filename, "
            "page_number: page.page_number, quote: evidence.quote}) AS evidence "
            "ORDER BY effective_from, event_uid",
            provision_uid=provision_uid,
        )
        entries = []
        for row in _rows(result):
            evidence = sorted(row["evidence"], key=lambda item: (item["page_number"], item["uid"]))
            retired = tuple(sorted(uid for uid in row["retired_version_uids"] if uid))
            entries.append(
                LineageEntry(
                    event_uid=row["event_uid"],
                    source_instrument_uid=row["source_instrument_uid"],
                    action=row["action"],
                    effective_from=row["effective_from"],
                    introduced_version_uids=tuple(
                        sorted(uid for uid in row["introduced_version_uids"] if uid)
                    ),
                    retired_version_uids=retired,
                    evidence_uids=tuple(item["uid"] for item in evidence),
                    evidence_pages=tuple(sorted({item["page_number"] for item in evidence})),
                    evidence_quotes=tuple(item["quote"] for item in evidence),
                    source_filenames=tuple(sorted({item["filename"] for item in evidence})),
                    predecessor_complete=(row["action"] != "REPLACE" or bool(retired)),
                )
            )
        return tuple(entries)

    def counts(self) -> GraphCounts:
        rows = _rows(
            self._execute(
                "MATCH (node) WITH count(node) AS nodes "
                "OPTIONAL MATCH ()-[relationship]->() "
                "RETURN nodes, count(relationship) AS relationships"
            )
        )
        return GraphCounts(
            nodes=rows[0]["nodes"] if rows else 0,
            relationships=rows[0]["relationships"] if rows else 0,
        )

    def _execute(self, statement: str, **parameters: Any) -> Any:
        return self._driver.execute_query(
            statement,
            **parameters,
            database_=self._database,
        )


def _properties(model: BaseModel) -> dict[str, Any]:
    return {key: _neo4j_value(value) for key, value in model.model_dump(mode="python").items()}


def _neo4j_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_neo4j_value(item) for item in value]
    return value


def _rows(result: Any) -> list[dict[str, Any]]:
    records = result.records if hasattr(result, "records") else result
    if isinstance(records, tuple) and len(records) == 3:
        records = records[0]
    return [
        _native_value(record.data() if hasattr(record, "data") else dict(record))
        for record in records
    ]


def _native_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _native_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native_value(item) for item in value]
    if hasattr(value, "to_native"):
        return value.to_native()
    return value
