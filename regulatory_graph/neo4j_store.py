from dataclasses import dataclass
from datetime import date
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict

from regulatory_graph.corpus_structure import ObservedEdition, plan_structural_sync
from regulatory_graph.models import (
    ChangeEvent,
    EvidenceSpan,
    GraphChunk,
    GraphPage,
    Instrument,
    InstrumentReference,
    LegalAction,
    Provision,
    ProvisionVersion,
    RegulatoryGraphBundle,
    SourceEdition,
    TargetSpan,
    VerificationStatus,
    VersionStatus,
)
from regulatory_graph.schema import install_schema, is_allowed_relationship
from regulatory_graph.validation import (
    validate_change_event_for_write,
    validate_instrument_reference_for_write,
)


_NODE_COLLECTIONS = (
    ("Instrument", "instruments"),
    ("SourceEdition", "source_editions"),
    ("Page", "pages"),
    ("Chunk", "chunks"),
    ("Provision", "provisions"),
    ("ProvisionVersion", "provision_versions"),
    ("TargetSpan", "target_spans"),
    ("EvidenceSpan", "evidence_spans"),
    ("ChangeEvent", "change_events"),
    ("InstrumentReference", "instrument_references"),
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


@dataclass(frozen=True)
class GraphSnapshot:
    nodes: int
    relationships: int
    content_sha256: str


@dataclass(frozen=True)
class StructuralSyncReceipt:
    observed_edition_count: int
    skipped_edition_count: int
    written_edition_count: int
    repaired_edition_count: int
    candidate_edition_count: int
    bundle_sha256: str | None


class TemporalResolution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    provision_uid: str
    as_of: date
    version: ProvisionVersion | None
    reason: Literal["resolved", "no_verified_version"]


class LineageEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    uid: str
    filename: str
    page_number: int
    quote: str


class LineageEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    event_uid: str
    source_instrument_uid: str
    action: LegalAction
    effective_from: date | None
    introduced_version_uids: tuple[str, ...]
    retired_version_uids: tuple[str, ...]
    evidence: tuple[LineageEvidence, ...]
    predecessor_complete: bool

    @property
    def evidence_uids(self) -> tuple[str, ...]:
        return tuple(item.uid for item in self.evidence)

    @property
    def evidence_pages(self) -> tuple[int, ...]:
        return tuple(sorted({item.page_number for item in self.evidence}))

    @property
    def evidence_quotes(self) -> tuple[str, ...]:
        return tuple(item.quote for item in self.evidence)

    @property
    def source_filenames(self) -> tuple[str, ...]:
        return tuple(sorted({item.filename for item in self.evidence}))


class RelationshipEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seed_filename: str
    seed_instrument_uid: str
    related_instrument_uid: str
    relation_kind: Literal["CITES"] | LegalAction
    direction: Literal["OUTGOING", "INCOMING"]
    fact_uid: str
    evidence_uid: str
    filename: str
    page_number: int
    quote: str

    @property
    def relation_label(self) -> str:
        if isinstance(self.relation_kind, LegalAction):
            return self.relation_kind.value
        return self.relation_kind

    @property
    def path(self) -> str:
        if self.direction == "OUTGOING":
            source = self.seed_instrument_uid
            target = self.related_instrument_uid
        else:
            source = self.related_instrument_uid
            target = self.seed_instrument_uid
        return f"{source} -[{self.relation_label}]-> {target}"


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
        bundle = RegulatoryGraphBundle.model_validate(bundle.model_dump(mode="python"))
        references = _BundleReferences(bundle)
        for event in bundle.change_events:
            validate_change_event_for_write(event, references)
        for reference in bundle.instrument_references:
            validate_instrument_reference_for_write(reference, references)

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
            "Page", "HAS_CHUNK", "Chunk",
            ((item.page_uid, item.uid) for item in bundle.chunks),
        )
        chunks_by_page: dict[str, list[GraphChunk]] = {}
        for chunk in bundle.chunks:
            chunks_by_page.setdefault(chunk.page_uid, []).append(chunk)
        next_chunk_pairs = []
        for chunks in chunks_by_page.values():
            ordered = sorted(chunks, key=lambda item: item.chunk_index)
            next_chunk_pairs.extend(
                (left.uid, right.uid) for left, right in zip(ordered, ordered[1:])
            )
        self._relationship(
            "Chunk", "NEXT_CHUNK", "Chunk",
            next_chunk_pairs,
        )
        self._relationship(
            "Instrument", "HAS_PROVISION", "Provision",
            ((item.instrument_uid, item.uid) for item in bundle.provisions),
        )
        self._relationship(
            "Provision", "CONTAINS_PROVISION", "Provision",
            (
                (item.parent_provision_uid, item.uid)
                for item in bundle.provisions
                if item.parent_provision_uid is not None
            ),
        )
        self._relationship(
            "Provision", "HAS_VERSION", "ProvisionVersion",
            ((item.provision_uid, item.uid) for item in bundle.provision_versions),
        )
        current_rows = [
            {"source_uid": uid}
            for uid in sorted({item.provision_uid for item in bundle.provision_versions})
        ]
        if current_rows:
            self._execute(
                "UNWIND $rows AS row MATCH (source:Provision {uid: row.source_uid})"
                "-[relationship:CURRENT_VERSION]->() DELETE relationship",
                rows=current_rows,
            )
        self._relationship(
            "Provision", "CURRENT_VERSION", "ProvisionVersion",
            (
                (item.provision_uid, item.uid)
                for item in bundle.provision_versions
                if item.status == VersionStatus.ACTIVE
                and item.verification_status == VerificationStatus.VERIFIED
                and item.valid_to is None
            ),
        )
        self._relationship(
            "ProvisionVersion", "SUPERSEDES_VERSION", "ProvisionVersion",
            (
                (item.uid, item.supersedes_version_uid)
                for item in bundle.provision_versions
                if item.supersedes_version_uid is not None
            ),
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
            "EvidenceSpan", "IN_CHUNK", "Chunk",
            (
                (item.uid, item.chunk_uid)
                for item in bundle.evidence_spans
                if item.chunk_uid is not None
            ),
        )
        self._relationship(
            "Instrument", "DECLARES_CHANGE", "ChangeEvent",
            ((event.source_instrument_uid, event.uid) for event in bundle.change_events),
        )
        self._event_relationships(bundle.change_events)
        self._reference_relationships(bundle.instrument_references)

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

    def _reference_relationships(
        self,
        references: tuple[InstrumentReference, ...],
    ) -> None:
        self._relationship(
            "Instrument", "DECLARES_REFERENCE", "InstrumentReference",
            ((item.source_instrument_uid, item.uid) for item in references),
        )
        self._relationship(
            "InstrumentReference", "TARGETS", "Instrument",
            ((item.uid, item.target_instrument_uid) for item in references),
        )
        self._relationship(
            "InstrumentReference", "EVIDENCED_BY", "EvidenceSpan",
            ((item.uid, item.evidence_uid) for item in references),
        )

    def _relationship(
        self,
        source_label: str,
        relationship: str,
        target_label: str,
        pairs: Iterable[tuple[str, str]],
    ) -> None:
        if not is_allowed_relationship(source_label, relationship, target_label):
            raise ValueError(
                "graph relationship is not allowlisted: "
                f"{source_label}-[{relationship}]->{target_label}"
            )
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


class Neo4jStructuralWriter:
    """Resume corpus structure writes per immutable source edition."""

    def __init__(self, driver: Any, *, database: str = "neo4j"):
        self._driver = driver
        self._database = database

    def sync_bundle(self, bundle: RegulatoryGraphBundle) -> StructuralSyncReceipt:
        rows = _rows(
            self._driver.execute_query(
                "MATCH (edition:SourceEdition) "
                "OPTIONAL MATCH (edition)-[:HAS_PAGE]->(page:Page) "
                "OPTIONAL MATCH (page)-[:HAS_CHUNK]->(chunk:Chunk) "
                "WITH properties(edition) AS edition_properties, page, chunk "
                "RETURN edition_properties.uid AS uid, "
                "coalesce(edition_properties.logical_edition_uid, "
                "edition_properties.uid) AS logical_edition_uid, "
                "edition_properties.relative_path AS relative_path, "
                "edition_properties.sha256 AS sha256, "
                "edition_properties.extraction_artifact_hash "
                "AS extraction_artifact_hash, "
                "edition_properties.lifecycle_status AS lifecycle_status, "
                "edition_properties.identity_verification_status "
                "AS identity_verification_status, "
                "edition_properties.identity_evidence AS identity_evidence, "
                "edition_properties.identity_rule AS identity_rule, "
                "edition_properties.identity_evidence_text "
                "AS identity_evidence_text, "
                "collect(DISTINCT properties(page)) AS pages, "
                "collect(DISTINCT properties(chunk)) AS chunks",
                database_=self._database,
            )
        )
        observed = tuple(
            ObservedEdition(
                uid=row["uid"],
                logical_edition_uid=row["logical_edition_uid"],
                relative_path=row.get("relative_path"),
                sha256=row["sha256"],
                extraction_artifact_hash=row.get("extraction_artifact_hash"),
                page_uids=frozenset(
                    page["uid"] for page in row.get("pages", []) if page.get("uid")
                ),
                chunk_uids=frozenset(
                    chunk["uid"] for chunk in row.get("chunks", []) if chunk.get("uid")
                ),
                page_fingerprints=frozenset(
                    (
                        page["uid"],
                        page.get("source_sha256"),
                        page.get("extraction_artifact_hash"),
                        page.get("text_hash"),
                    )
                    for page in row.get("pages", [])
                    if page.get("uid")
                ),
                chunk_fingerprints=frozenset(
                    (
                        chunk["uid"],
                        chunk.get("source_sha256"),
                        chunk.get("extraction_artifact_hash"),
                        chunk.get("content_hash"),
                    )
                    for chunk in row.get("chunks", [])
                    if chunk.get("uid")
                ),
                lifecycle_status=row.get("lifecycle_status"),
                identity_verification_status=row.get("identity_verification_status"),
                identity_evidence=row.get("identity_evidence"),
                identity_rule=row.get("identity_rule"),
                identity_evidence_text=row.get("identity_evidence_text"),
            )
            for row in rows
        )
        plan = plan_structural_sync(bundle, observed)
        write_receipt = None
        if plan.bundle_to_write.source_editions:
            write_receipt = Neo4jGraphWriter(
                self._driver, database=self._database
            ).write_bundle(plan.bundle_to_write)
        return StructuralSyncReceipt(
            observed_edition_count=len(observed),
            skipped_edition_count=len(plan.skipped_edition_uids),
            written_edition_count=len(plan.bundle_to_write.source_editions),
            repaired_edition_count=len(plan.repaired_edition_uids),
            candidate_edition_count=len(plan.candidate_edition_uids),
            bundle_sha256=(write_receipt.bundle_sha256 if write_receipt else None),
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
            "OPTIONAL MATCH (event)-[retired_relation]->(retired:ProvisionVersion) "
            "WHERE type(retired_relation) = 'RETIRES_VERSION' "
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
                    evidence=tuple(LineageEvidence(**item) for item in evidence),
                    predecessor_complete=(row["action"] != "REPLACE" or bool(retired)),
                )
            )
        return tuple(entries)

    def relationship_evidence(
        self,
        seed_filenames: tuple[str, ...],
        *,
        limit: int = 10,
    ) -> tuple[RelationshipEvidence, ...]:
        if not 1 <= limit <= 50:
            raise ValueError("relationship evidence limit must be between 1 and 50")
        unique_filenames = list(dict.fromkeys(seed_filenames))
        if not unique_filenames:
            return ()
        result = self._execute(
            "UNWIND $seed_filenames AS seed_filename "
            "MATCH (seed:Instrument)-[:HAS_EDITION]->"
            "(:SourceEdition {filename: seed_filename}) "
            "CALL (seed) { "
            "MATCH (seed)-[declares]->"
            "(fact)-[:TARGETS]->(target) "
            "WHERE type(declares) IN ['DECLARES_REFERENCE', 'DECLARES_CHANGE'] "
            "AND fact.verification_status = 'VERIFIED' "
            "OPTIONAL MATCH (related_by_provision:Instrument)-[:HAS_PROVISION]->"
            "(target:Provision) "
            "WITH declares, fact, target, "
            "CASE WHEN 'Instrument' IN labels(target) THEN target "
            "ELSE related_by_provision END AS related "
            "WHERE related IS NOT NULL "
            "RETURN declares, fact, related, "
            "'OUTGOING' AS direction "
            "UNION ALL "
            "MATCH (related:Instrument)-[declares]->"
            "(fact)-[:TARGETS]->(target) "
            "WHERE type(declares) IN ['DECLARES_REFERENCE', 'DECLARES_CHANGE'] "
            "AND fact.verification_status = 'VERIFIED' "
            "AND (target = seed OR EXISTS { "
            "MATCH (seed)-[:HAS_PROVISION]->(target) }) "
            "RETURN declares, fact, related, "
            "'INCOMING' AS direction "
            "} "
            "MATCH (fact)-[:EVIDENCED_BY]->(evidence:EvidenceSpan)-[:ON_PAGE]->"
            "(page:Page)<-[:HAS_PAGE]-(edition:SourceEdition) "
            "RETURN DISTINCT seed_filename, seed.uid AS seed_instrument_uid, "
            "related.uid AS related_instrument_uid, "
            "CASE WHEN 'InstrumentReference' IN labels(fact) THEN 'CITES' "
            "ELSE fact.action END AS relation_kind, direction, "
            "fact.uid AS fact_uid, evidence.uid AS evidence_uid, "
            "edition.filename AS filename, page.page_number AS page_number, "
            "evidence.quote AS quote "
            "ORDER BY seed_filename, evidence_uid, related_instrument_uid "
            "LIMIT $limit",
            seed_filenames=unique_filenames,
            limit=limit,
        )
        return tuple(RelationshipEvidence(**row) for row in _rows(result))

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

    def snapshot(self) -> GraphSnapshot:
        nodes = _rows(
            self._execute(
                "MATCH (node) RETURN labels(node) AS labels, "
                "properties(node) AS properties"
            )
        )
        relationships = _rows(
            self._execute(
                "MATCH (source)-[relationship]->(target) "
                "RETURN source.uid AS source_uid, type(relationship) AS relationship_type, "
                "target.uid AS target_uid, properties(relationship) AS properties"
            )
        )
        payload = {
            "nodes": _canonical_rows(nodes),
            "relationships": _canonical_rows(relationships),
        }
        digest = sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            ).encode("utf-8")
        ).hexdigest()
        return GraphSnapshot(
            nodes=len(nodes),
            relationships=len(relationships),
            content_sha256=digest,
        )

    def _execute(self, statement: str, **parameters: Any) -> Any:
        return self._driver.execute_query(
            statement,
            **parameters,
            database_=self._database,
        )


def _properties(model: BaseModel) -> dict[str, Any]:
    return {
        key: _neo4j_value(value)
        for key, value in model.model_dump(mode="python").items()
        if value is not None
    }


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


def _canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        ),
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, Enum)):
        return value.isoformat() if isinstance(value, date) else value.value
    raise TypeError(f"unsupported graph snapshot value: {type(value).__name__}")
