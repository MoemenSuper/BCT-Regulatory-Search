from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from .models import RelationshipCandidate, RelationshipType, VerificationStatus


_RELATIONSHIP_TYPES = tuple(item.value for item in RelationshipType)

_REL_SET = """
SET r.verification_status=$status,
    r.evidence_file=$evidence_file,
    r.evidence_page=$evidence_page,
    r.evidence_quote=$evidence_quote,
    r.proposed_effective_date=$proposed_effective_date,
    r.target_in_catalog=$target_in_catalog,
    r.verification_method=$verification_method,
    r.active=$active,
    r.auto_verified_at=CASE WHEN $status = 'VERIFIED' THEN coalesce(r.auto_verified_at, datetime()) ELSE r.auto_verified_at END,
    r.updated_at=datetime()
"""


def _checked_relation_type(value: str) -> str:
    normalized = str(value).upper()
    if normalized not in _RELATIONSHIP_TYPES:
        raise ValueError(f"Unsupported relationship type: {value!r}")
    return normalized


@dataclass
class Neo4jGraphLiteStore:
    driver: Any
    database: str = "neo4j"

    def ensure_schema(self) -> None:
        statements = (
            "CREATE CONSTRAINT bct_instrument_id IF NOT EXISTS "
            "FOR (n:BCTInstrument) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT bct_provision_uid IF NOT EXISTS "
            "FOR (n:BCTProvision) REQUIRE n.uid IS UNIQUE",
        )
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement).consume()

    def upsert_candidate(self, candidate: RelationshipCandidate, *, active: bool = True) -> None:
        relation = _checked_relation_type(candidate.relation.value)
        params = candidate.as_dict()
        params["active"] = bool(active)
        params.update(
            {
                "source_id": candidate.source.id,
                "source_kind": candidate.source.kind,
                "source_year": candidate.source.year,
                "source_number": candidate.source.number,
                "source_filename": candidate.source.filename,
                "target_id": candidate.target.id,
                "target_kind": candidate.target.kind,
                "target_year": candidate.target.year,
                "target_number": candidate.target.number,
                "target_filename": candidate.target.filename,
                "status": candidate.verification_status.value,
            }
        )
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (s:BCTInstrument {id:$source_id})
                SET s.kind=$source_kind, s.year=$source_year, s.number=$source_number,
                    s.filename=coalesce($source_filename, s.filename)
                MERGE (t:BCTInstrument {id:$target_id})
                SET t.kind=$target_kind, t.year=$target_year, t.number=$target_number,
                    t.filename=coalesce($target_filename, t.filename)
                """,
                **params,
            ).consume()

            if candidate.source_provision is None:
                session.run(
                    f"""
                    MATCH (s:BCTInstrument {{id:$source_id}}), (t:BCTInstrument {{id:$target_id}})
                    MERGE (s)-[r:{relation} {{candidate_id:$candidate_id}}]->(t)
                    ON CREATE SET r.created_at=datetime()
                    {_REL_SET}
                    """,
                    **params,
                ).consume()
            else:
                source_uid = f"{candidate.source.id}#{candidate.source_provision.casefold()}"
                target_uid = f"{candidate.target.id}#{candidate.target_provision.casefold()}"
                params.update({"source_uid": source_uid, "target_uid": target_uid})
                session.run(
                    """
                    MATCH (s:BCTInstrument {id:$source_id}), (t:BCTInstrument {id:$target_id})
                    MERGE (sp:BCTProvision {uid:$source_uid})
                    SET sp.label=$source_provision, sp.instrument_id=$source_id
                    MERGE (tp:BCTProvision {uid:$target_uid})
                    SET tp.label=$target_provision, tp.instrument_id=$target_id
                    MERGE (sp)-[:BELONGS_TO]->(s)
                    MERGE (tp)-[:BELONGS_TO]->(t)
                    """,
                    **params,
                ).consume()
                session.run(
                    f"""
                    MATCH (sp:BCTProvision {{uid:$source_uid}}), (tp:BCTProvision {{uid:$target_uid}})
                    MERGE (sp)-[r:{relation} {{candidate_id:$candidate_id}}]->(tp)
                    ON CREATE SET r.created_at=datetime()
                    {_REL_SET}
                    """,
                    **params,
                ).consume()

    def list_candidates(
        self,
        *,
        status: VerificationStatus = VerificationStatus.VERIFIED,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        query = """
            MATCH (a)-[r]->(b)
            WHERE type(r) IN $types AND r.verification_status = $status
            RETURN r.candidate_id AS candidate_id, type(r) AS relation,
                   labels(a) AS source_labels, properties(a) AS source,
                   labels(b) AS target_labels, properties(b) AS target,
                   r.evidence_file AS evidence_file, r.evidence_page AS evidence_page,
                   r.evidence_quote AS evidence_quote,
                   r.proposed_effective_date AS proposed_effective_date,
                   r.target_in_catalog AS target_in_catalog,
                   r.verification_method AS verification_method,
                   coalesce(r.active, true) AS active
            ORDER BY r.evidence_file, r.evidence_page, r.candidate_id
            LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            return [
                record.data()
                for record in session.run(
                    query,
                    types=list(_RELATIONSHIP_TYPES),
                    status=status.value,
                    limit=limit,
                )
            ]

    def _count(self, query: str, **params) -> int:
        with self.driver.session(database=self.database) as session:
            record = session.run(query, types=list(_RELATIONSHIP_TYPES), **params).single()
        return int(record["changed"]) if record else 0

    def deactivate_source_relationships(self, evidence_file: str, *, keep_candidate_ids: list[str] | None = None) -> int:
        return self._count(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $types
              AND toLower(r.evidence_file) = toLower($evidence_file)
              AND NOT r.candidate_id IN $keep
            SET r.active = false, r.deactivated_at = datetime()
            RETURN count(r) AS changed
            """,
            evidence_file=Path(evidence_file).name,
            keep=list(dict.fromkeys(keep_candidate_ids or [])),
        )

    def activate_candidates(self, candidate_ids: list[str]) -> int:
        if not candidate_ids:
            return 0
        return self._count(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $types AND r.candidate_id IN $candidate_ids
            SET r.active = true, r.activated_at = datetime()
            RETURN count(r) AS changed
            """,
            candidate_ids=list(dict.fromkeys(candidate_ids)),
        )

    def delete_candidates(self, candidate_ids: list[str]) -> int:
        if not candidate_ids:
            return 0
        return self._count(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN $types AND r.candidate_id IN $candidate_ids
            DELETE r
            RETURN count(r) AS changed
            """,
            candidate_ids=list(dict.fromkeys(candidate_ids)),
        )

    def verified_edges_for_instruments(
        self,
        instrument_ids: tuple[str, ...],
        *,
        limit: int = 10,
        relationship_types: tuple[str, ...] | None = None,
    ) -> list[dict[str, object]]:
        if not instrument_ids:
            return []
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        selected_types = tuple(
            _checked_relation_type(value)
            for value in (relationship_types or _RELATIONSHIP_TYPES)
        )
        if not selected_types:
            return []
        query = """
            MATCH (a)-[r]->(b)
            WHERE type(r) IN $types
              AND r.verification_status = 'VERIFIED'
              AND coalesce(r.active, true) = true
              AND (
                (a:BCTInstrument AND a.id IN $ids) OR
                (b:BCTInstrument AND b.id IN $ids) OR
                (a:BCTProvision AND a.instrument_id IN $ids) OR
                (b:BCTProvision AND b.instrument_id IN $ids)
              )
            OPTIONAL MATCH (a)-[:BELONGS_TO]->(ai:BCTInstrument)
            OPTIONAL MATCH (b)-[:BELONGS_TO]->(bi:BCTInstrument)
            RETURN type(r) AS relation,
                   CASE WHEN a:BCTInstrument THEN a.id ELSE ai.id END AS source_id,
                   CASE WHEN b:BCTInstrument THEN b.id ELSE bi.id END AS target_id,
                   CASE WHEN a:BCTProvision THEN a.label ELSE null END AS source_provision,
                   CASE WHEN b:BCTProvision THEN b.label ELSE null END AS target_provision,
                   r.candidate_id AS candidate_id,
                   r.evidence_file AS evidence_file,
                   r.evidence_page AS evidence_page,
                   r.evidence_quote AS evidence_quote,
                   r.proposed_effective_date AS proposed_effective_date
            ORDER BY r.evidence_file, r.evidence_page
            LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            return [
                record.data()
                for record in session.run(
                    query,
                    types=list(selected_types),
                    ids=list(dict.fromkeys(instrument_ids)),
                    limit=limit,
                )
            ]


def open_neo4j_driver_from_environment():
    try:
        from neo4j import GraphDatabase
    except ImportError as error:
        raise RuntimeError(
            "Neo4j graph support is not installed. Install requirements-graph.txt."
        ) from error

    uri = os.environ.get("BCT_NEO4J_URI", "bolt://127.0.0.1:7687").strip()
    user = os.environ.get("BCT_NEO4J_USERNAME", "neo4j")
    password = os.environ.get("BCT_NEO4J_PASSWORD")
    auth = None if os.environ.get("BCT_NEO4J_AUTH", "password") == "none" else (user, password)
    if auth is not None and not password:
        raise RuntimeError("BCT_NEO4J_PASSWORD is required when Neo4j authentication is enabled")
    driver = GraphDatabase.driver(uri, auth=auth, connection_timeout=3.0)
    driver.verify_connectivity()
    return driver
