from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from graph_contract import (
    GraphRetrievalResult,
    GraphRetrievalStatus,
    GraphRetrievalTrace,
    TemporalFailureReason,
    TemporalRetrievalStatus,
    is_relationship_query,
    is_temporal_rule_query,
)
from retrieval_selection import parse_query_identity, parse_source_identity

from .identity import canonical_instrument_id, normalize_kind
from .models import RelationshipType
from .store import Neo4jGraphLiteStore, open_neo4j_driver_from_environment


logger = logging.getLogger(__name__)


def _seed_ids(documents: Iterable[Document], *, limit: int = 5) -> tuple[str, ...]:
    ids: list[str] = []
    for document in documents:
        source = Path(str(document.metadata.get("source", ""))).name
        identity = parse_source_identity(source)
        if identity is None:
            continue
        kind = normalize_kind(identity.get("kind"))
        if kind is None:
            continue
        value = canonical_instrument_id(kind, identity["year"], identity["number"])
        if value not in ids:
            ids.append(value)
        if len(ids) >= limit:
            break
    return tuple(ids)


def _query_seed_id(query: str) -> str | None:
    """Use an explicit circular/note identity even when ordinary retrieval misses it."""
    identity = parse_query_identity(query)
    if identity is None or identity.get("kind") is None or identity.get("number") is None:
        return None
    kind = normalize_kind(identity.get("kind"))
    if kind is None:
        return None
    try:
        return canonical_instrument_id(kind, int(identity["year"]), int(identity["number"]))
    except (TypeError, ValueError):
        return None


def _relationship_document(edge: dict[str, object]) -> Document | None:
    quote = str(edge.get("evidence_quote") or "").strip()
    source = Path(str(edge.get("evidence_file") or "")).name
    try:
        page = int(edge.get("evidence_page"))
    except (TypeError, ValueError):
        return None
    if not quote or not source or page < 1:
        return None
    metadata = {
        "source": source,
        # Set both keys: answer citations use page_label; diversify uses page.
        "page": page,
        "page_label": page,
        "temporal_relation": str(edge.get("relation") or ""),
        "temporal_source_id": str(edge.get("source_id") or ""),
        "temporal_target_id": str(edge.get("target_id") or ""),
        "temporal_source_provision": edge.get("source_provision"),
        "temporal_target_provision": edge.get("target_provision"),
        "temporal_verification": "VERIFIED_RELATIONSHIP_ONLY",
    }
    return Document(page_content=quote, metadata=metadata)


def _trace(
    status: GraphRetrievalStatus,
    *,
    temporal_intent: bool,
    temporal_status: TemporalRetrievalStatus = TemporalRetrievalStatus.NO_CANDIDATE,
    temporal_reason: TemporalFailureReason | None = None,
    **kwargs,
) -> GraphRetrievalTrace:
    return GraphRetrievalTrace(
        status=status,
        temporal_status=temporal_status if temporal_intent else TemporalRetrievalStatus.NOT_REQUESTED,
        temporal_reason=temporal_reason if temporal_intent else None,
        **kwargs,
    )


class RelationshipGraphLiteRetriever:
    """Two-hop, verified-only relationship expansion.

    This deliberately does not pretend to resolve a complete temporal legal
    lineage.  For current/as-of questions it contributes verified amendment or
    replacement evidence while leaving temporal resolution marked incomplete.
    """

    def __init__(self, store: Neo4jGraphLiteStore, *, max_evidence: int = 8):
        if max_evidence < 1:
            raise ValueError("Graph Lite limits are invalid")
        self.store = store
        self.max_evidence = max_evidence

    def retrieve(self, query: str, seed_documents: Iterable[Document]) -> GraphRetrievalResult:
        relationship_intent = is_relationship_query(query)
        temporal_intent = is_temporal_rule_query(query)
        if not relationship_intent and not temporal_intent:
            return GraphRetrievalResult(
                documents=(),
                trace=GraphRetrievalTrace(status=GraphRetrievalStatus.NOT_REQUESTED),
            )

        seed_ids = list(_seed_ids(seed_documents))
        explicit_seed = _query_seed_id(query)
        if explicit_seed and explicit_seed not in seed_ids:
            seed_ids.insert(0, explicit_seed)
        seed_ids = tuple(seed_ids[:5])
        if not seed_ids:
            return GraphRetrievalResult(
                documents=(),
                trace=_trace(
                    GraphRetrievalStatus.NO_SEED,
                    temporal_intent=temporal_intent,
                    temporal_reason=TemporalFailureReason.NO_RETRIEVAL_SEED,
                ),
            )

        relation_types = (
            (
                RelationshipType.AMENDS.value,
                RelationshipType.REPLACES.value,
                RelationshipType.ABROGATES.value,
            )
            if temporal_intent
            else tuple(item.value for item in RelationshipType)
        )
        try:
            edges = self.store.verified_edges_for_instruments(
                seed_ids,
                limit=self.max_evidence,
                relationship_types=relation_types,
            )
            # A tiny two-hop expansion is enough to expose chains such as
            # 2017 -> 2020 -> 2025 without introducing a general graph search.
            if len(edges) < self.max_evidence:
                next_ids = []
                seen_seed_ids = set(seed_ids)
                for edge in edges:
                    for key in ("source_id", "target_id"):
                        value = str(edge.get(key) or "")
                        if value and value not in seen_seed_ids and value not in next_ids:
                            next_ids.append(value)
                if next_ids:
                    second = self.store.verified_edges_for_instruments(
                        tuple(next_ids),
                        limit=self.max_evidence - len(edges),
                        relationship_types=relation_types,
                    )
                    seen_candidates = {str(edge.get("candidate_id") or "") for edge in edges}
                    for edge in second:
                        candidate_id = str(edge.get("candidate_id") or "")
                        if candidate_id and candidate_id in seen_candidates:
                            continue
                        edges.append(edge)
                        if candidate_id:
                            seen_candidates.add(candidate_id)
                        if len(edges) >= self.max_evidence:
                            break
        except Exception as error:
            logger.warning("Graph Lite query failed: %s", type(error).__name__)
            return GraphRetrievalResult(
                documents=(),
                trace=_trace(
                    GraphRetrievalStatus.UNAVAILABLE,
                    temporal_intent=temporal_intent,
                    temporal_status=TemporalRetrievalStatus.UNAVAILABLE,
                    temporal_reason=TemporalFailureReason.TEMPORAL_GRAPH_UNAVAILABLE,
                    seed_filenames=seed_ids,
                    error_type=type(error).__name__,
                ),
            )

        documents = tuple(
            document
            for edge in edges
            if (document := _relationship_document(edge)) is not None
        )
        if not documents:
            return GraphRetrievalResult(
                documents=(),
                trace=_trace(
                    GraphRetrievalStatus.NO_EVIDENCE,
                    temporal_intent=temporal_intent,
                    temporal_status=TemporalRetrievalStatus.INCOMPLETE,
                    temporal_reason=TemporalFailureReason.RELATIONSHIP_ONLY_NOT_PROVISION_RESOLVED,
                    seed_filenames=seed_ids,
                ),
            )

        paths = tuple(
            f"{edge.get('source_id')} -[{edge.get('relation')}]-> {edge.get('target_id')}"
            for edge in edges
        )
        return GraphRetrievalResult(
            documents=documents,
            trace=_trace(
                GraphRetrievalStatus.EXPANDED,
                temporal_intent=temporal_intent,
                temporal_status=TemporalRetrievalStatus.INCOMPLETE,
                temporal_reason=TemporalFailureReason.RELATIONSHIP_ONLY_NOT_PROVISION_RESOLVED,
                seed_filenames=seed_ids,
                evidence_count=len(documents),
                paths=paths,
            ),
        )


@dataclass
class GraphLiteRuntime:
    retriever: RelationshipGraphLiteRetriever
    driver: object

    def close(self) -> None:
        self.driver.close()


def open_relationship_graph_runtime() -> GraphLiteRuntime | None:
    if os.environ.get("BCT_ENABLE_GRAPH") != "1":
        return None
    try:
        driver = open_neo4j_driver_from_environment()
    except Exception as error:
        logger.warning(
            "Graph Lite is enabled but Neo4j is unavailable; ordinary retrieval remains active: %s",
            type(error).__name__,
        )
        return None
    database = os.environ.get("BCT_NEO4J_DATABASE", "neo4j")
    store = Neo4jGraphLiteStore(driver=driver, database=database)
    return GraphLiteRuntime(retriever=RelationshipGraphLiteRetriever(store), driver=driver)
