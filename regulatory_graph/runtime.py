from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Protocol

from langchain_core.documents import Document

from regulatory_graph.neo4j_store import Neo4jRegulatoryGraph, RelationshipEvidence


logger = logging.getLogger(__name__)


_RELATIONSHIP_SIGNALS = (
    "relation entre",
    "relations entre",
    "lien entre",
    "liens entre",
    "fait référence",
    "font référence",
    "référence à",
    "cite la circulaire",
    "cite le document",
    "modifie",
    "remplace",
    "abroge",
    "dérogation",
    "prédécesseur",
    "successeur",
    "relationship between",
    "relation between",
    "refers to",
    "references the",
    "cites the",
    "amends",
    "replaces",
    "repeals",
    "supersedes",
    "ما العلاقة",
    "العلاقة بين",
    "يشير إلى",
    "تحيل إلى",
    "يعدل",
    "ينقح",
    "يعوض",
    "يلغي",
)


class GraphRetrievalStatus(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    NO_SEED = "NO_SEED"
    NO_EVIDENCE = "NO_EVIDENCE"
    EXPANDED = "EXPANDED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class GraphRetrievalTrace:
    status: GraphRetrievalStatus
    seed_filenames: tuple[str, ...] = ()
    evidence_count: int = 0
    paths: tuple[str, ...] = ()
    error_type: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "seed_filenames": list(self.seed_filenames),
            "evidence_count": self.evidence_count,
            "paths": list(self.paths),
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class GraphRetrievalResult:
    documents: tuple[Document, ...]
    trace: GraphRetrievalTrace


class RelationshipEvidenceGraph(Protocol):
    def relationship_evidence(
        self,
        seed_filenames: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[RelationshipEvidence, ...]: ...


def is_relationship_query(query: str) -> bool:
    normalized = " ".join(query.casefold().split())
    return any(signal in normalized for signal in _RELATIONSHIP_SIGNALS)


class RelationshipGraphRetriever:
    def __init__(
        self,
        graph: RelationshipEvidenceGraph,
        *,
        max_seeds: int = 5,
        max_evidence: int = 10,
    ):
        if max_seeds < 1 or max_evidence < 1:
            raise ValueError("GraphRAG limits must be positive")
        self._graph = graph
        self._max_seeds = max_seeds
        self._max_evidence = max_evidence

    def retrieve(
        self,
        query: str,
        seed_documents: Iterable[Document],
    ) -> GraphRetrievalResult:
        if not is_relationship_query(query):
            return _empty_result(GraphRetrievalStatus.NOT_REQUESTED)

        seed_filenames = _seed_filenames(seed_documents, limit=self._max_seeds)
        if not seed_filenames:
            return _empty_result(GraphRetrievalStatus.NO_SEED)

        try:
            evidence = self._graph.relationship_evidence(
                seed_filenames,
                limit=self._max_evidence,
            )
        except Exception as error:
            logger.warning(
                "Neo4j relationship expansion unavailable: %s",
                type(error).__name__,
            )
            return GraphRetrievalResult(
                documents=(),
                trace=GraphRetrievalTrace(
                    status=GraphRetrievalStatus.UNAVAILABLE,
                    seed_filenames=seed_filenames,
                    error_type=type(error).__name__,
                ),
            )

        if not evidence:
            return GraphRetrievalResult(
                documents=(),
                trace=GraphRetrievalTrace(
                    status=GraphRetrievalStatus.NO_EVIDENCE,
                    seed_filenames=seed_filenames,
                ),
            )

        documents = tuple(_evidence_document(item) for item in evidence)
        paths = tuple(dict.fromkeys(item.path for item in evidence))
        return GraphRetrievalResult(
            documents=documents,
            trace=GraphRetrievalTrace(
                status=GraphRetrievalStatus.EXPANDED,
                seed_filenames=seed_filenames,
                evidence_count=len(evidence),
                paths=paths,
            ),
        )


def open_relationship_graph_runtime(
    *,
    uri: str | None = None,
    database: str | None = None,
) -> tuple[Any | None, RelationshipGraphRetriever | None]:
    resolved_uri = (
        uri
        if uri is not None
        else os.environ.get("BCT_NEO4J_URI", "bolt://127.0.0.1:17687")
    ).strip()
    if not resolved_uri:
        return None, None
    resolved_database = database or os.environ.get("BCT_NEO4J_DATABASE", "neo4j")

    driver = None
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            resolved_uri,
            auth=None,
            connection_timeout=2.0,
        )
        driver.verify_connectivity()
    except Exception as error:
        if driver is not None:
            driver.close()
        logger.warning(
            "Local Neo4j GraphRAG is unavailable; ordinary retrieval remains active: %s",
            type(error).__name__,
        )
        return None, None

    graph = Neo4jRegulatoryGraph(driver, database=resolved_database)
    return driver, RelationshipGraphRetriever(graph)


def _seed_filenames(
    documents: Iterable[Document],
    *,
    limit: int,
) -> tuple[str, ...]:
    filenames = []
    seen = set()
    for document in documents:
        source = document.metadata.get("source")
        filename = Path(str(source)).name if source else ""
        if not filename or filename in seen:
            continue
        seen.add(filename)
        filenames.append(filename)
        if len(filenames) == limit:
            break
    return tuple(filenames)


def _evidence_document(evidence: RelationshipEvidence) -> Document:
    content = (
        "Preuve relationnelle vérifiée du graphe Neo4j.\n"
        f"Chemin: {evidence.path}\n"
        f"Sens: {evidence.direction}\n"
        f"Extrait source: {evidence.quote}"
    )
    return Document(
        page_content=content,
        metadata={
            "source": evidence.filename,
            "page": evidence.page_number - 1,
            "page_label": evidence.page_number,
            "retrieval_source": "neo4j_relationship",
            "graph_direction": evidence.direction,
            "graph_fact_uid": evidence.fact_uid,
            "graph_path": evidence.path,
            "graph_relation": evidence.relation_kind,
        },
    )


def _empty_result(status: GraphRetrievalStatus) -> GraphRetrievalResult:
    return GraphRetrievalResult(
        documents=(),
        trace=GraphRetrievalTrace(status=status),
    )
