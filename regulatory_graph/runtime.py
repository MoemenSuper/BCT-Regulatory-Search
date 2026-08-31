from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
import logging
import os
from pathlib import Path
import re
from typing import Any, Iterable, Protocol

from langchain_core.documents import Document

from regulatory_graph.models import LegalAction
from regulatory_graph.neo4j_store import (
    AffectedProvision,
    LineageEntry,
    Neo4jRegulatoryGraph,
    RelationshipEvidence,
    SourcePageSeed,
    TemporalResolution,
)


logger = logging.getLogger(__name__)


_FRENCH_DOCUMENT = r"(?:documents?|circulaires?|notes?|textes?|instruments?)"
_FRENCH_RELATION = (
    r"(?:cit(?:e|ent|é|ée|és|ées)|référenc(?:e|ent|é|ée|és|ées)|"
    r"(?:fait|font)\s+référence|"
    r"modifi(?:e|ent|é|ée|és|ées)|remplac(?:e|ent|é|ée|és|ées)|"
    r"abrog(?:e|ent|é|ée|és|ées)|dérog(?:e|ent|ation)|"
    r"li(?:e|ent|é|ée|és|ées)|prédécesseur|successeur)"
)
_ENGLISH_DOCUMENT = r"(?:documents?|circulars?|notes?|texts?|instruments?)"
_ENGLISH_RELATION = (
    r"(?:cite|cites|cited|refer\s+to|refers\s+to|reference|references|"
    r"referenced|amend|amends|amended|replace|replaces|replaced|"
    r"repeal|repeals|repealed|supersede|supersedes|superseded|"
    r"related|predecessor|successor)"
)
_ARABIC_DOCUMENT = r"(?:المنشور(?:ات|ين)?|منشور|المذكرة|مذكرة|الوثيقة|وثيقة)"
_ARABIC_RELATION = (
    r"(?:يشير|تشير|تحيل|يعدل|تعدل|ينقح|تنقح|يعوض|تعوض|يلغي|تلغي|"
    r"يستبدل|تستبدل|يرتبط|ترتبط|مرتبطة)"
)
_REGULATORY_DOCUMENT_CONTEXT = re.compile(
    r"\b(?:circulaires?|notes?|instruments?|"
    r"documents?\s+réglementaires?|textes?\s+réglementaires?|"
    r"circulars?|regulatory\s+(?:documents?|notes?|texts?|instruments?))\b|"
    rf"{_ARABIC_DOCUMENT}",
    re.IGNORECASE,
)
_RELATIONSHIP_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:relations?|liens?)\s+entre\b",
        rf"\b{_FRENCH_DOCUMENT}\b.{{0,120}}\b{_FRENCH_RELATION}\b",
        rf"\b{_FRENCH_RELATION}\b.{{0,120}}\b{_FRENCH_DOCUMENT}\b",
        r"\b(?:relationship|relation|link)s?\s+between\b",
        rf"\b{_ENGLISH_DOCUMENT}\b.{{0,120}}\b{_ENGLISH_RELATION}\b",
        rf"\b{_ENGLISH_RELATION}\b.{{0,120}}\b{_ENGLISH_DOCUMENT}\b",
        r"(?:ما\s+)?العلاقة\s+بين",
        rf"{_ARABIC_DOCUMENT}.{{0,120}}{_ARABIC_RELATION}",
        rf"{_ARABIC_RELATION}.{{0,120}}{_ARABIC_DOCUMENT}",
    )
)
_ANAPHORIC_RELATIONSHIP_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwhat\s+(?:did|does)\s+(?:it|this|that)\s+"
        r"(?:cite|amend|replace|repeal|supersede)\b",
        r"\bwhat\s+(?:is|was)\s+(?:its|the)\s+(?:predecessor|successor)\b",
        r"\bque\s+(?:cite|modifie|remplace|abroge)"
        r"(?:-t-(?:elle|il)|\s+(?:elle|il))?\b",
        r"\bquel(?:le)?\s+est\s+son\s+(?:prédécesseur|successeur)\b",
        r"(?:ما\s+الذي|ماذا)\s+(?:يلغيه|يعدله|يستبدله|يستشهد\s+به)",
    )
)
_TEMPORAL_RULE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:current|currently|in\s+force|applicable\s+(?:rule|provision)|"
        r"changed\s+across\s+years|over\s+the\s+years|as\s+of)\b",
        r"\b(?:actuel(?:le)?|en\s+vigueur|règle\s+applicable|"
        r"au\s+fil\s+des\s+années|à\s+la\s+date)\b",
        r"(?:الساري|النافذ|الحالي|عبر\s+السنوات|اعتبارا\s+من)",
    )
)
_TEMPORAL_SUBJECT_CONTEXT = re.compile(
    r"\b(?:rules?|provisions?|articles?|annex(?:es)?|circulars?|regulations?|"
    r"règles?|dispositions?|annexes?|circulaires?|"
    r"documents?\s+réglementaires?|textes?\s+réglementaires?)\b|"
    r"(?:القاعدة|القواعد|الحكم|الأحكام|الفصل|المادة|المنشور|الوثيقة)",
    re.IGNORECASE,
)


class GraphRetrievalStatus(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    NO_SEED = "NO_SEED"
    NO_EVIDENCE = "NO_EVIDENCE"
    EXPANDED = "EXPANDED"
    UNAVAILABLE = "UNAVAILABLE"


class TemporalRetrievalStatus(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    DATE_AMBIGUOUS = "DATE_AMBIGUOUS"
    NO_CANDIDATE = "NO_CANDIDATE"
    AMBIGUOUS = "AMBIGUOUS"
    INCOMPLETE = "INCOMPLETE"
    RESOLVED = "RESOLVED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class GraphRetrievalTrace:
    status: GraphRetrievalStatus
    seed_filenames: tuple[str, ...] = ()
    evidence_count: int = 0
    paths: tuple[str, ...] = ()
    error_type: str | None = None
    temporal_status: TemporalRetrievalStatus = TemporalRetrievalStatus.NOT_REQUESTED
    as_of: date | None = None
    provision_uids: tuple[str, ...] = ()
    temporal_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "seed_filenames": list(self.seed_filenames),
            "evidence_count": self.evidence_count,
            "paths": list(self.paths),
            "error_type": self.error_type,
            "temporal_status": self.temporal_status.value,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "provision_uids": list(self.provision_uids),
            "temporal_reason": self.temporal_reason,
        }


@dataclass(frozen=True)
class GraphRetrievalResult:
    documents: tuple[Document, ...]
    trace: GraphRetrievalTrace

    @property
    def requires_temporal_abstention(self) -> bool:
        return self.trace.temporal_status not in {
            TemporalRetrievalStatus.NOT_REQUESTED,
            TemporalRetrievalStatus.RESOLVED,
        }


class RelationshipEvidenceGraph(Protocol):
    def relationship_evidence(
        self,
        seed_filenames: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[RelationshipEvidence, ...]: ...

    def affected_provisions(
        self,
        seeds: tuple[SourcePageSeed, ...],
        *,
        limit: int,
    ) -> tuple[AffectedProvision, ...]: ...

    def resolve_provision_as_of(
        self,
        provision_uid: str,
        as_of: date,
    ) -> TemporalResolution: ...

    def lineage(self, provision_uid: str) -> tuple[LineageEntry, ...]: ...


def is_relationship_query(query: str) -> bool:
    normalized = " ".join(query.casefold().split())
    return bool(_REGULATORY_DOCUMENT_CONTEXT.search(normalized)) and any(
        pattern.search(normalized) for pattern in _RELATIONSHIP_PATTERNS
    )


def _is_anaphoric_relationship_query(query: str) -> bool:
    normalized = " ".join(query.casefold().split())
    return any(
        pattern.search(normalized) for pattern in _ANAPHORIC_RELATIONSHIP_PATTERNS
    )


def is_temporal_rule_query(query: str) -> bool:
    normalized = " ".join(query.casefold().split())
    return bool(_TEMPORAL_SUBJECT_CONTEXT.search(normalized)) and any(
        pattern.search(normalized) for pattern in _TEMPORAL_RULE_PATTERNS
    )


def _temporal_as_of(query: str, *, current_date: date) -> date | None:
    iso_match = re.search(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", query)
    if iso_match:
        try:
            return date.fromisoformat(iso_match.group(1))
        except ValueError:
            return None
    local_match = re.search(
        r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)",
        query,
    )
    if local_match:
        day, month, year = (int(value) for value in local_match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return current_date


def _has_ambiguous_historical_date(query: str) -> bool:
    has_exact_date = bool(
        re.search(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", query)
        or re.search(r"(?<!\d)\d{1,2}/\d{1,2}/\d{4}(?!\d)", query)
    )
    if has_exact_date:
        return False
    return bool(
        re.search(
            r"\b(?:in|during|as\s+of|en|durant)\s+(?:the\s+year\s+)?\d{4}\b|"
            r"(?:عام|سنة)\s*\d{4}",
            query,
            re.IGNORECASE,
        )
    )


class RelationshipGraphRetriever:
    def __init__(
        self,
        graph: RelationshipEvidenceGraph,
        *,
        max_seeds: int = 5,
        max_evidence: int = 10,
        max_provisions: int = 10,
        current_date: date | None = None,
    ):
        if max_seeds < 1 or max_evidence < 1 or max_provisions < 1:
            raise ValueError("GraphRAG limits must be positive")
        self._graph = graph
        self._max_seeds = max_seeds
        self._max_evidence = max_evidence
        self._max_provisions = max_provisions
        self._current_date = current_date or date.today()

    def retrieve(
        self,
        query: str,
        seed_documents: Iterable[Document],
    ) -> GraphRetrievalResult:
        explicit_intent = is_relationship_query(query)
        temporal_intent = is_temporal_rule_query(query)
        temporal_as_of = _temporal_as_of(query, current_date=self._current_date)
        if temporal_intent and temporal_as_of is None:
            return _temporal_failure(
                GraphRetrievalStatus.NO_EVIDENCE,
                TemporalRetrievalStatus.DATE_AMBIGUOUS,
                as_of=None,
                reason="invalid_explicit_date",
            )
        if temporal_intent and temporal_as_of > self._current_date:
            return _temporal_failure(
                GraphRetrievalStatus.NO_EVIDENCE,
                TemporalRetrievalStatus.DATE_AMBIGUOUS,
                as_of=temporal_as_of,
                reason="future_as_of_not_supported",
            )
        if temporal_intent and _has_ambiguous_historical_date(query):
            return _temporal_failure(
                GraphRetrievalStatus.NO_EVIDENCE,
                TemporalRetrievalStatus.DATE_AMBIGUOUS,
                as_of=None,
                reason="exact_historical_date_required",
            )
        seed_locations = _seed_locations(seed_documents, limit=self._max_seeds)
        seed_filenames = tuple(seed.filename for seed in seed_locations)
        seeded_follow_up = bool(seed_filenames) and _is_anaphoric_relationship_query(
            query
        )
        if not explicit_intent and not seeded_follow_up and not temporal_intent:
            return _empty_result(GraphRetrievalStatus.NOT_REQUESTED)

        if not seed_filenames:
            if temporal_intent:
                return _temporal_failure(
                    GraphRetrievalStatus.NO_SEED,
                    TemporalRetrievalStatus.NO_CANDIDATE,
                    as_of=temporal_as_of,
                    reason="no_retrieval_seed",
                )
            return _empty_result(GraphRetrievalStatus.NO_SEED)

        evidence: tuple[RelationshipEvidence, ...] = ()
        if explicit_intent or seeded_follow_up:
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
                        temporal_status=(
                            TemporalRetrievalStatus.UNAVAILABLE
                            if temporal_intent
                            else TemporalRetrievalStatus.NOT_REQUESTED
                        ),
                        as_of=temporal_as_of if temporal_intent else None,
                        temporal_reason=(
                            "temporal_graph_unavailable"
                            if temporal_intent
                            else None
                        ),
                    ),
                )

        if temporal_intent:
            if temporal_as_of is None:
                raise AssertionError("validated temporal query requires an as-of date")
            return self._retrieve_temporal_context(
                query,
                seed_locations,
                as_of=temporal_as_of,
                relationship_evidence=evidence,
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

    def _retrieve_temporal_context(
        self,
        query: str,
        seed_locations: tuple[SourcePageSeed, ...],
        *,
        as_of: date,
        relationship_evidence: tuple[RelationshipEvidence, ...],
    ) -> GraphRetrievalResult:
        seed_filenames = tuple(seed.filename for seed in seed_locations)
        try:
            candidates = self._graph.affected_provisions(
                seed_locations,
                limit=self._max_provisions,
            )
            selected = _select_provision(query, candidates)
            if selected is None:
                status = (
                    TemporalRetrievalStatus.NO_CANDIDATE
                    if not candidates
                    else TemporalRetrievalStatus.AMBIGUOUS
                )
                return _temporal_failure(
                    GraphRetrievalStatus.NO_EVIDENCE,
                    status,
                    as_of=as_of,
                    seed_filenames=seed_filenames,
                    provision_uids=tuple(item.uid for item in candidates),
                    reason=(
                        "no_verified_affected_provision"
                        if not candidates
                        else "multiple_affected_provisions"
                    ),
                )
            resolution = self._graph.resolve_provision_as_of(
                selected.uid,
                as_of,
            )
            lineage = self._graph.lineage(selected.uid)
        except Exception as error:
            logger.warning(
                "Neo4j temporal provision resolution unavailable: %s",
                type(error).__name__,
            )
            return _temporal_failure(
                GraphRetrievalStatus.UNAVAILABLE,
                TemporalRetrievalStatus.UNAVAILABLE,
                as_of=as_of,
                seed_filenames=seed_filenames,
                error_type=type(error).__name__,
                reason="temporal_graph_error",
            )

        incomplete_reason = _incomplete_temporal_reason(
            resolution,
            lineage,
            as_of=as_of,
        )
        if incomplete_reason is not None:
            return _temporal_failure(
                GraphRetrievalStatus.NO_EVIDENCE,
                TemporalRetrievalStatus.INCOMPLETE,
                as_of=as_of,
                seed_filenames=seed_filenames,
                provision_uids=(selected.uid,),
                reason=incomplete_reason,
            )

        temporal_document = _temporal_document(
            selected,
            resolution,
            lineage,
        )
        relationship_documents = tuple(
            _evidence_document(item) for item in relationship_evidence
        )
        path = f"{selected.uid} -[AS_OF {as_of.isoformat()}]-> {resolution.version.uid}"
        return GraphRetrievalResult(
            documents=(temporal_document,) + relationship_documents,
            trace=GraphRetrievalTrace(
                status=GraphRetrievalStatus.EXPANDED,
                seed_filenames=seed_filenames,
                evidence_count=1 + len(relationship_documents),
                paths=(path,),
                temporal_status=TemporalRetrievalStatus.RESOLVED,
                as_of=as_of,
                provision_uids=(selected.uid,),
            ),
        )


@dataclass(frozen=True)
class LocalRelationshipGraphRuntime:
    retriever: RelationshipGraphRetriever
    _driver: Any

    def close(self) -> None:
        self._driver.close()


def open_relationship_graph_runtime(
    *,
    uri: str | None = None,
    database: str | None = None,
) -> LocalRelationshipGraphRuntime | None:
    resolved_uri = (
        uri
        if uri is not None
        else os.environ.get("BCT_NEO4J_URI", "bolt://127.0.0.1:17687")
    ).strip()
    if not resolved_uri:
        return None
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
        return None

    graph = Neo4jRegulatoryGraph(driver, database=resolved_database)
    return LocalRelationshipGraphRuntime(
        retriever=RelationshipGraphRetriever(graph),
        _driver=driver,
    )


def _seed_locations(
    documents: Iterable[Document],
    *,
    limit: int,
) -> tuple[SourcePageSeed, ...]:
    pages_by_filename: dict[str, set[int]] = {}
    for document in documents:
        source = document.metadata.get("source")
        filename = Path(str(source)).name if source else ""
        if not filename:
            continue
        if filename not in pages_by_filename and len(pages_by_filename) == limit:
            break
        page_label = document.metadata.get("page_label")
        page = document.metadata.get("page")
        try:
            page_number = (
                int(page_label)
                if page_label is not None
                else int(page) + 1
            )
        except (TypeError, ValueError):
            page_number = None
        pages = pages_by_filename.setdefault(filename, set())
        if page_number is not None and page_number >= 1:
            pages.add(page_number)
    return tuple(
        SourcePageSeed(
            filename=filename,
            page_numbers=tuple(sorted(page_numbers)),
        )
        for filename, page_numbers in pages_by_filename.items()
    )


def _select_provision(
    query: str,
    candidates: tuple[AffectedProvision, ...],
) -> AffectedProvision | None:
    if len(candidates) == 1:
        return candidates[0]
    normalized = " ".join(query.casefold().split())
    named = [
        candidate
        for candidate in candidates
        if re.search(
            rf"(?<!\w){re.escape(candidate.label.casefold())}(?!\w)",
            normalized,
        )
    ]
    return named[0] if len(named) == 1 else None


def _incomplete_temporal_reason(
    resolution: TemporalResolution,
    lineage: tuple[LineageEntry, ...],
    *,
    as_of: date,
) -> str | None:
    if resolution.version is None:
        return "no_verified_version_as_of_date"
    applicable = tuple(
        entry
        for entry in lineage
        if entry.effective_from is None or entry.effective_from <= as_of
    )
    if not applicable:
        return "no_applicable_verified_lineage"
    if any(not entry.evidence for entry in applicable):
        return "lineage_evidence_missing"
    if any(entry.effective_from is None for entry in applicable):
        return "effective_date_unresolved"
    for entry in applicable:
        if not entry.introduced_version_uids:
            return "introduced_version_missing"
        if entry.action == LegalAction.REPLACE and (
            not entry.predecessor_complete or not entry.retired_version_uids
        ):
            return "replacement_predecessor_incomplete"
    latest = max(
        applicable,
        key=lambda entry: (entry.effective_from, entry.event_uid),
    )
    if resolution.version.uid not in latest.introduced_version_uids:
        return "resolved_version_not_latest_introduced_version"
    return None


def _temporal_document(
    provision: AffectedProvision,
    resolution: TemporalResolution,
    lineage: tuple[LineageEntry, ...],
) -> Document:
    version = resolution.version
    if version is None:
        raise ValueError("temporal document requires a resolved version")
    applicable = tuple(
        entry
        for entry in lineage
        if entry.effective_from is not None and entry.effective_from <= resolution.as_of
    )
    latest = max(
        applicable,
        key=lambda entry: (entry.effective_from, entry.event_uid),
    )
    primary_evidence = latest.evidence[0]
    lineage_text = "\n".join(
        f"- {entry.effective_from.isoformat()} {entry.action.value} "
        f"({entry.event_uid})"
        for entry in applicable
    )
    evidence_text = "\n".join(
        f"- {item.filename}, page {item.page_number}: {item.quote}"
        for entry in applicable
        for item in entry.evidence
    )
    content = (
        "Verified temporal provision resolution.\n"
        f"Provision: {provision.label} ({provision.uid})\n"
        f"Applicable as of: {resolution.as_of.isoformat()}\n"
        f"Verified version: {version.uid}\n"
        f"Valid interval: {version.valid_from} to {version.valid_to or 'open'}\n"
        f"Controlling text:\n{version.text}\n"
        f"Verified lineage:\n{lineage_text}\n"
        f"Source evidence:\n{evidence_text}"
    )
    return Document(
        page_content=content,
        metadata={
            "source": primary_evidence.filename,
            "page": primary_evidence.page_number - 1,
            "page_label": primary_evidence.page_number,
            "retrieval_source": "neo4j_temporal_provision",
            "temporal_resolution": "VERIFIED",
            "as_of": resolution.as_of.isoformat(),
            "provision_uid": provision.uid,
            "provision_label": provision.label,
            "provision_version_uid": version.uid,
            "valid_from": version.valid_from.isoformat() if version.valid_from else None,
            "valid_to": version.valid_to.isoformat() if version.valid_to else None,
            "lineage_event_uids": ",".join(entry.event_uid for entry in applicable),
        },
    )


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
            "graph_relation": evidence.relation_label,
        },
    )


def _empty_result(status: GraphRetrievalStatus) -> GraphRetrievalResult:
    return GraphRetrievalResult(
        documents=(),
        trace=GraphRetrievalTrace(status=status),
    )


def _temporal_failure(
    graph_status: GraphRetrievalStatus,
    temporal_status: TemporalRetrievalStatus,
    *,
    as_of: date | None,
    reason: str,
    seed_filenames: tuple[str, ...] = (),
    provision_uids: tuple[str, ...] = (),
    error_type: str | None = None,
) -> GraphRetrievalResult:
    return GraphRetrievalResult(
        documents=(),
        trace=GraphRetrievalTrace(
            status=graph_status,
            seed_filenames=seed_filenames,
            error_type=error_type,
            temporal_status=temporal_status,
            as_of=as_of,
            provision_uids=provision_uids,
            temporal_reason=reason,
        ),
    )
