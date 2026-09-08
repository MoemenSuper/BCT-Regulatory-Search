"""Lightweight query classification and result types for the optional graph adapter."""
from dataclasses import dataclass
from datetime import date
from enum import Enum
import re
from langchain_core.documents import Document


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
        r"(?:was|were)\s+applicable|applied\s+(?:on|at)|"
        r"will\s+(?:apply|be\s+applicable)|"
        r"changed\s+across\s+years|over\s+the\s+years|as\s+of)\b",
        r"\b(?:actuel(?:le)?|en\s+vigueur|règle\s+applicable|"
        r"(?:était|étaient)\s+applicable|"
        r"(?:sera|seront)\s+applicable|"
        r"au\s+fil\s+des\s+années|à\s+la\s+date)\b",
        r"(?:الساري|النافذ|الحالي|المطبق|سيطبق|عبر\s+السنوات|اعتبارا\s+من)",
    )
)
_TEMPORAL_SUBJECT_CONTEXT = re.compile(
    r"\b(?:rules?|provisions?|articles?|annex(?:es)?|circulars?|regulations?|"
    r"règles?|dispositions?|annexes?|circulaires?|"
    r"documents?\s+réglementaires?|textes?\s+réglementaires?)\b|"
    r"(?:القاعدة|القواعد|الحكم|الأحكام|الفصل|المادة|المنشور|الوثيقة)",
    re.IGNORECASE,
)
_EXPLICIT_CURRENT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:current|currently|latest|today)\b",
        r"\bin\s+force\b",
        r"\b(?:actuel(?:le(?:ment)?)?s?|aujourd['’]hui)\b",
        r"\ben\s+vigueur\b",
        r"\bplus\s+récente?s?\b",
        r"\b(?:dernier|dernière)s?\s+(?:taux|plafond|règle|valeur|version|disposition)s?\b",
        r"(?:الساري|النافذ|الحالي|سارية|اليوم|الأحدث|أحدث|آخر\s+(?:قيمة|نسبة|سقف|قاعدة))",
    )
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


class TemporalFailureReason(str, Enum):
    INVALID_EXPLICIT_DATE = "invalid_explicit_date"
    EXACT_HISTORICAL_DATE_REQUIRED = "exact_historical_date_required"
    NO_RETRIEVAL_SEED = "no_retrieval_seed"
    TEMPORAL_GRAPH_UNAVAILABLE = "temporal_graph_unavailable"
    NO_VERIFIED_AFFECTED_PROVISION = "no_verified_affected_provision"
    MULTIPLE_AFFECTED_PROVISIONS = "multiple_affected_provisions"
    TEMPORAL_GRAPH_ERROR = "temporal_graph_error"
    NO_VERIFIED_VERSION_AS_OF_DATE = "no_verified_version_as_of_date"
    NO_APPLICABLE_VERIFIED_LINEAGE = "no_applicable_verified_lineage"
    LINEAGE_EVIDENCE_MISSING = "lineage_evidence_missing"
    EFFECTIVE_DATE_UNRESOLVED = "effective_date_unresolved"
    INTRODUCED_VERSION_MISSING = "introduced_version_missing"
    REPLACEMENT_PREDECESSOR_INCOMPLETE = "replacement_predecessor_incomplete"
    VERSION_PREDECESSOR_INCOMPLETE = "version_predecessor_incomplete"
    SAME_DATE_LINEAGE_ORDER_AMBIGUOUS = "same_date_lineage_order_ambiguous"
    RESOLVED_VERSION_NOT_LATEST = "resolved_version_not_latest_introduced_version"
    RELATIONSHIP_ONLY_NOT_PROVISION_RESOLVED = "relationship_only_not_provision_resolved"


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
    temporal_reason: TemporalFailureReason | None = None

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
            "temporal_reason": (
                self.temporal_reason.value if self.temporal_reason else None
            ),
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
    if any(pattern.search(normalized) for pattern in _EXPLICIT_CURRENT_PATTERNS):
        return True
    if re.search(r"\b(?:au|as\s+of)\s+\d{1,2}(?:er)?\s+\w+\s+\d{4}\b", normalized):
        return True
    return bool(_TEMPORAL_SUBJECT_CONTEXT.search(normalized)) and any(
        pattern.search(normalized) for pattern in _TEMPORAL_RULE_PATTERNS
    )
