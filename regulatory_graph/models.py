from datetime import date
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-fA-F]{64}$")]
Language = Literal["fr", "ar", "unknown"]


class InstrumentKind(str, Enum):
    CIRCULAR = "CIRCULAR"
    NOTE = "NOTE"
    LAW = "LAW"
    DECREE = "DECREE"
    CODE = "CODE"
    OPINION = "OPINION"
    DECISION = "DECISION"
    OTHER = "OTHER"


class SourceStatus(str, Enum):
    LOCAL = "LOCAL"
    EXTERNAL_STUB = "EXTERNAL_STUB"


class ProvisionType(str, Enum):
    TITLE = "TITLE"
    PART = "PART"
    CHAPTER = "CHAPTER"
    SECTION = "SECTION"
    ARTICLE = "ARTICLE"
    PARAGRAPH = "PARAGRAPH"
    ALINEA = "ALINEA"
    ITEM = "ITEM"
    BULLET = "BULLET"
    ANNEX = "ANNEX"
    TABLE = "TABLE"
    FORM = "FORM"
    OTHER = "OTHER"


class VersionStatus(str, Enum):
    FUTURE = "FUTURE"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    ABROGATED = "ABROGATED"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"


class LegalAction(str, Enum):
    REPLACE = "REPLACE"
    ABROGATE = "ABROGATE"
    MODIFY = "MODIFY"
    ADD = "ADD"
    SUPPLEMENT = "SUPPLEMENT"
    EXTEND = "EXTEND"
    REMOVE = "REMOVE"
    DEROGATE = "DEROGATE"
    SUSPEND = "SUSPEND"


class TargetScope(str, Enum):
    INSTRUMENT = "INSTRUMENT"
    TITLE = "TITLE"
    CHAPTER = "CHAPTER"
    SECTION = "SECTION"
    ARTICLE = "ARTICLE"
    PARAGRAPH = "PARAGRAPH"
    ITEM = "ITEM"
    ANNEX = "ANNEX"
    TABLE = "TABLE"
    TABLE_ROW = "TABLE_ROW"
    TABLE_CELL = "TABLE_CELL"
    PHRASE = "PHRASE"
    DATE = "DATE"
    AMOUNT = "AMOUNT"
    RATE = "RATE"
    OTHER_FRAGMENT = "OTHER_FRAGMENT"


class TargetSelectorType(str, Enum):
    PARAGRAPH = "PARAGRAPH"
    BULLET = "BULLET"
    LETTERED_ITEM = "LETTERED_ITEM"
    SENTENCE = "SENTENCE"
    PHRASE = "PHRASE"
    DATE = "DATE"
    AMOUNT = "AMOUNT"
    RATE = "RATE"
    TABLE_ROW = "TABLE_ROW"
    TABLE_CELL = "TABLE_CELL"


class VerificationStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class GraphModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Instrument(GraphModel):
    uid: NonEmptyStr
    authority: NonEmptyStr
    kind: InstrumentKind
    year: int | None = Field(default=None, ge=0)
    number: NonEmptyStr | None = None
    issue_date: date | None = None
    title: NonEmptyStr | None = None
    subject: NonEmptyStr | None = None
    corpus_present: bool
    canonical_citation: NonEmptyStr | None = None
    source_status: SourceStatus

    @model_validator(mode="after")
    def validate_corpus_status(self) -> "Instrument":
        expected = SourceStatus.LOCAL if self.corpus_present else SourceStatus.EXTERNAL_STUB
        if self.source_status != expected:
            raise ValueError(
                "source_status must be LOCAL for corpus instruments and "
                "EXTERNAL_STUB when corpus_present is false"
            )
        return self


class SourceEdition(GraphModel):
    uid: NonEmptyStr
    instrument_uid: NonEmptyStr
    language: Language
    filename: NonEmptyStr
    sha256: Sha256
    source_url: NonEmptyStr | None = None
    extraction_status: NonEmptyStr
    page_count: int = Field(ge=1)
    is_scan: bool


class GraphPage(GraphModel):
    uid: NonEmptyStr
    source_edition_uid: NonEmptyStr
    page_number: int = Field(ge=1)
    page_label: NonEmptyStr


class Provision(GraphModel):
    uid: NonEmptyStr
    instrument_uid: NonEmptyStr
    provision_type: ProvisionType
    label: NonEmptyStr
    ordinal: int | None = Field(default=None, ge=0)
    canonical_path: NonEmptyStr
    heading: NonEmptyStr | None = None


class ProvisionVersion(GraphModel):
    uid: NonEmptyStr
    provision_uid: NonEmptyStr
    version_number: int = Field(ge=1)
    text: NonEmptyStr
    normalized_text: str | None = None
    language: Language
    valid_from: date | None = None
    valid_to: date | None = None
    effective_trigger: NonEmptyStr | None = None
    status: VersionStatus
    content_hash: Sha256
    confidence: float | None = Field(default=None, ge=0, le=1)
    verification_status: VerificationStatus | None = None

    @model_validator(mode="after")
    def validate_half_open_interval(self) -> "ProvisionVersion":
        _validate_half_open_interval(self.valid_from, self.valid_to, "valid")
        if self.status == VersionStatus.ACTIVE:
            if self.verification_status != VerificationStatus.VERIFIED:
                raise ValueError("ACTIVE provision version must be VERIFIED")
        return self


class TargetSpan(GraphModel):
    uid: NonEmptyStr
    provision_version_uid: NonEmptyStr
    selector_type: TargetSelectorType
    raw_selector: NonEmptyStr
    old_text: str | None = None
    new_text: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "TargetSpan":
        _validate_offsets(self.char_start, self.char_end)
        return self


class EvidenceSpan(GraphModel):
    uid: NonEmptyStr
    source_edition_uid: NonEmptyStr
    quote: NonEmptyStr
    page_number: int = Field(ge=1)
    extraction_method: NonEmptyStr
    source_sha256: Sha256
    extraction_artifact_hash: Sha256
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    bounding_box: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceSpan":
        _validate_offsets(self.char_start, self.char_end)
        return self


class ChangeEvent(GraphModel):
    uid: NonEmptyStr
    source_instrument_uid: NonEmptyStr
    action: LegalAction
    target_scope: TargetScope
    target_instrument_uids: tuple[NonEmptyStr, ...] = ()
    target_provision_uids: tuple[NonEmptyStr, ...] = ()
    target_span_uids: tuple[NonEmptyStr, ...] = ()
    effective_from: date | None = None
    effective_to: date | None = None
    effective_trigger: NonEmptyStr | None = None
    effective_trigger_resolved: bool = False
    transition_end: date | None = None
    raw_effect_text: NonEmptyStr
    evidence_uids: tuple[NonEmptyStr, ...] = ()
    retires_version_uids: tuple[NonEmptyStr, ...] = ()
    introduces_version_uids: tuple[NonEmptyStr, ...] = ()
    confidence: float = Field(ge=0, le=1)
    extraction_method: NonEmptyStr = "structured"
    verification_status: VerificationStatus
    validator_reason: str | None = None

    @model_validator(mode="after")
    def validate_legal_effect_contract(self) -> "ChangeEvent":
        targets = (
            self.target_instrument_uids,
            self.target_provision_uids,
            self.target_span_uids,
        )
        if not any(targets):
            raise ValueError("change event requires at least one target")
        if self.target_scope == TargetScope.INSTRUMENT:
            if not self.target_instrument_uids or any(targets[1:]):
                raise ValueError("INSTRUMENT scope requires only Instrument targets")
        elif self.target_scope in _SPAN_REQUIRED_SCOPES:
            if not self.target_span_uids or self.target_instrument_uids:
                raise ValueError(
                    f"{self.target_scope.value} scope requires a TargetSpan target"
                )
        elif self.target_scope in _PROVISION_REQUIRED_SCOPES:
            if not self.target_provision_uids or self.target_instrument_uids:
                raise ValueError(
                    f"{self.target_scope.value} scope requires a Provision target"
                )
        elif self.target_scope in _PROVISION_OR_SPAN_SCOPES:
            if self.target_instrument_uids or not (
                self.target_provision_uids or self.target_span_uids
            ):
                raise ValueError(
                    f"{self.target_scope.value} scope requires a Provision or TargetSpan"
                )
        elif self.target_instrument_uids:
            raise ValueError(
                f"{self.target_scope.value} scope cannot target an Instrument"
            )

        _validate_half_open_interval(self.effective_from, self.effective_to, "effective")
        if self.effective_trigger_resolved:
            if self.effective_trigger is None or self.effective_from is None:
                raise ValueError(
                    "resolved effective trigger requires trigger text and effective_from"
                )
        if self.verification_status == VerificationStatus.VERIFIED:
            if not self.evidence_uids:
                raise ValueError("verified change event requires evidence")
            if self.effective_from is None and self.effective_trigger is None:
                raise ValueError(
                    "verified change event requires an effective date or unresolved trigger"
                )
        return self

    @property
    def temporal_state_ready(self) -> bool:
        return (
            self.verification_status == VerificationStatus.VERIFIED
            and bool(self.evidence_uids)
            and self.effective_from is not None
            and (self.effective_trigger is None or self.effective_trigger_resolved)
        )


class RegulatoryGraphBundle(GraphModel):
    instruments: tuple[Instrument, ...]
    source_editions: tuple[SourceEdition, ...]
    pages: tuple[GraphPage, ...]
    provisions: tuple[Provision, ...]
    provision_versions: tuple[ProvisionVersion, ...]
    target_spans: tuple[TargetSpan, ...] = ()
    evidence_spans: tuple[EvidenceSpan, ...]
    change_events: tuple[ChangeEvent, ...]

    @model_validator(mode="after")
    def validate_bundle_references(self) -> "RegulatoryGraphBundle":
        collections = {
            "Instrument": self.instruments,
            "SourceEdition": self.source_editions,
            "Page": self.pages,
            "Provision": self.provisions,
            "ProvisionVersion": self.provision_versions,
            "TargetSpan": self.target_spans,
            "EvidenceSpan": self.evidence_spans,
            "ChangeEvent": self.change_events,
        }
        ids = {label: {item.uid for item in items} for label, items in collections.items()}
        for label, items in collections.items():
            if len(ids[label]) != len(items):
                raise ValueError(f"duplicate {label} uid in graph bundle")

        required = []
        required.extend(
            ("Instrument", item.instrument_uid) for item in self.source_editions
        )
        required.extend(
            ("SourceEdition", item.source_edition_uid) for item in self.pages
        )
        required.extend(("Instrument", item.instrument_uid) for item in self.provisions)
        required.extend(
            ("Provision", item.provision_uid) for item in self.provision_versions
        )
        required.extend(
            ("ProvisionVersion", item.provision_version_uid)
            for item in self.target_spans
        )
        required.extend(
            ("SourceEdition", item.source_edition_uid)
            for item in self.evidence_spans
        )
        missing = [f"{label} {uid}" for label, uid in required if uid not in ids[label]]
        page_keys = {
            (page.source_edition_uid, page.page_number) for page in self.pages
        }
        missing.extend(
            f"Page {evidence.source_edition_uid}:{evidence.page_number}"
            for evidence in self.evidence_spans
            if (evidence.source_edition_uid, evidence.page_number) not in page_keys
        )
        if missing:
            raise ValueError("graph bundle references missing nodes: " + ", ".join(missing))
        return self

    @property
    def node_count(self) -> int:
        return sum(
            len(items)
            for items in (
                self.instruments,
                self.source_editions,
                self.pages,
                self.provisions,
                self.provision_versions,
                self.target_spans,
                self.evidence_spans,
                self.change_events,
            )
        )


def _validate_offsets(char_start: int | None, char_end: int | None) -> None:
    if (char_start is None) != (char_end is None):
        raise ValueError("char_start and char_end must be provided together")
    if char_start is not None and char_end is not None and char_end <= char_start:
        raise ValueError("char_end must be later than char_start")


def _validate_half_open_interval(
    interval_start: date | None,
    interval_end: date | None,
    field_prefix: str,
) -> None:
    if interval_start is not None and interval_end is not None:
        if interval_end <= interval_start:
            raise ValueError(
                f"{field_prefix}_to must be later than {field_prefix}_from"
            )


_SPAN_REQUIRED_SCOPES = frozenset(
    {
        TargetScope.TABLE_ROW,
        TargetScope.TABLE_CELL,
        TargetScope.PHRASE,
        TargetScope.DATE,
        TargetScope.AMOUNT,
        TargetScope.RATE,
        TargetScope.OTHER_FRAGMENT,
    }
)


_PROVISION_REQUIRED_SCOPES = frozenset(
    {
        TargetScope.TITLE,
        TargetScope.CHAPTER,
        TargetScope.SECTION,
        TargetScope.ARTICLE,
        TargetScope.ANNEX,
        TargetScope.TABLE,
    }
)


_PROVISION_OR_SPAN_SCOPES = frozenset(
    {
        TargetScope.PARAGRAPH,
        TargetScope.ITEM,
    }
)
