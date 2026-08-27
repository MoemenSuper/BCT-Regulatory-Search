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
        if self.valid_from is not None and self.valid_to is not None:
            if self.valid_to <= self.valid_from:
                raise ValueError("valid_to must be later than valid_from")
        return self


class TargetSpan(GraphModel):
    uid: NonEmptyStr
    provision_version_uid: NonEmptyStr
    selector_type: TargetScope
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
    target_provision_uids: tuple[NonEmptyStr, ...] = ()
    target_span_uids: tuple[NonEmptyStr, ...] = ()
    effective_from: date | None = None
    effective_to: date | None = None
    effective_trigger: NonEmptyStr | None = None
    transition_end: date | None = None
    raw_effect_text: NonEmptyStr
    evidence_uids: tuple[NonEmptyStr, ...] = ()
    confidence: float = Field(ge=0, le=1)
    extraction_method: NonEmptyStr = "structured"
    verification_status: VerificationStatus
    validator_reason: str | None = None

    @model_validator(mode="after")
    def validate_legal_effect_contract(self) -> "ChangeEvent":
        if not self.target_provision_uids and not self.target_span_uids:
            raise ValueError("change event requires at least one target")
        if self.effective_from is not None and self.effective_to is not None:
            if self.effective_to <= self.effective_from:
                raise ValueError("effective_to must be later than effective_from")
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
        )


def _validate_offsets(char_start: int | None, char_end: int | None) -> None:
    if (char_start is None) != (char_end is None):
        raise ValueError("char_start and char_end must be provided together")
    if char_start is not None and char_end is not None and char_end <= char_start:
        raise ValueError("char_end must be later than char_start")

