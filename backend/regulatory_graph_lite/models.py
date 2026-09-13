from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class RelationshipType(str, Enum):
    CITES = "CITES"
    AMENDS = "AMENDS"
    REPLACES = "REPLACES"
    ABROGATES = "ABROGATES"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"


@dataclass(frozen=True)
class InstrumentRef:
    id: str
    kind: str
    year: int
    number: int
    filename: str | None = None


@dataclass(frozen=True)
class RelationshipCandidate:
    candidate_id: str
    relation: RelationshipType
    source: InstrumentRef
    target: InstrumentRef
    evidence_file: str
    evidence_page: int
    evidence_quote: str
    source_provision: str | None = None
    target_provision: str | None = None
    proposed_effective_date: str | None = None
    verification_status: VerificationStatus = VerificationStatus.VERIFIED
    target_in_catalog: bool = False
    verification_method: str = "AUTO_DETERMINISTIC_V1"

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["relation"] = self.relation.value
        value["verification_status"] = self.verification_status.value
        return value
