from typing import Protocol

from regulatory_graph.models import (
    ChangeEvent,
    InstrumentReference,
    VerificationStatus,
)


class ReferenceCatalog(Protocol):
    def exists(self, label: str, uid: str) -> bool: ...


class GraphReferenceError(ValueError):
    """Raised when a graph write would reference a missing node."""


class GraphVerificationError(ValueError):
    """Raised when an unverified semantic fact reaches the graph writer."""


def validate_change_event_for_write(
    event: ChangeEvent,
    references: ReferenceCatalog,
) -> ChangeEvent:
    required_references = [
        ("Instrument", event.source_instrument_uid),
        *(("Instrument", uid) for uid in event.target_instrument_uids),
        *(("Provision", uid) for uid in event.target_provision_uids),
        *(("TargetSpan", uid) for uid in event.target_span_uids),
        *(("EvidenceSpan", uid) for uid in event.evidence_uids),
        *(("ProvisionVersion", uid) for uid in event.retires_version_uids),
        *(("ProvisionVersion", uid) for uid in event.introduces_version_uids),
    ]
    missing = [
        f"{label} {uid}"
        for label, uid in required_references
        if not references.exists(label, uid)
    ]
    if missing:
        raise GraphReferenceError(
            "change event references missing graph nodes: " + ", ".join(missing)
        )
    return event


def validate_instrument_reference_for_write(
    reference: InstrumentReference,
    references: ReferenceCatalog,
) -> InstrumentReference:
    if reference.verification_status != VerificationStatus.VERIFIED:
        raise GraphVerificationError(
            "instrument reference must be VERIFIED before graph write"
        )
    required_references = (
        ("source Instrument", "Instrument", reference.source_instrument_uid),
        ("target Instrument", "Instrument", reference.target_instrument_uid),
        ("EvidenceSpan", "EvidenceSpan", reference.evidence_uid),
    )
    missing = [
        f"{description} {uid}"
        for description, label, uid in required_references
        if not references.exists(label, uid)
    ]
    if missing:
        raise GraphReferenceError(
            "instrument reference references missing graph nodes: "
            + ", ".join(missing)
        )
    return reference
