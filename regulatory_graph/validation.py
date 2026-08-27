from typing import Protocol

from regulatory_graph.models import ChangeEvent


class ReferenceCatalog(Protocol):
    def exists(self, label: str, uid: str) -> bool: ...


class GraphReferenceError(ValueError):
    """Raised when a graph write would reference a missing node."""


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
