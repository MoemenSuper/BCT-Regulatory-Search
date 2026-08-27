from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from regulatory_graph.models import RegulatoryGraphBundle, VerificationStatus


class SourceVerificationError(ValueError):
    """Raised when a graph fixture is not exactly supported by its source PDF."""


@dataclass(frozen=True)
class SourceVerificationReceipt:
    source_sha256_verified: bool
    page_count_verified: bool
    exact_evidence_count: int
    exact_verified_version_count: int


def verify_bundle_source(
    bundle: RegulatoryGraphBundle,
    pdf_path: str | Path,
) -> SourceVerificationReceipt:
    import fitz

    path = Path(pdf_path)
    source_bytes = path.read_bytes()
    source_hash = sha256(source_bytes).hexdigest()
    matching_editions = [
        edition
        for edition in bundle.source_editions
        if edition.sha256.lower() == source_hash.lower()
    ]
    if len(matching_editions) != 1:
        raise SourceVerificationError(
            "source PDF hash must match exactly one graph source edition"
        )
    edition = matching_editions[0]

    with fitz.open(path) as document:
        if len(document) != edition.page_count:
            raise SourceVerificationError(
                f"source page count mismatch: expected {edition.page_count}, got {len(document)}"
            )
        page_text = {
            page_number: _normalize(document[page_number - 1].get_text())
            for page_number in {item.page_number for item in bundle.evidence_spans}
        }

    exact_evidence = []
    for evidence in bundle.evidence_spans:
        if evidence.source_edition_uid != edition.uid:
            continue
        if _normalize(evidence.quote) not in page_text[evidence.page_number]:
            raise SourceVerificationError(
                f"evidence quote is not exact source text: {evidence.uid}"
            )
        exact_evidence.append(evidence.uid)

    evidence_by_uid = {item.uid: item for item in bundle.evidence_spans}
    introducing_events = {
        version_uid: event
        for event in bundle.change_events
        for version_uid in event.introduces_version_uids
    }
    exact_versions = []
    for version in bundle.provision_versions:
        if version.verification_status != VerificationStatus.VERIFIED:
            continue
        event = introducing_events.get(version.uid)
        if event is None:
            raise SourceVerificationError(
                f"verified version lacks an introducing change event: {version.uid}"
            )
        evidence_pages = {
            evidence_by_uid[uid].page_number
            for uid in event.evidence_uids
            if evidence_by_uid[uid].source_edition_uid == edition.uid
        }
        if not any(
            _normalize(version.text) in page_text[page_number]
            for page_number in evidence_pages
        ):
            raise SourceVerificationError(
                f"verified version text is not exact source text: {version.uid}"
            )
        exact_versions.append(version.uid)

    return SourceVerificationReceipt(
        source_sha256_verified=True,
        page_count_verified=True,
        exact_evidence_count=len(exact_evidence),
        exact_verified_version_count=len(exact_versions),
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
