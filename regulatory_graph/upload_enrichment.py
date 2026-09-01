from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from regulatory_graph.models import RegulatoryGraphBundle, VerificationStatus
from regulatory_graph.neo4j_store import WriteReceipt
from regulatory_graph.reference_ingestion import (
    InstrumentReferenceCandidate,
    ReferencePage,
    ReferencePromotionDecision,
    ReferencePromotionEvidence,
    VerifiedInstrumentCatalog,
    VerifiedReferencePromotion,
    enrich_bundle_with_verified_references,
    extract_document_reference_candidates,
    promote_reference_candidate,
)
from regulatory_graph.semantic_candidates import (
    CandidateType,
    SemanticCandidate,
    extract_page_candidates,
)


class GraphBundleWriter(Protocol):
    def write_bundle(self, bundle: RegulatoryGraphBundle) -> WriteReceipt: ...


@dataclass(frozen=True)
class UploadedPdfGraphReceipt:
    source_edition_uid: str
    reference_candidate_count: int
    verified_reference_count: int
    needs_review_reference_count: int
    legal_action_candidate_count: int
    write_receipt: WriteReceipt


@dataclass(frozen=True)
class UploadedPdfGraphResult:
    receipt: UploadedPdfGraphReceipt
    reference_candidates: tuple[InstrumentReferenceCandidate, ...]
    semantic_candidates: tuple[SemanticCandidate, ...]
    reference_decisions: tuple[ReferencePromotionDecision, ...]


def process_uploaded_pdf_graph(
    bundle: RegulatoryGraphBundle,
    pages: tuple[ReferencePage, ...],
    *,
    source_pdf_path: str | Path,
    extraction_artifact_hash: str,
    instrument_catalog: VerifiedInstrumentCatalog,
    graph_writer: GraphBundleWriter,
    reference_reviews: Mapping[str, ReferencePromotionEvidence] | None = None,
) -> UploadedPdfGraphResult:
    if len(bundle.source_editions) != 1:
        raise ValueError("uploaded PDF graph processing requires one source edition")
    edition = bundle.source_editions[0]
    if bundle.instrument_references:
        raise ValueError(
            "uploaded PDF graph bundle must not contain pre-existing "
            "instrument references"
        )
    if any(
        event.verification_status != VerificationStatus.VERIFIED
        for event in bundle.change_events
    ):
        raise ValueError("uploaded PDF graph bundle contains unverified change events")
    if (
        edition.extraction_artifact_hash is None
        or extraction_artifact_hash.casefold()
        != edition.extraction_artifact_hash.casefold()
    ):
        raise ValueError(
            "extracted pages do not match the source edition extraction artifact hash"
        )
    try:
        source_bytes = Path(source_pdf_path).read_bytes()
    except OSError as error:
        raise ValueError("uploaded source PDF is unavailable") from error
    if sha256(source_bytes).hexdigest().casefold() != edition.sha256.casefold():
        raise ValueError("uploaded source PDF hash does not match its source edition")

    ordered_pages = tuple(sorted(pages, key=lambda item: item.page_number))
    reference_candidates = extract_document_reference_candidates(
        edition,
        ordered_pages,
        instrument_catalog=instrument_catalog,
    )
    semantic_candidates = tuple(
        sorted(
            (
                candidate
                for page in ordered_pages
                for candidate in extract_page_candidates(
                    page.text,
                    filename=edition.filename,
                    source_edition_uid=edition.uid,
                    instrument_uid=edition.instrument_uid,
                    language=edition.language,
                    page_number=page.page_number,
                    source_sha256=edition.sha256,
                    extraction_artifact_hash=edition.extraction_artifact_hash,
                )
            ),
            key=lambda item: item.uid,
        )
    )

    reviews = reference_reviews or {}
    candidates_by_uid = {
        candidate.uid: candidate for candidate in reference_candidates
    }
    unknown_review_uids = sorted(set(reviews) - set(candidates_by_uid))
    if unknown_review_uids:
        raise ValueError(
            "reference reviews do not belong to the uploaded PDF: "
            f"{unknown_review_uids}"
        )
    decisions = tuple(
        promote_reference_candidate(
            candidates_by_uid[candidate_uid],
            review,
            source_pdf_path=source_pdf_path,
            instrument_catalog=instrument_catalog,
        )
        for candidate_uid, review in sorted(reviews.items())
    )
    verified_decisions = tuple(
        decision
        for decision in decisions
        if isinstance(decision, VerifiedReferencePromotion)
    )
    enriched_bundle = enrich_bundle_with_verified_references(
        bundle,
        verified_decisions,
    )
    write_receipt = graph_writer.write_bundle(enriched_bundle)
    legal_action_count = sum(
        candidate.candidate_type == CandidateType.LEGAL_ACTION
        for candidate in semantic_candidates
    )
    return UploadedPdfGraphResult(
        receipt=UploadedPdfGraphReceipt(
            source_edition_uid=edition.uid,
            reference_candidate_count=len(reference_candidates),
            verified_reference_count=len(verified_decisions),
            needs_review_reference_count=(
                len(reference_candidates) - len(verified_decisions)
            ),
            legal_action_candidate_count=legal_action_count,
            write_receipt=write_receipt,
        ),
        reference_candidates=reference_candidates,
        semantic_candidates=semantic_candidates,
        reference_decisions=decisions,
    )
