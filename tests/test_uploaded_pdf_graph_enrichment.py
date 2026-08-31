from hashlib import sha256

import fitz
import pytest

from regulatory_graph.fixtures import circular_2016_03_fr_bundle
from regulatory_graph.models import VerificationStatus
from regulatory_graph.neo4j_store import WriteReceipt
from regulatory_graph.reference_ingestion import ReferencePage, VerifiedInstrumentCatalog
from regulatory_graph.reference_ingestion import ReferencePromotionEvidence
from regulatory_graph.semantic_candidates import CandidateType
from regulatory_graph.upload_enrichment import process_uploaded_pdf_graph


class CapturingGraphWriter:
    def __init__(self):
        self.bundles = []

    def write_bundle(self, bundle):
        self.bundles.append(bundle)
        return WriteReceipt(
            bundle_sha256="c" * 64,
            change_event_count=len(bundle.change_events),
        )


def _write_pdf(path, page_texts):
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()
    return sha256(path.read_bytes()).hexdigest().upper()


def _uploaded_bundle(tmp_path):
    pdf_path = tmp_path / "Cir_2016_03_fr.pdf"
    relationship_text = (
        "Article 2 : Les dispositions de l'article 4 de la circulaire "
        "n 91-24 sont abrogees et remplacees."
    )
    source_sha256 = _write_pdf(
        pdf_path,
        ("Page 1", relationship_text, "Page 3", "Page 4"),
    )
    bundle = circular_2016_03_fr_bundle()
    edition = bundle.source_editions[0].model_copy(
        update={
            "sha256": source_sha256,
            "extraction_artifact_hash": "b" * 64,
            "identity_verification_status": VerificationStatus.VERIFIED,
        }
    )
    bundle = bundle.model_copy(update={"source_editions": (edition,)})
    pages = (
        ReferencePage(
            page_number=2,
            text=relationship_text,
            extraction_method="native",
        ),
    )
    return pdf_path, bundle, edition, pages


def test_uploaded_pdf_processing_writes_structure_and_returns_relationship_candidates(
    tmp_path,
):
    pdf_path, bundle, edition, pages = _uploaded_bundle(tmp_path)
    writer = CapturingGraphWriter()

    result = process_uploaded_pdf_graph(
        bundle,
        pages,
        source_pdf_path=pdf_path,
        instrument_catalog=VerifiedInstrumentCatalog.from_bundle(bundle),
        graph_writer=writer,
    )

    assert len(writer.bundles) == 1
    assert writer.bundles[0].instrument_references == ()
    assert result.receipt.source_edition_uid == edition.uid
    assert result.receipt.reference_candidate_count == 1
    assert result.receipt.verified_reference_count == 0
    assert result.receipt.needs_review_reference_count == 1
    assert result.receipt.legal_action_candidate_count == 2
    assert result.receipt.write_receipt.bundle_sha256 == "c" * 64
    assert result.reference_candidates[0].target_instrument_uid == (
        "BCT:CIRCULAR:1991:24"
    )
    assert {
        candidate.proposed_action.value
        for candidate in result.semantic_candidates
        if candidate.candidate_type == CandidateType.LEGAL_ACTION
    } == {"ABROGATE", "REPLACE"}
    assert all(
        candidate.verification_status == VerificationStatus.NEEDS_REVIEW
        for candidate in result.semantic_candidates
    )


def test_uploaded_pdf_processing_persists_only_a_reviewed_exact_reference(tmp_path):
    pdf_path, bundle, _edition, pages = _uploaded_bundle(tmp_path)
    catalog = VerifiedInstrumentCatalog.from_bundle(bundle)
    initial = process_uploaded_pdf_graph(
        bundle,
        pages,
        source_pdf_path=pdf_path,
        instrument_catalog=catalog,
        graph_writer=CapturingGraphWriter(),
    )
    candidate = initial.reference_candidates[0]
    writer = CapturingGraphWriter()

    result = process_uploaded_pdf_graph(
        bundle,
        pages,
        source_pdf_path=pdf_path,
        instrument_catalog=catalog,
        graph_writer=writer,
        reference_reviews={
            candidate.uid: ReferencePromotionEvidence(
                reviewed_target_instrument_uid=candidate.target_instrument_uid,
                reviewer="manual:test-reviewer",
            )
        },
    )

    written = writer.bundles[0]
    assert result.receipt.verified_reference_count == 1
    assert result.receipt.needs_review_reference_count == 0
    assert result.reference_decisions[0].status == VerificationStatus.VERIFIED
    assert len(written.instrument_references) == 1
    assert written.instrument_references[0].target_instrument_uid == (
        "BCT:CIRCULAR:1991:24"
    )
    assert written.instrument_references[0].verified_by == "manual:test-reviewer"
    assert written.instrument_references[0].evidence_uid in {
        evidence.uid for evidence in written.evidence_spans
    }


def test_uploaded_pdf_processing_rejects_stale_review_before_writing(tmp_path):
    pdf_path, bundle, _edition, pages = _uploaded_bundle(tmp_path)
    writer = CapturingGraphWriter()

    with pytest.raises(ValueError, match="do not belong to the uploaded PDF"):
        process_uploaded_pdf_graph(
            bundle,
            pages,
            source_pdf_path=pdf_path,
            instrument_catalog=VerifiedInstrumentCatalog.from_bundle(bundle),
            graph_writer=writer,
            reference_reviews={
                "reference-candidate:from-an-older-upload": (
                    ReferencePromotionEvidence(
                        reviewed_target_instrument_uid="BCT:CIRCULAR:1991:24",
                        reviewer="manual:test-reviewer",
                    )
                )
            },
        )

    assert writer.bundles == []
