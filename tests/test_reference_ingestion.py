import pytest

from regulatory_graph.models import SourceEdition, SourceStatus, VerificationStatus
from regulatory_graph.fixtures import circular_2016_03_fr_bundle
from regulatory_graph.reference_ingestion import (
    ReferencePage,
    ReferencePromotionEvidence,
    enrich_bundle_with_verified_references,
    extract_document_reference_candidates,
    promote_reference_candidate,
)


def _edition(*, language="fr") -> SourceEdition:
    return SourceEdition(
        uid="edition:cb-2017-08-fr",
        instrument_uid="BCT:CIRCULAR:2017:08",
        language=language,
        filename="CB_2017_08_FR.pdf",
        sha256="a" * 64,
        extraction_status="validated",
        page_count=39,
        is_scan=False,
        extraction_artifact_hash="b" * 64,
        identity_verification_status=VerificationStatus.VERIFIED,
    )


def test_one_document_extracts_stable_exact_source_reference_candidates():
    text = (
        "Article 63 :\n"
        "La présente circulaire abroge et remplace la circulaire n°2013-15."
    )
    page = ReferencePage(page_number=1, text=text, extraction_method="native")

    first = extract_document_reference_candidates(
        _edition(),
        (page,),
        known_instrument_uids={"BCT:CIRCULAR:2017:08"},
    )
    second = extract_document_reference_candidates(
        _edition(),
        (page,),
        known_instrument_uids={"BCT:CIRCULAR:2017:08"},
    )

    assert first == second
    assert len(first) == 1
    candidate = first[0]
    assert candidate.target_instrument_uid == "BCT:CIRCULAR:2013:15"
    assert candidate.target_corpus_present is False
    assert candidate.verification_status == VerificationStatus.NEEDS_REVIEW
    assert candidate.quote in text
    assert text[candidate.match_start:candidate.match_end] == candidate.signal


def test_reference_ingestion_rejects_unverified_identity_and_invalid_pages():
    page = ReferencePage(
        page_number=1,
        text="Vu la circulaire n°2013-15.",
        extraction_method="native",
    )
    unverified = _edition().model_copy(
        update={"identity_verification_status": VerificationStatus.NEEDS_REVIEW}
    )

    with pytest.raises(ValueError, match="verified source instrument identity"):
        extract_document_reference_candidates(
            unverified,
            (page,),
            known_instrument_uids=set(),
        )

    with pytest.raises(ValueError, match="unique and within the source edition"):
        extract_document_reference_candidates(
            _edition(),
            (page, page),
            known_instrument_uids=set(),
        )


def test_arabic_indic_digits_resolve_without_reversal_or_normalizing_the_quote():
    text = "طبقا لأحكام المنشور عدد ٦ لسنة ٢٠٢٢ المتعلق بالرقابة الداخلية."
    page = ReferencePage(page_number=1, text=text, extraction_method="native")

    candidates = extract_document_reference_candidates(
        _edition(language="ar"),
        (page,),
        known_instrument_uids={
            "BCT:CIRCULAR:2017:08",
            "BCT:CIRCULAR:2022:06",
        },
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.target_instrument_uid == "BCT:CIRCULAR:2022:06"
    assert candidate.target_corpus_present is True
    assert candidate.signal == "المنشور عدد ٦ لسنة ٢٠٢٢"


def test_reference_promotion_fails_closed_until_every_source_check_passes():
    page = ReferencePage(
        page_number=28,
        text="La présente circulaire abroge et remplace la circulaire n°2013-15.",
        extraction_method="native",
    )
    candidate = extract_document_reference_candidates(
        _edition(),
        (page,),
        known_instrument_uids={"BCT:CIRCULAR:2017:08"},
    )[0]

    incomplete = promote_reference_candidate(
        candidate,
        ReferencePromotionEvidence(
            reviewed_source_sha256=candidate.source_sha256,
            reviewed_page_number=candidate.page_number,
            rendered_page_confirmed=False,
            reviewed_target_instrument_uid=candidate.target_instrument_uid,
            reviewer="manual:test-reviewer",
        ),
    )

    assert incomplete.status == VerificationStatus.NEEDS_REVIEW
    assert incomplete.reference is None
    assert incomplete.evidence_span is None
    assert incomplete.target_instrument is None
    assert incomplete.reasons == ("rendered_page_confirmation_missing",)

    complete = promote_reference_candidate(
        candidate,
        ReferencePromotionEvidence(
            reviewed_source_sha256=candidate.source_sha256,
            reviewed_page_number=candidate.page_number,
            rendered_page_confirmed=True,
            reviewed_target_instrument_uid=candidate.target_instrument_uid,
            reviewer="manual:test-reviewer",
        ),
    )

    assert complete.status == VerificationStatus.VERIFIED
    assert complete.reasons == ()
    assert complete.reference.verification_status == VerificationStatus.VERIFIED
    assert complete.reference.verified_by == "manual:test-reviewer"
    assert complete.reference.evidence_uid == complete.evidence_span.uid
    assert complete.target_instrument.uid == "BCT:CIRCULAR:2013:15"
    assert complete.target_instrument.source_status == SourceStatus.EXTERNAL_STUB


def test_verified_reference_enrichment_is_idempotent_for_one_document_bundle():
    bundle = circular_2016_03_fr_bundle()
    edition = bundle.source_editions[0].model_copy(
        update={
            "extraction_artifact_hash": "b" * 64,
            "identity_verification_status": VerificationStatus.VERIFIED,
        }
    )
    bundle = bundle.model_copy(update={"source_editions": (edition,)})
    page = ReferencePage(
        page_number=2,
        text="Vu la circulaire n°91-24.",
        extraction_method="native",
    )
    candidate = extract_document_reference_candidates(
        edition,
        (page,),
        known_instrument_uids={item.uid for item in bundle.instruments},
    )[0]
    decision = promote_reference_candidate(
        candidate,
        ReferencePromotionEvidence(
            reviewed_source_sha256=candidate.source_sha256,
            reviewed_page_number=candidate.page_number,
            rendered_page_confirmed=True,
            reviewed_target_instrument_uid=candidate.target_instrument_uid,
            reviewer="manual:test-reviewer",
        ),
    )

    first = enrich_bundle_with_verified_references(bundle, (decision,))
    second = enrich_bundle_with_verified_references(first, (decision,))

    assert first == second
    assert len(first.instrument_references) == 1
    assert len(first.evidence_spans) == len(bundle.evidence_spans) + 1
    assert len(first.instruments) == len(bundle.instruments)
