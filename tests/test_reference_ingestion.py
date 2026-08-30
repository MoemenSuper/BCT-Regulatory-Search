import pytest

from regulatory_graph.models import (
    Instrument,
    InstrumentKind,
    SourceEdition,
    SourceStatus,
    VerificationStatus,
)
from regulatory_graph.fixtures import circular_2016_03_fr_bundle
from regulatory_graph.reference_ingestion import (
    ReferencePage,
    ReferencePromotionEvidence,
    VerifiedInstrumentCatalog,
    enrich_bundle_with_verified_references,
    extract_document_reference_candidates,
    promote_reference_candidate,
)


def _edition(*, language="fr") -> SourceEdition:
    return SourceEdition(
        uid=f"edition:cb-2017-08-{language}",
        instrument_uid="BCT:CIRCULAR:2017:08",
        language=language,
        filename=f"CB_2017_08_{language.upper()}.pdf",
        sha256="a" * 64,
        extraction_status="validated",
        page_count=39,
        is_scan=False,
        extraction_artifact_hash="b" * 64,
        identity_verification_status=VerificationStatus.VERIFIED,
    )


def _catalog(*targets: Instrument) -> VerifiedInstrumentCatalog:
    source = Instrument(
        uid="BCT:CIRCULAR:2017:08",
        authority="BCT",
        kind=InstrumentKind.CIRCULAR,
        year=2017,
        number="08",
        corpus_present=True,
        source_status=SourceStatus.LOCAL,
    )
    target_editions = tuple(
        SourceEdition(
            uid=f"edition:{target.uid.casefold().replace(':', '-')}:fr",
            instrument_uid=target.uid,
            language="fr",
            filename=f"{target.uid.replace(':', '_')}_fr.pdf",
            sha256="d" * 64,
            extraction_status="validated",
            page_count=1,
            is_scan=False,
            extraction_artifact_hash="e" * 64,
            identity_verification_status=VerificationStatus.VERIFIED,
        )
        for target in targets
        if target.corpus_present
    )
    return VerifiedInstrumentCatalog(
        (source, *targets),
        (_edition(), _edition(language="ar"), *target_editions),
    )


def _review(candidate, *, reviewed_quote=None) -> ReferencePromotionEvidence:
    return ReferencePromotionEvidence(
        reviewed_source_sha256=candidate.source_sha256,
        reviewed_page_number=candidate.page_number,
        reviewed_signal=candidate.signal,
        reviewed_quote=(candidate.quote if reviewed_quote is None else reviewed_quote),
        reviewed_match_start=candidate.match_start,
        reviewed_match_end=candidate.match_end,
        reviewed_target_instrument_uid=candidate.target_instrument_uid,
        rendered_image_sha256="c" * 64,
        reviewer="manual:test-reviewer",
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
        instrument_catalog=_catalog(),
    )
    second = extract_document_reference_candidates(
        _edition(),
        (page,),
        instrument_catalog=_catalog(),
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
            instrument_catalog=_catalog(),
        )

    with pytest.raises(ValueError, match="unique and within the source edition"):
        extract_document_reference_candidates(
            _edition(),
            (page, page),
            instrument_catalog=_catalog(),
        )


def test_candidate_uid_includes_immutable_edition_and_artifact_identity():
    page = ReferencePage(
        page_number=1,
        text="Vu la circulaire n°2013-15.",
        extraction_method="native",
    )
    first_edition = _edition()
    second_edition = first_edition.model_copy(
        update={
            "uid": "edition:cb-2017-08-fr:second-extraction",
            "extraction_artifact_hash": "f" * 64,
        }
    )
    source = _catalog().get(first_edition.instrument_uid)
    catalog = VerifiedInstrumentCatalog(
        (source,),
        (first_edition, second_edition),
    )

    first = extract_document_reference_candidates(
        first_edition,
        (page,),
        instrument_catalog=catalog,
    )[0]
    second = extract_document_reference_candidates(
        second_edition,
        (page,),
        instrument_catalog=catalog,
    )[0]

    assert first.uid != second.uid


def test_arabic_indic_digits_resolve_without_reversal_or_normalizing_the_quote():
    text = "طبقا لأحكام المنشور عدد ٦ لسنة ٢٠٢٢ المتعلق بالرقابة الداخلية."
    page = ReferencePage(page_number=1, text=text, extraction_method="native")

    target = Instrument(
        uid="BCT:CIRCULAR:2022:06",
        authority="BCT",
        kind=InstrumentKind.CIRCULAR,
        year=2022,
        number="06",
        corpus_present=True,
        source_status=SourceStatus.LOCAL,
    )
    candidates = extract_document_reference_candidates(
        _edition(language="ar"),
        (page,),
        instrument_catalog=_catalog(target),
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
        instrument_catalog=_catalog(),
    )[0]

    incomplete = promote_reference_candidate(
        candidate,
        _review(candidate, reviewed_quote="not the rendered source quote"),
        instrument_catalog=_catalog(),
    )

    assert incomplete.status == VerificationStatus.NEEDS_REVIEW
    assert incomplete.reference is None
    assert incomplete.evidence_span is None
    assert incomplete.target_instrument is None
    assert incomplete.reasons == ("reviewed_quote_mismatch",)

    complete = promote_reference_candidate(
        candidate,
        _review(candidate),
        instrument_catalog=_catalog(),
    )

    assert complete.status == VerificationStatus.VERIFIED
    assert complete.reasons == ()
    assert complete.reference.verification_status == VerificationStatus.VERIFIED
    assert complete.reference.verified_by == "manual:test-reviewer"
    assert complete.reference.rendered_image_sha256 == "c" * 64
    assert complete.reference.evidence_uid == complete.evidence_span.uid
    assert complete.target_instrument.uid == "BCT:CIRCULAR:2013:15"
    assert complete.target_instrument.source_status == SourceStatus.EXTERNAL_STUB

    newly_local_target = Instrument(
        uid="BCT:CIRCULAR:2013:15",
        authority="BCT",
        kind=InstrumentKind.CIRCULAR,
        year=2013,
        number="15",
        corpus_present=True,
        source_status=SourceStatus.LOCAL,
    )
    stale = promote_reference_candidate(
        candidate,
        _review(candidate),
        instrument_catalog=_catalog(newly_local_target),
    )
    assert stale.status == VerificationStatus.NEEDS_REVIEW
    assert stale.reasons == ("target_catalog_changed",)


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
    catalog = VerifiedInstrumentCatalog.from_bundle(bundle)
    candidate = extract_document_reference_candidates(
        edition,
        (page,),
        instrument_catalog=catalog,
    )[0]
    decision = promote_reference_candidate(
        candidate,
        _review(candidate),
        instrument_catalog=catalog,
    )

    first = enrich_bundle_with_verified_references(bundle, (decision,))
    second = enrich_bundle_with_verified_references(first, (decision,))

    assert first == second
    assert len(first.instrument_references) == 1
    assert len(first.evidence_spans) == len(bundle.evidence_spans) + 1
    assert len(first.instruments) == len(bundle.instruments)
