from hashlib import sha256

import fitz
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


def _catalog(
    *targets: Instrument,
    source_edition: SourceEdition | None = None,
) -> VerifiedInstrumentCatalog:
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
        (source_edition or _edition(), *target_editions),
    )


def _review(candidate) -> ReferencePromotionEvidence:
    return ReferencePromotionEvidence(
        reviewed_target_instrument_uid=candidate.target_instrument_uid,
        reviewer="manual:test-reviewer",
    )


def _write_pdf(path, page_texts) -> str:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()
    return sha256(path.read_bytes()).hexdigest().upper()


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
        instrument_catalog=_catalog(
            target,
            source_edition=_edition(language="ar"),
        ),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.target_instrument_uid == "BCT:CIRCULAR:2022:06"
    assert candidate.target_corpus_present is True
    assert candidate.signal == "المنشور عدد ٦ لسنة ٢٠٢٢"


def test_reference_promotion_fails_closed_until_every_source_check_passes(tmp_path):
    pdf_path = tmp_path / "CB_2017_08_FR.pdf"
    source_text = "La presente circulaire cite la circulaire n 2013-15."
    source_sha256 = _write_pdf(pdf_path, (source_text,))
    edition = _edition().model_copy(
        update={"sha256": source_sha256, "page_count": 1}
    )
    page = ReferencePage(
        page_number=1,
        text=source_text,
        extraction_method="native",
    )
    catalog = _catalog(source_edition=edition)
    candidate = extract_document_reference_candidates(
        edition,
        (page,),
        instrument_catalog=catalog,
    )[0]

    wrong_target_review = ReferencePromotionEvidence(
        reviewed_target_instrument_uid="BCT:CIRCULAR:2013:14",
        reviewer="manual:test-reviewer",
    )
    incomplete = promote_reference_candidate(
        candidate,
        wrong_target_review,
        source_pdf_path=pdf_path,
        instrument_catalog=catalog,
    )

    assert incomplete.status == VerificationStatus.NEEDS_REVIEW
    assert incomplete.reference is None
    assert incomplete.evidence_span is None
    assert incomplete.target_instrument is None
    assert incomplete.reasons == ("target_identity_ambiguous",)

    unrelated_pdf = tmp_path / "unrelated.pdf"
    _write_pdf(unrelated_pdf, ("Unrelated page",))
    unrelated = promote_reference_candidate(
        candidate,
        _review(candidate),
        source_pdf_path=unrelated_pdf,
        instrument_catalog=catalog,
    )
    assert unrelated.status == VerificationStatus.NEEDS_REVIEW
    assert unrelated.reasons == ("source_hash_mismatch",)

    complete = promote_reference_candidate(
        candidate,
        _review(candidate),
        source_pdf_path=pdf_path,
        instrument_catalog=catalog,
    )

    assert complete.status == VerificationStatus.VERIFIED
    assert complete.reasons == ()
    assert complete.reference.verification_status == VerificationStatus.VERIFIED
    assert complete.reference.verified_by == "manual:test-reviewer"
    assert len(complete.reference.rendered_image_sha256) == 64
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
        source_pdf_path=pdf_path,
        instrument_catalog=_catalog(
            newly_local_target,
            source_edition=edition,
        ),
    )
    assert stale.status == VerificationStatus.NEEDS_REVIEW
    assert stale.reasons == ("target_catalog_changed",)


def test_verified_reference_enrichment_is_idempotent_for_one_document_bundle(
    tmp_path,
):
    bundle = circular_2016_03_fr_bundle()
    pdf_path = tmp_path / "Cir_2016_03_fr.pdf"
    source_sha256 = _write_pdf(
        pdf_path,
        ("Page 1", "Vu la circulaire n 91-24.", "Page 3", "Page 4"),
    )
    edition = bundle.source_editions[0].model_copy(
        update={
            "sha256": source_sha256,
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
        source_pdf_path=pdf_path,
        instrument_catalog=catalog,
    )

    first = enrich_bundle_with_verified_references(bundle, (decision,))
    second = enrich_bundle_with_verified_references(first, (decision,))

    assert first == second
    assert len(first.instrument_references) == 1
    assert len(first.evidence_spans) == len(bundle.evidence_spans) + 1
    assert len(first.instruments) == len(bundle.instruments)
