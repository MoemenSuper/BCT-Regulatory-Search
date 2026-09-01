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
    verify_reference_candidate,
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


def test_arabic_official_audience_phrase_survives_extraction_spacing_variants():
    structured_text = (
        "وعلى ال منشور إلى البنوك والمؤسسات المالية عدد 6 لسنة 2022.\n"
        "من المنشور إلى البنوك والمؤسسات المالية عدد6 لسنة2022.\n"
        "من المنشور إلى البنوك والمؤسسات المالية عدد 6 لسنة2022."
    )
    rendered_text = (
        "وعلى ال  منشور  إلى  البنوك والمؤسسات  المالية عدد  \n"
        "6\n  لسنة2022.\n"
        "من المنشور إلى البنوك والمؤسسات المالية عدد6 لسنة2022.\n"
        "من المنشور إلى\nالبنوك والمؤسسات المالية\nعدد6\nلسنة2022."
    )
    target = Instrument(
        uid="BCT:CIRCULAR:2022:06",
        authority="BCT",
        kind=InstrumentKind.CIRCULAR,
        year=2022,
        number="06",
        corpus_present=True,
        source_status=SourceStatus.LOCAL,
    )
    edition = _edition(language="ar")
    catalog = _catalog(target, source_edition=edition)

    structured = extract_document_reference_candidates(
        edition,
        (
            ReferencePage(
                page_number=1,
                text=structured_text,
                extraction_method="structured",
            ),
        ),
        instrument_catalog=catalog,
    )
    rendered = extract_document_reference_candidates(
        edition,
        (
            ReferencePage(
                page_number=1,
                text=rendered_text,
                extraction_method="native",
            ),
        ),
        instrument_catalog=catalog,
    )

    assert [item.target_instrument_uid for item in structured] == [
        "BCT:CIRCULAR:2022:06",
    ] * 3
    assert [item.target_occurrence_index for item in structured] == [0, 1, 2]
    assert [item.target_instrument_uid for item in rendered] == [
        "BCT:CIRCULAR:2022:06",
    ] * 3
    assert [item.target_occurrence_index for item in rendered] == [0, 1, 2]


@pytest.mark.parametrize(
    "text",
    (
        "\u200fال\u200f منشور \u200eإلى\u200f البنوك "
        "والمؤسسات المالية عدد\u200f6 لسنة\u200e2022",
        "المنشور إلى البنوك عدد: 6 لسنة 2022",
        "المنشور إلى البنوك عدد، 6 لسنة 2022",
    ),
)
def test_arabic_reference_accepts_bidi_controls_and_number_punctuation(text):
    candidates = extract_document_reference_candidates(
        _edition(language="ar"),
        (ReferencePage(page_number=1, text=text, extraction_method="native"),),
        instrument_catalog=_catalog(source_edition=_edition(language="ar")),
    )

    assert len(candidates) == 1
    assert candidates[0].target_instrument_uid == "BCT:CIRCULAR:2022:06"
    assert candidates[0].signal in text


def test_arabic_visually_reversed_year_is_not_a_reference_candidate():
    page = ReferencePage(
        page_number=1,
        text="المذكرة إلى البنوك عدد 02 لسنة 6102",
        extraction_method="native",
    )

    candidates = extract_document_reference_candidates(
        _edition(language="ar"),
        (page,),
        instrument_catalog=_catalog(source_edition=_edition(language="ar")),
    )

    assert candidates == ()


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


def test_exact_rendered_reference_can_be_verified_without_manual_routing_review(
    tmp_path,
):
    pdf_path = tmp_path / "CB_2017_08_FR.pdf"
    source_text = "La presente circulaire cite la circulaire n 2013-15."
    source_sha256 = _write_pdf(pdf_path, (source_text,))
    edition = _edition().model_copy(
        update={"sha256": source_sha256, "page_count": 1}
    )
    catalog = _catalog(source_edition=edition)
    candidate = extract_document_reference_candidates(
        edition,
        (
            ReferencePage(
                page_number=1,
                text=source_text,
                extraction_method="native",
            ),
        ),
        instrument_catalog=catalog,
    )[0]

    decision = verify_reference_candidate(
        candidate,
        source_pdf_path=pdf_path,
        instrument_catalog=catalog,
    )

    assert decision.status == VerificationStatus.VERIFIED
    assert decision.reference.verification_method == (
        "deterministic_rendered_pdf_v1"
    )
    assert decision.reference.verified_by == "deterministic_exact_reference_v1"
    assert decision.evidence_span.quote == source_text
    assert decision.target_instrument.uid == "BCT:CIRCULAR:2013:15"


def test_deterministic_reference_verification_rejects_self_reference(tmp_path):
    pdf_path = tmp_path / "CB_2017_08_FR.pdf"
    source_text = "La presente circulaire est la circulaire n 2017-08."
    source_sha256 = _write_pdf(pdf_path, (source_text,))
    edition = _edition().model_copy(
        update={"sha256": source_sha256, "page_count": 1}
    )
    catalog = _catalog(source_edition=edition)
    candidate = extract_document_reference_candidates(
        edition,
        (
            ReferencePage(
                page_number=1,
                text=source_text,
                extraction_method="native",
            ),
        ),
        instrument_catalog=catalog,
    )[0]

    decision = verify_reference_candidate(
        candidate,
        source_pdf_path=pdf_path,
        instrument_catalog=catalog,
    )

    assert decision.status == VerificationStatus.NEEDS_REVIEW
    assert decision.reasons == ("self_reference",)


def test_deterministic_verifier_quarantines_arabic_reversed_self_number_only(
    tmp_path,
):
    source = Instrument(
        uid="BCT:NOTE:2019:05",
        authority="BCT",
        kind=InstrumentKind.NOTE,
        year=2019,
        number="05",
        corpus_present=True,
        source_status=SourceStatus.LOCAL,
    )
    edition = _edition(language="ar").model_copy(
        update={
            "uid": "edition:note-2019-05-ar",
            "instrument_uid": source.uid,
            "filename": "Note_2019_05_ar.pdf",
        }
    )
    catalog = VerifiedInstrumentCatalog((source,), (edition,))
    candidate = extract_document_reference_candidates(
        edition,
        (
            ReferencePage(
                page_number=1,
                text="المذكرة إلى البنوك عدد 50 لسنة 2019",
                extraction_method="native",
            ),
        ),
        instrument_catalog=catalog,
    )[0]

    assert candidate.target_instrument_uid == "BCT:NOTE:2019:50"
    decision = verify_reference_candidate(
        candidate,
        source_pdf_path=tmp_path / "not-needed.pdf",
        instrument_catalog=catalog,
    )

    assert decision.status == VerificationStatus.NEEDS_REVIEW
    assert decision.reasons == ("possible_rtl_reversed_self_reference",)

    cross_kind_candidate = extract_document_reference_candidates(
        edition,
        (
            ReferencePage(
                page_number=1,
                text="المنشور إلى البنوك عدد 50 لسنة 2019",
                extraction_method="native",
            ),
        ),
        instrument_catalog=catalog,
    )[0]
    cross_kind = verify_reference_candidate(
        cross_kind_candidate,
        source_pdf_path=tmp_path / "not-needed.pdf",
        instrument_catalog=catalog,
    )

    assert cross_kind.status == VerificationStatus.NEEDS_REVIEW
    assert cross_kind.reasons == ("source_pdf_unavailable",)


def test_promotion_preserves_repeated_target_occurrence_on_the_same_page(tmp_path):
    pdf_path = tmp_path / "CB_2017_08_FR.pdf"
    source_text = (
        "Premiere mention de la circulaire n 2013-15.\n"
        "Deuxieme mention de la circulaire n 2013-15."
    )
    source_sha256 = _write_pdf(pdf_path, (source_text,))
    edition = _edition().model_copy(
        update={"sha256": source_sha256, "page_count": 1}
    )
    catalog = _catalog(source_edition=edition)
    candidates = extract_document_reference_candidates(
        edition,
        (
            ReferencePage(
                page_number=1,
                text=source_text,
                extraction_method="native",
            ),
        ),
        instrument_catalog=catalog,
    )

    assert [candidate.target_occurrence_index for candidate in candidates] == [0, 1]
    invalid_occurrence = candidates[0].model_dump(mode="python")
    invalid_occurrence.update(
        target_occurrence_index=2,
        target_occurrence_count=1,
    )
    with pytest.raises(ValueError, match="lower than occurrence count"):
        type(candidates[0]).model_validate(invalid_occurrence)

    first = promote_reference_candidate(
        candidates[0],
        _review(candidates[0]),
        source_pdf_path=pdf_path,
        instrument_catalog=catalog,
    )
    second = promote_reference_candidate(
        candidates[1],
        _review(candidates[1]),
        source_pdf_path=pdf_path,
        instrument_catalog=catalog,
    )

    assert first.status == VerificationStatus.VERIFIED
    assert second.status == VerificationStatus.VERIFIED
    assert first.evidence_span.quote.startswith("Premiere mention")
    assert second.evidence_span.quote.startswith("Deuxieme mention")
    assert first.evidence_span.char_start < second.evidence_span.char_start


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
