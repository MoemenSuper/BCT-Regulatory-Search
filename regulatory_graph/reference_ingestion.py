from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Collection, Literal

from pydantic import Field

from regulatory_graph.models import (
    EvidenceSpan,
    GraphModel,
    Instrument,
    InstrumentKind,
    InstrumentReference,
    NonEmptyStr,
    ReferenceResolverRule,
    ReferenceVerificationMethod,
    RegulatoryGraphBundle,
    Sha256,
    SourceEdition,
    SourceStatus,
    VerificationStatus,
)


class ReferencePage(GraphModel):
    page_number: int = Field(ge=1)
    text: str
    extraction_method: NonEmptyStr


class InstrumentReferenceCandidate(GraphModel):
    uid: NonEmptyStr
    source_instrument_uid: NonEmptyStr
    source_edition_uid: NonEmptyStr
    source_filename: NonEmptyStr
    source_sha256: Sha256
    extraction_artifact_hash: Sha256
    page_number: int = Field(ge=1)
    extraction_method: NonEmptyStr
    target_instrument: Instrument
    signal: NonEmptyStr
    quote: NonEmptyStr
    match_start: int = Field(ge=0)
    match_end: int = Field(gt=0)
    resolver_rule: ReferenceResolverRule
    verification_status: Literal[VerificationStatus.NEEDS_REVIEW] = (
        VerificationStatus.NEEDS_REVIEW
    )

    @property
    def target_instrument_uid(self) -> str:
        return self.target_instrument.uid

    @property
    def target_corpus_present(self) -> bool:
        return self.target_instrument.corpus_present


class ReferencePromotionEvidence(GraphModel):
    reviewed_target_instrument_uid: NonEmptyStr
    verification_method: Literal[
        ReferenceVerificationMethod.MANUAL_RENDERED_PDF_V1
    ] = ReferenceVerificationMethod.MANUAL_RENDERED_PDF_V1
    reviewer: NonEmptyStr


class NeedsReviewReferencePromotion(GraphModel):
    status: Literal[VerificationStatus.NEEDS_REVIEW] = VerificationStatus.NEEDS_REVIEW
    reasons: tuple[NonEmptyStr, ...] = Field(min_length=1)
    reference: None = None
    evidence_span: None = None
    target_instrument: None = None


class VerifiedReferencePromotion(GraphModel):
    status: Literal[VerificationStatus.VERIFIED] = VerificationStatus.VERIFIED
    reasons: tuple[NonEmptyStr, ...] = ()
    reference: InstrumentReference
    evidence_span: EvidenceSpan
    target_instrument: Instrument


ReferencePromotionDecision = (
    NeedsReviewReferencePromotion | VerifiedReferencePromotion
)


class VerifiedInstrumentCatalog:
    def __init__(
        self,
        instruments: Collection[Instrument],
        source_editions: Collection[SourceEdition],
    ):
        self._by_uid: dict[str, Instrument] = {}
        for instrument in instruments:
            existing = self._by_uid.get(instrument.uid)
            if existing is not None and existing != instrument:
                raise ValueError(f"conflicting catalog instruments share uid {instrument.uid}")
            self._by_uid[instrument.uid] = instrument
        self._editions_by_uid: dict[str, SourceEdition] = {}
        for edition in source_editions:
            existing_edition = self._editions_by_uid.get(edition.uid)
            if existing_edition is not None and existing_edition != edition:
                raise ValueError(
                    f"conflicting catalog source editions share uid {edition.uid}"
                )
            self._editions_by_uid[edition.uid] = edition
        verified_local_uids = {
            edition.instrument_uid
            for edition in self._editions_by_uid.values()
            if edition.lifecycle_status == "VALIDATED"
            and edition.identity_verification_status == VerificationStatus.VERIFIED
        }
        missing_provenance = sorted(
            instrument.uid
            for instrument in self._by_uid.values()
            if instrument.corpus_present and instrument.uid not in verified_local_uids
        )
        if missing_provenance:
            raise ValueError(
                "local catalog instruments require a validated identity-verified "
                f"source edition: {missing_provenance}"
            )

    @classmethod
    def from_bundle(
        cls,
        bundle: RegulatoryGraphBundle,
    ) -> "VerifiedInstrumentCatalog":
        return cls(bundle.instruments, bundle.source_editions)

    def get(self, uid: str) -> Instrument | None:
        return self._by_uid.get(uid)

    def contains_source_edition(self, edition: SourceEdition) -> bool:
        return self._editions_by_uid.get(edition.uid) == edition

    def get_source_edition(self, uid: str) -> SourceEdition | None:
        return self._editions_by_uid.get(uid)

    def resolve_bct_reference(
        self,
        *,
        kind: InstrumentKind,
        year: int,
        number: str,
        raw_citation: str,
    ) -> Instrument:
        uid = f"BCT:{kind.value}:{year}:{number}"
        existing = self.get(uid)
        if existing is not None:
            expected = ("BCT", kind, year, number)
            observed = (
                existing.authority,
                existing.kind,
                existing.year,
                existing.number,
            )
            if observed != expected:
                raise ValueError(f"catalog identity fields conflict for {uid}")
            return existing
        return Instrument(
            uid=uid,
            authority="BCT",
            kind=kind,
            year=year,
            number=number,
            corpus_present=False,
            canonical_citation=raw_citation,
            source_status=SourceStatus.EXTERNAL_STUB,
        )


_FRENCH_REFERENCE = re.compile(
    r"\b(?P<kind>circulaire|note)\s+"
    r"(?:de\s+la\s+banque\s+centrale\s+de\s+tunisie\s+)?"
    r"n\s*[°º�]?\s*(?P<year>[0-9]{2,4})\s*[-/]\s*(?P<number>[0-9]+)\b",
    re.IGNORECASE,
)
_ARABIC_REFERENCE = re.compile(
    r"(?P<kind>المنشور|منشور|المذكرة|مذكرة)\s+"
    r"(?:عدد\s+)?(?P<number>[0-9٠-٩]+)\s+"
    r"لسنة\s+(?P<year>[0-9٠-٩]{4})"
)
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def extract_document_reference_candidates(
    source_edition: SourceEdition,
    pages: tuple[ReferencePage, ...],
    *,
    instrument_catalog: VerifiedInstrumentCatalog,
) -> tuple[InstrumentReferenceCandidate, ...]:
    if source_edition.extraction_artifact_hash is None:
        raise ValueError("validated reference ingestion requires an artifact hash")
    if (
        source_edition.lifecycle_status != "VALIDATED"
        or source_edition.identity_verification_status != VerificationStatus.VERIFIED
    ):
        raise ValueError(
            "reference ingestion requires a validated edition with verified "
            "source instrument identity"
        )
    source_instrument = instrument_catalog.get(source_edition.instrument_uid)
    if (
        source_instrument is None
        or not source_instrument.corpus_present
        or not instrument_catalog.contains_source_edition(source_edition)
    ):
        raise ValueError(
            "reference ingestion requires the source instrument in the verified catalog"
        )
    page_numbers = [page.page_number for page in pages]
    if (
        len(page_numbers) != len(set(page_numbers))
        or any(number > source_edition.page_count for number in page_numbers)
    ):
        raise ValueError(
            "reference pages must be unique and within the source edition"
        )
    candidates = []
    for page in pages:
        for rule, pattern in (
            (ReferenceResolverRule.FRENCH_BCT_INSTRUMENT_V1, _FRENCH_REFERENCE),
            (ReferenceResolverRule.ARABIC_BCT_INSTRUMENT_V1, _ARABIC_REFERENCE),
        ):
            for match in pattern.finditer(page.text):
                kind = _instrument_kind(match.group("kind"))
                year = _four_digit_year(match.group("year"))
                number = _normalize_number(match.group("number"))
                target_uid = f"BCT:{kind.value}:{year}:{number}"
                signal = match.group(0)
                target = instrument_catalog.resolve_bct_reference(
                    kind=kind,
                    year=year,
                    number=number,
                    raw_citation=signal,
                )
                identity = "|".join(
                    (
                        source_edition.sha256.upper(),
                        source_edition.uid,
                        source_edition.extraction_artifact_hash.upper(),
                        str(page.page_number),
                        str(match.start()),
                        str(match.end()),
                        target_uid,
                        signal,
                    )
                )
                candidates.append(
                    InstrumentReferenceCandidate(
                        uid=(
                            "reference-candidate:"
                            + sha256(identity.encode("utf-8")).hexdigest()
                        ),
                        source_instrument_uid=source_edition.instrument_uid,
                        source_edition_uid=source_edition.uid,
                        source_filename=source_edition.filename,
                        source_sha256=source_edition.sha256,
                        extraction_artifact_hash=(
                            source_edition.extraction_artifact_hash
                        ),
                        page_number=page.page_number,
                        extraction_method=page.extraction_method,
                        target_instrument=target,
                        signal=signal,
                        quote=_line_quote(page.text, match.start(), match.end()),
                        match_start=match.start(),
                        match_end=match.end(),
                        resolver_rule=rule,
                    )
                )
    return tuple(sorted(candidates, key=_candidate_sort_key))


def promote_reference_candidate(
    candidate: InstrumentReferenceCandidate,
    evidence: ReferencePromotionEvidence,
    *,
    source_pdf_path: str | Path,
    instrument_catalog: VerifiedInstrumentCatalog,
) -> ReferencePromotionDecision:
    target = candidate.target_instrument
    if (
        target.authority != "BCT"
        or target.kind not in {InstrumentKind.CIRCULAR, InstrumentKind.NOTE}
        or target.year is None
        or target.number is None
    ):
        return NeedsReviewReferencePromotion(
            reasons=("target_identity_incomplete",),
        )
    catalog_target = instrument_catalog.resolve_bct_reference(
        kind=target.kind,
        year=target.year,
        number=target.number,
        raw_citation=candidate.signal,
    )
    catalog_edition = instrument_catalog.get_source_edition(
        candidate.source_edition_uid
    )
    checks = [
        (
            catalog_edition is not None
            and catalog_edition.instrument_uid == candidate.source_instrument_uid
            and catalog_edition.filename == candidate.source_filename
            and catalog_edition.sha256.casefold()
            == candidate.source_sha256.casefold()
            and catalog_edition.extraction_artifact_hash is not None
            and catalog_edition.extraction_artifact_hash.casefold()
            == candidate.extraction_artifact_hash.casefold(),
            "source_catalog_changed",
        ),
        (
            evidence.reviewed_target_instrument_uid
            == candidate.target_instrument_uid,
            "target_identity_ambiguous",
        ),
        (catalog_target == candidate.target_instrument, "target_catalog_changed"),
    ]
    rendered = _render_and_verify_source_reference(candidate, source_pdf_path)
    checks.extend(rendered[0])
    reasons = tuple(reason for passed, reason in checks if not passed)
    if reasons:
        return NeedsReviewReferencePromotion(
            reasons=reasons,
        )

    assert rendered[1] is not None
    verified_signal, verified_quote, quote_start, quote_end, render_hash = rendered[1]

    evidence_uid = candidate.uid.replace("reference-candidate:", "evidence:reference:")
    reference_uid = candidate.uid.replace("reference-candidate:", "reference:")
    evidence_span = EvidenceSpan(
        uid=evidence_uid,
        source_edition_uid=candidate.source_edition_uid,
        quote=verified_quote,
        page_number=candidate.page_number,
        extraction_method=candidate.extraction_method,
        source_sha256=candidate.source_sha256,
        extraction_artifact_hash=candidate.extraction_artifact_hash,
        char_start=quote_start,
        char_end=quote_end,
    )
    reference = InstrumentReference(
        uid=reference_uid,
        source_instrument_uid=candidate.source_instrument_uid,
        target_instrument_uid=candidate.target_instrument_uid,
        evidence_uid=evidence_span.uid,
        raw_citation=verified_signal,
        extraction_method=candidate.extraction_method,
        resolver_rule=candidate.resolver_rule,
        verification_status=VerificationStatus.VERIFIED,
        verification_method=evidence.verification_method,
        rendered_image_sha256=render_hash,
        verified_by=evidence.reviewer,
    )
    return VerifiedReferencePromotion(
        reference=reference,
        evidence_span=evidence_span,
        target_instrument=catalog_target,
    )


def enrich_bundle_with_verified_references(
    bundle: RegulatoryGraphBundle,
    decisions: tuple[ReferencePromotionDecision, ...],
) -> RegulatoryGraphBundle:
    verified_decisions = tuple(
        item for item in decisions if isinstance(item, VerifiedReferencePromotion)
    )
    if len(verified_decisions) != len(decisions):
        raise ValueError("only complete VERIFIED reference decisions can enrich a bundle")

    instruments_by_uid = {item.uid: item for item in bundle.instruments}
    evidence_by_uid = {item.uid: item for item in bundle.evidence_spans}
    references_by_uid = {item.uid: item for item in bundle.instrument_references}
    for decision in verified_decisions:
        target = decision.target_instrument
        evidence_span = decision.evidence_span
        reference = decision.reference
        instruments_by_uid.setdefault(target.uid, target)
        _add_exact(evidence_by_uid, evidence_span)
        _add_exact(references_by_uid, reference)

    payload = bundle.model_dump(mode="python")
    payload.update(
        instruments=tuple(instruments_by_uid.values()),
        evidence_spans=tuple(evidence_by_uid.values()),
        instrument_references=tuple(references_by_uid.values()),
    )
    return RegulatoryGraphBundle.model_validate(payload)


def _instrument_kind(value: str) -> InstrumentKind:
    normalized = value.casefold()
    if "circulaire" in normalized or "منشور" in value:
        return InstrumentKind.CIRCULAR
    return InstrumentKind.NOTE


def _four_digit_year(value: str) -> int:
    normalized = int(value.translate(_ARABIC_DIGITS))
    if len(value) == 2:
        return 2000 + normalized if normalized < 50 else 1900 + normalized
    return normalized


def _normalize_number(value: str) -> str:
    normalized = value.translate(_ARABIC_DIGITS)
    return normalized.zfill(2) if len(normalized) < 2 else normalized


def _line_quote(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()


def _render_and_verify_source_reference(
    candidate: InstrumentReferenceCandidate,
    source_pdf_path: str | Path,
) -> tuple[
    list[tuple[bool, str]],
    tuple[str, str, int, int, str] | None,
]:
    import fitz

    path = Path(source_pdf_path)
    try:
        source_bytes = path.read_bytes()
    except OSError:
        return [(False, "source_pdf_unavailable")], None
    if sha256(source_bytes).hexdigest().casefold() != candidate.source_sha256.casefold():
        return [(False, "source_hash_mismatch")], None

    try:
        with fitz.open(path) as document:
            if candidate.page_number > len(document):
                return [(False, "source_page_mismatch")], None
            page = document[candidate.page_number - 1]
            page_text = page.get_text()
            verified = _find_target_reference(
                page_text,
                candidate.target_instrument_uid,
            )
            if verified is None:
                return [(False, "rendered_page_reference_not_found")], None
            signal, quote, quote_start, quote_end = verified
            rendered_png = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False,
            ).tobytes("png")
    except (OSError, RuntimeError, ValueError):
        return [(False, "rendered_page_unavailable")], None

    render_hash = sha256(rendered_png).hexdigest().upper()
    return [(True, "rendered_page_reference_not_found")], (
        signal,
        quote,
        quote_start,
        quote_end,
        render_hash,
    )


def _find_target_reference(
    page_text: str,
    target_instrument_uid: str,
) -> tuple[str, str, int, int] | None:
    for pattern in (_FRENCH_REFERENCE, _ARABIC_REFERENCE):
        for match in pattern.finditer(page_text):
            kind = _instrument_kind(match.group("kind"))
            year = _four_digit_year(match.group("year"))
            number = _normalize_number(match.group("number"))
            if f"BCT:{kind.value}:{year}:{number}" != target_instrument_uid:
                continue
            line_start = page_text.rfind("\n", 0, match.start()) + 1
            line_end = page_text.find("\n", match.end())
            if line_end == -1:
                line_end = len(page_text)
            quote_start = line_start
            quote_end = line_end
            while quote_start < quote_end and page_text[quote_start].isspace():
                quote_start += 1
            while quote_end > quote_start and page_text[quote_end - 1].isspace():
                quote_end -= 1
            return (
                match.group(0),
                page_text[quote_start:quote_end],
                quote_start,
                quote_end,
            )
    return None


def _candidate_sort_key(
    item: InstrumentReferenceCandidate,
) -> tuple[object, ...]:
    return (
        item.page_number,
        item.match_start,
        item.target_instrument_uid,
        item.uid,
    )


def _add_exact(collection: dict[str, GraphModel], item: GraphModel) -> None:
    existing = collection.get(item.uid)
    if existing is not None and existing != item:
        raise ValueError(f"conflicting graph facts share uid {item.uid}")
    collection.setdefault(item.uid, item)
