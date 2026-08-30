from __future__ import annotations

from hashlib import sha256
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
    target_instrument_uid: NonEmptyStr
    target_kind: InstrumentKind
    target_year: int = Field(ge=0)
    target_number: NonEmptyStr
    target_corpus_present: bool
    signal: NonEmptyStr
    quote: NonEmptyStr
    match_start: int = Field(ge=0)
    match_end: int = Field(gt=0)
    resolver_rule: NonEmptyStr
    verification_status: Literal[VerificationStatus.NEEDS_REVIEW] = (
        VerificationStatus.NEEDS_REVIEW
    )


class ReferencePromotionEvidence(GraphModel):
    reviewed_source_sha256: Sha256
    reviewed_page_number: int = Field(ge=1)
    rendered_page_confirmed: bool = False
    reviewed_target_instrument_uid: NonEmptyStr
    reviewer: NonEmptyStr


class ReferencePromotionDecision(GraphModel):
    status: VerificationStatus
    reasons: tuple[NonEmptyStr, ...]
    reference: InstrumentReference | None = None
    evidence_span: EvidenceSpan | None = None
    target_instrument: Instrument | None = None


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
    known_instrument_uids: Collection[str],
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
            ("french_bct_instrument_reference_v1", _FRENCH_REFERENCE),
            ("arabic_bct_instrument_reference_v1", _ARABIC_REFERENCE),
        ):
            for match in pattern.finditer(page.text):
                kind = _instrument_kind(match.group("kind"))
                year = _four_digit_year(match.group("year"))
                number = _normalize_number(match.group("number"))
                target_uid = f"BCT:{kind.value}:{year}:{number}"
                signal = match.group(0)
                identity = "|".join(
                    (
                        source_edition.sha256.upper(),
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
                        target_instrument_uid=target_uid,
                        target_kind=kind,
                        target_year=year,
                        target_number=number,
                        target_corpus_present=target_uid in known_instrument_uids,
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
) -> ReferencePromotionDecision:
    checks = (
        (
            evidence.reviewed_source_sha256.casefold()
            == candidate.source_sha256.casefold(),
            "source_hash_mismatch",
        ),
        (
            evidence.reviewed_page_number == candidate.page_number,
            "source_page_mismatch",
        ),
        (evidence.rendered_page_confirmed, "rendered_page_confirmation_missing"),
        (
            evidence.reviewed_target_instrument_uid
            == candidate.target_instrument_uid,
            "target_identity_ambiguous",
        ),
    )
    reasons = tuple(reason for passed, reason in checks if not passed)
    if reasons:
        return ReferencePromotionDecision(
            status=VerificationStatus.NEEDS_REVIEW,
            reasons=reasons,
        )

    evidence_uid = candidate.uid.replace("reference-candidate:", "evidence:reference:")
    reference_uid = candidate.uid.replace("reference-candidate:", "reference:")
    evidence_span = EvidenceSpan(
        uid=evidence_uid,
        source_edition_uid=candidate.source_edition_uid,
        quote=candidate.quote,
        page_number=candidate.page_number,
        extraction_method=candidate.extraction_method,
        source_sha256=candidate.source_sha256,
        extraction_artifact_hash=candidate.extraction_artifact_hash,
    )
    reference = InstrumentReference(
        uid=reference_uid,
        source_instrument_uid=candidate.source_instrument_uid,
        target_instrument_uid=candidate.target_instrument_uid,
        evidence_uid=evidence_span.uid,
        raw_citation=candidate.signal,
        extraction_method=candidate.extraction_method,
        resolver_rule=candidate.resolver_rule,
        verification_status=VerificationStatus.VERIFIED,
        verified_by=evidence.reviewer,
    )
    target = Instrument(
        uid=candidate.target_instrument_uid,
        authority="BCT",
        kind=candidate.target_kind,
        year=candidate.target_year,
        number=candidate.target_number,
        corpus_present=candidate.target_corpus_present,
        canonical_citation=candidate.signal,
        source_status=(
            SourceStatus.LOCAL
            if candidate.target_corpus_present
            else SourceStatus.EXTERNAL_STUB
        ),
    )
    return ReferencePromotionDecision(
        status=VerificationStatus.VERIFIED,
        reasons=(),
        reference=reference,
        evidence_span=evidence_span,
        target_instrument=target,
    )


def enrich_bundle_with_verified_references(
    bundle: RegulatoryGraphBundle,
    decisions: tuple[ReferencePromotionDecision, ...],
) -> RegulatoryGraphBundle:
    incomplete = [item for item in decisions if item.status != VerificationStatus.VERIFIED]
    if incomplete or any(
        item.reference is None
        or item.evidence_span is None
        or item.target_instrument is None
        for item in decisions
    ):
        raise ValueError("only complete VERIFIED reference decisions can enrich a bundle")

    instruments_by_uid = {item.uid: item for item in bundle.instruments}
    evidence_by_uid = {item.uid: item for item in bundle.evidence_spans}
    references_by_uid = {item.uid: item for item in bundle.instrument_references}
    for decision in decisions:
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
