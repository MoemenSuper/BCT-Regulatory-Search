from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

from pydantic import Field

from regulatory_graph.models import (
    GraphModel,
    Language,
    LegalAction,
    NonEmptyStr,
    Sha256,
    VerificationStatus,
)
from regulatory_graph.corpus_structure import CacheInventory, build_structural_bundle


class CandidateType(str, Enum):
    PROVISION_HEADING = "PROVISION_HEADING"
    LEGAL_ACTION = "LEGAL_ACTION"
    CROSS_REFERENCE = "CROSS_REFERENCE"


class SemanticCandidate(GraphModel):
    uid: NonEmptyStr
    candidate_type: CandidateType
    filename: NonEmptyStr
    source_edition_uid: NonEmptyStr
    instrument_uid: NonEmptyStr
    language: Language
    page_number: int = Field(ge=1)
    source_sha256: Sha256
    extraction_artifact_hash: Sha256
    signal: NonEmptyStr
    match_start: int = Field(ge=0)
    match_end: int = Field(gt=0)
    evidence_quote: NonEmptyStr
    rule: NonEmptyStr
    proposed_action: LegalAction | None = None
    verification_status: VerificationStatus = VerificationStatus.NEEDS_REVIEW
    review_reason: NonEmptyStr = "deterministic_signal_requires_legal_review"


@dataclass(frozen=True)
class CandidateQueueReceipt:
    candidate_count: int
    content_sha256: str


_HEADING_RULES = (
    (
        "french_article_heading_v1",
        re.compile(
            r"^\s*(?:article|art[.]?)\s+(?:premier|[0-9]+(?:\s*(?:bis|ter))?)\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "arabic_article_heading_v1",
        re.compile(
            r"^\s*\u0627\u0644\u0641\u0635\u0644\s+"
            r"(?:\u0627\u0644\u0623\u0648\u0644|[0-9\u0660-\u0669]+)",
            re.MULTILINE,
        ),
    ),
)

_ACTION_RULES = (
    (
        "french_replace_v1",
        LegalAction.REPLACE,
        re.compile(r"\bremplac\w*\b", re.IGNORECASE),
    ),
    (
        "french_abrogate_v1",
        LegalAction.ABROGATE,
        re.compile(r"\babrog\w*\b", re.IGNORECASE),
    ),
    (
        "french_add_v1",
        LegalAction.ADD,
        re.compile(r"\b(?:ajout\w*|ins[eé]r\w*)\b", re.IGNORECASE),
    ),
    (
        "french_modify_v1",
        LegalAction.MODIFY,
        re.compile(r"\bmodifi\w*\b", re.IGNORECASE),
    ),
    (
        "french_supplement_v1",
        LegalAction.SUPPLEMENT,
        re.compile(r"\bcompl[eé]t\w*\b", re.IGNORECASE),
    ),
    (
        "french_derogate_v1",
        LegalAction.DEROGATE,
        re.compile(r"\bd[eé]rog\w*\b", re.IGNORECASE),
    ),
    (
        "arabic_replace_v1",
        LegalAction.REPLACE,
        re.compile(r"\u062a\u0639\u0648\u0651?\u0636"),
    ),
    (
        "arabic_modify_v1",
        LegalAction.MODIFY,
        re.compile(r"\u062a\u0646\u0642\u0651?\u062d"),
    ),
    (
        "arabic_abrogate_v1",
        LegalAction.ABROGATE,
        re.compile(r"(?:\u062a\u0644\u063a\u0649|\u0625\u0644\u063a\u0627\u0621)"),
    ),
    (
        "arabic_add_v1",
        LegalAction.ADD,
        re.compile(r"\u064a\u0636\u0627\u0641"),
    ),
    (
        "arabic_derogate_v1",
        LegalAction.DEROGATE,
        re.compile(r"\u0627\u0633\u062a\u062b\u0646\u0627\u0621"),
    ),
)

_REFERENCE_RULES = (
    (
        "french_provision_reference_v1",
        re.compile(
            r"\b(?:article|art[.]?|annexe|section|chapitre)\s+"
            r"(?:premier|[0-9]+(?:\s*(?:bis|ter))?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "french_instrument_reference_v1",
        re.compile(
            r"\b(?:circulaire|note|loi|d[eé]cret)\s+"
            r"(?:n\s*[°º]?\s*)?[0-9]{2,4}\s*[-/]\s*[0-9]+\b",
            re.IGNORECASE,
        ),
    ),
    (
        "arabic_provision_reference_v1",
        re.compile(
            r"\u0627\u0644\u0641\u0635\u0644\s+"
            r"(?:\u0627\u0644\u0623\u0648\u0644|[0-9\u0660-\u0669]+)"
        ),
    ),
    (
        "arabic_instrument_reference_v1",
        re.compile(
            r"(?:\u0645\u0646\u0634\u0648\u0631|\u0645\u0630\u0643\u0631\u0629|"
            r"\u0642\u0627\u0646\u0648\u0646)\s+"
            r"(?:\u0639\u062f\u062f\s+)?[0-9\u0660-\u0669]+\s+"
            r"(?:\u0644\u0633\u0646\u0629\s+)?[0-9\u0660-\u0669]{4}"
        ),
    ),
)


def extract_page_candidates(
    text: str,
    *,
    filename: str,
    source_edition_uid: str,
    instrument_uid: str,
    language: Language,
    page_number: int,
    source_sha256: str,
    extraction_artifact_hash: str,
) -> tuple[SemanticCandidate, ...]:
    candidates = []
    for rule, pattern in _HEADING_RULES:
        for match in pattern.finditer(text):
            candidates.append(
                _candidate(
                    candidate_type=CandidateType.PROVISION_HEADING,
                    match=match,
                    rule=rule,
                    proposed_action=None,
                    text=text,
                    filename=filename,
                    source_edition_uid=source_edition_uid,
                    instrument_uid=instrument_uid,
                    language=language,
                    page_number=page_number,
                    source_sha256=source_sha256,
                    extraction_artifact_hash=extraction_artifact_hash,
                )
            )
    for rule, action, pattern in _ACTION_RULES:
        for match in pattern.finditer(text):
            candidates.append(
                _candidate(
                    candidate_type=CandidateType.LEGAL_ACTION,
                    match=match,
                    rule=rule,
                    proposed_action=action,
                    text=text,
                    filename=filename,
                    source_edition_uid=source_edition_uid,
                    instrument_uid=instrument_uid,
                    language=language,
                    page_number=page_number,
                    source_sha256=source_sha256,
                    extraction_artifact_hash=extraction_artifact_hash,
                )
            )
    for rule, pattern in _REFERENCE_RULES:
        for match in pattern.finditer(text):
            candidates.append(
                _candidate(
                    candidate_type=CandidateType.CROSS_REFERENCE,
                    match=match,
                    rule=rule,
                    proposed_action=None,
                    text=text,
                    filename=filename,
                    source_edition_uid=source_edition_uid,
                    instrument_uid=instrument_uid,
                    language=language,
                    page_number=page_number,
                    source_sha256=source_sha256,
                    extraction_artifact_hash=extraction_artifact_hash,
                )
            )
    return tuple(sorted(candidates, key=_candidate_sort_key))


def extract_corpus_candidates(
    inventory: CacheInventory,
) -> tuple[SemanticCandidate, ...]:
    bundle = build_structural_bundle(inventory)
    editions_by_filename = {item.filename: item for item in bundle.source_editions}
    candidates = []
    for cached in inventory.editions:
        edition = editions_by_filename[cached.filename]
        for page in cached.document["pages"]:
            candidates.extend(
                extract_page_candidates(
                    str(page.get("raw_text", "")),
                    filename=cached.filename,
                    source_edition_uid=edition.uid,
                    instrument_uid=edition.instrument_uid,
                    language=edition.language,
                    page_number=int(page["page_number"]),
                    source_sha256=edition.sha256,
                    extraction_artifact_hash=cached.artifact_sha256,
                )
            )
    return tuple(sorted(candidates, key=_candidate_sort_key))


def write_candidate_review_queue(
    candidates: Iterable[SemanticCandidate],
    path: str | Path,
) -> CandidateQueueReceipt:
    ordered = sorted(candidates, key=_candidate_sort_key)
    payload = "".join(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        + "\n"
        for item in ordered
    )
    encoded = payload.encode("utf-8")
    Path(path).write_bytes(encoded)
    return CandidateQueueReceipt(
        candidate_count=len(ordered),
        content_sha256=sha256(encoded).hexdigest().upper(),
    )


def _candidate(
    *,
    candidate_type: CandidateType,
    match: re.Match[str],
    rule: str,
    proposed_action: LegalAction | None,
    text: str,
    filename: str,
    source_edition_uid: str,
    instrument_uid: str,
    language: Language,
    page_number: int,
    source_sha256: str,
    extraction_artifact_hash: str,
) -> SemanticCandidate:
    signal = match.group(0)
    identity = "|".join(
        (
            source_sha256.upper(),
            str(page_number),
            candidate_type.value,
            str(match.start()),
            str(match.end()),
            signal,
        )
    )
    return SemanticCandidate(
        uid=f"semantic-candidate:{sha256(identity.encode('utf-8')).hexdigest()}",
        candidate_type=candidate_type,
        filename=filename,
        source_edition_uid=source_edition_uid,
        instrument_uid=instrument_uid,
        language=language,
        page_number=page_number,
        source_sha256=source_sha256,
        extraction_artifact_hash=extraction_artifact_hash,
        signal=signal,
        match_start=match.start(),
        match_end=match.end(),
        evidence_quote=_evidence_quote(text, match.start(), match.end()),
        rule=rule,
        proposed_action=proposed_action,
    )


def _evidence_quote(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    if line_end - line_start > 500:
        line_start = max(line_start, start - 250)
        line_end = min(line_end, line_start + 500)
        if line_end < end:
            line_end = end
            line_start = max(0, line_end - 500)
    return text[line_start:line_end].strip()


def _candidate_sort_key(item: SemanticCandidate) -> tuple[object, ...]:
    return (
        item.filename.casefold(),
        item.page_number,
        item.match_start,
        item.candidate_type.value,
        item.uid,
    )
