from __future__ import annotations

import hashlib
import re
from pathlib import Path

from retrieval_selection import parse_query_identity

from .identity import instrument_from_filename, instrument_from_properties, instrument_from_text
from .models import RelationshipCandidate, RelationshipType, VerificationStatus


_ACTION_PATTERNS = {
    RelationshipType.AMENDS: re.compile(
        r"\b(?:modifi(?:e|ent|é|ée|és|ées|ant|cation)|amend(?:s|ed|ing)?)\b|"
        r"(?:يعدل|تعدل|يعدّل|تعدّل|ينقح|تنقح|تعديل|تنقيح)",
        re.IGNORECASE,
    ),
    RelationshipType.REPLACES: re.compile(
        r"\b(?:remplac(?:e|ent|é|ée|és|ées|ement)|replace(?:s|d|ment)?|supersed(?:e|es|ed))\b|"
        r"(?:يعوض|تعوض|يستبدل|تستبدل|تعويض|استبدال)",
        re.IGNORECASE,
    ),
    RelationshipType.ABROGATES: re.compile(
        r"\b(?:abrog(?:e|ent|é|ée|és|ées|ation)|repeal(?:s|ed)?)\b|"
        r"(?:يلغي|تلغي|إلغاء|الغاء)",
        re.IGNORECASE,
    ),
}


def _recover_exact_quote(source_text: str, proposed_quote: str) -> str | None:
    """Recover exact source characters while tolerating whitespace normalization by the LLM."""
    proposed = proposed_quote.strip()
    if not proposed:
        return None

    tokens = proposed.split()
    if len(tokens) >= 3 and "..." not in proposed and "…" not in proposed:
        pattern = r"\s+".join(re.escape(token) for token in tokens)
        match = re.search(pattern, source_text, re.DOTALL)
        if match:
            return match.group(0)

    # When the model uses an ellipsis, match each kept segment tightly and allow
    # only a bounded gap between segments.
    parts = [part.strip() for part in re.split(r"(?:\.\.\.|…)", proposed) if part.strip()]
    segments = []
    token_count = 0
    for part in parts:
        part_tokens = part.split()
        if not part_tokens:
            continue
        token_count += len(part_tokens)
        segments.append(r"\s+".join(re.escape(token) for token in part_tokens))
    if token_count < 3 or not segments:
        return None
    pattern = r".{0,80}?".join(segments)
    match = re.search(pattern, source_text, re.DOTALL)
    return match.group(0) if match else None


def _quote_names_target(quote: str, target_id: str) -> bool:
    parsed = parse_query_identity(quote)
    if parsed and parsed.get("kind") and parsed.get("number") is not None:
        ref = instrument_from_text(quote)
        if ref is not None and ref.id == target_id:
            return True

    # Conservative fallback: require a BCT-instrument marker plus both year and number.
    _kind, year, number = target_id.split(":", 2)
    marker = re.search(
        r"\b(?:circulair(?:e|es)|circulars?|notes?)\b|(?:المنشور|منشور|المذكرة|مذكرة)",
        quote,
        re.IGNORECASE,
    )
    return bool(
        marker
        and re.search(rf"(?<!\d){re.escape(year)}(?!\d)", quote)
        and re.search(rf"(?<!\d)0*{re.escape(str(int(number)))}(?!\d)", quote)
    )


def _provision_label(node: dict[str, object]) -> str | None:
    if str(node.get("label", "")).casefold() != "provision":
        return None
    properties = dict(node.get("properties") or {})
    value = properties.get("label") or properties.get("name")
    text = str(value or "").strip()
    return text or None


def _instrument_for_node(node: dict[str, object]):
    label = str(node.get("label", "")).casefold()
    properties = dict(node.get("properties") or {})
    if label == "instrument":
        return instrument_from_properties(properties)
    if label == "provision":
        instrument_text = str(properties.get("instrument") or "").strip()
        return instrument_from_text(instrument_text) if instrument_text else None
    return None


def _candidate_id(
    relation: RelationshipType,
    source_id: str,
    target_id: str,
    source_provision: str | None,
    target_provision: str | None,
    evidence_file: str,
    evidence_page: int,
    evidence_quote: str,
) -> str:
    value = "\x1f".join(
        [
            relation.value,
            source_id,
            target_id,
            source_provision or "",
            target_provision or "",
            Path(evidence_file).name.casefold(),
            str(evidence_page),
            " ".join(evidence_quote.split()),
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def validate_extracted_relationship(
    *,
    source_file: str,
    source_page: int,
    source_text: str,
    nodes_by_id: dict[str, dict[str, object]],
    relationship: dict[str, object],
    catalog: dict,
) -> RelationshipCandidate | None:
    """Convert one builder edge into a bounded candidate or reject it.

    The current PDF is trusted as the actor.  The LLM is not allowed to invent a
    different source instrument, a target absent from the extracted endpoint, or
    evidence text that cannot be recovered from the source page.
    """
    try:
        relation = RelationshipType(str(relationship.get("type", "")).upper())
    except ValueError:
        return None

    source_ref = instrument_from_filename(source_file)
    if source_ref is None or source_page < 1:
        return None

    start = nodes_by_id.get(str(relationship.get("start_node_id", "")))
    end = nodes_by_id.get(str(relationship.get("end_node_id", "")))
    if start is None or end is None:
        return None

    extracted_source = _instrument_for_node(start)
    target_ref = _instrument_for_node(end)
    if extracted_source is None or target_ref is None:
        return None

    # Direction is intentionally strict: the newly ingested source document acts on/cites the target.
    if extracted_source.id != source_ref.id or target_ref.id == source_ref.id:
        return None

    properties = dict(relationship.get("properties") or {})
    proposed_quote = str(properties.get("evidence_quote") or "").strip()
    exact_quote = _recover_exact_quote(source_text, proposed_quote)
    if exact_quote is None or not _quote_names_target(exact_quote, target_ref.id):
        return None

    if relation in _ACTION_PATTERNS and not _ACTION_PATTERNS[relation].search(exact_quote):
        return None

    source_provision = _provision_label(start)
    target_provision = _provision_label(end)
    if (source_provision is None) != (target_provision is None):
        return None

    # Graph Lite is fully automatic. An extracted target must resolve to a real
    # instrument in the trusted corpus catalog; otherwise the edge is discarded
    # rather than parked for human review.
    target_in_catalog = target_ref.id in catalog
    if not target_in_catalog:
        return None
    trusted = catalog[target_ref.id]
    target_ref = type(target_ref)(
        id=target_ref.id,
        kind=target_ref.kind,
        year=target_ref.year,
        number=target_ref.number,
        filename=trusted.filename,
    )

    # Reaching this point means the deterministic evidence gate passed: the
    # source is the PDF being ingested, the target exists in the trusted corpus,
    # the exact quote is recoverable from this physical page, it names that
    # target, and legal-change edges contain the matching action wording.
    status = VerificationStatus.VERIFIED
    effective_date = str(properties.get("effective_date") or "").strip() or None
    candidate_id = _candidate_id(
        relation,
        source_ref.id,
        target_ref.id,
        source_provision,
        target_provision,
        source_file,
        source_page,
        exact_quote,
    )
    return RelationshipCandidate(
        candidate_id=candidate_id,
        relation=relation,
        source=source_ref,
        target=target_ref,
        source_provision=source_provision,
        target_provision=target_provision,
        evidence_file=Path(source_file).name,
        evidence_page=int(source_page),
        evidence_quote=exact_quote,
        proposed_effective_date=effective_date,
        verification_status=status,
        target_in_catalog=target_in_catalog,
        verification_method="AUTO_DETERMINISTIC_V1",
    )
