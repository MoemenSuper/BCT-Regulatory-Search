import json

from regulatory_graph.models import Language, VerificationStatus
from regulatory_graph.semantic_candidates import (
    CandidateType,
    PromotionEvidence,
    extract_page_candidates,
    validate_candidate_promotion,
    write_candidate_review_queue,
)


def _extract(text: str, *, language: Language = "fr"):
    return extract_page_candidates(
        text,
        filename="Cir_2020_03_fr.pdf",
        source_edition_uid="edition:cir-2020-03-fr",
        instrument_uid="BCT:CIRCULAR:2020:03",
        language=language,
        page_number=2,
        source_sha256="a" * 64,
        extraction_artifact_hash="b" * 64,
    )


def test_french_candidates_keep_exact_source_offsets_and_never_auto_verify():
    text = (
        "Article 2 : Il est ajouté à l'article 25 de la circulaire n° 2016-08 "
        "un nouveau paragraphe.\nArticle 3 : Le texte antérieur est abrogé."
    )

    candidates = _extract(text)

    assert {item.candidate_type for item in candidates} >= {
        CandidateType.PROVISION_HEADING,
        CandidateType.LEGAL_ACTION,
        CandidateType.CROSS_REFERENCE,
    }
    assert {item.proposed_action for item in candidates if item.proposed_action} >= {
        "ADD",
        "ABROGATE",
    }
    assert any(
        item.candidate_type == CandidateType.CROSS_REFERENCE
        and item.signal.casefold() == "article 25"
        for item in candidates
    )
    assert all(item.verification_status == VerificationStatus.NEEDS_REVIEW for item in candidates)
    assert all(item.evidence_quote in text for item in candidates)
    assert all(text[item.match_start:item.match_end] == item.signal for item in candidates)


def test_arabic_heading_and_action_are_detected_without_digit_normalization():
    text = "الفصل ٢ : يضاف إلى المنشور عدد ٦ لسنة ٢٠٢٢ حكم جديد."

    candidates = _extract(text, language="ar")

    assert any(item.candidate_type == CandidateType.PROVISION_HEADING for item in candidates)
    action = next(item for item in candidates if item.candidate_type == CandidateType.LEGAL_ACTION)
    assert action.proposed_action == "ADD"
    assert action.signal == "يضاف"
    assert "٢" in next(
        item.signal
        for item in candidates
        if item.candidate_type == CandidateType.PROVISION_HEADING
    )


def test_candidate_uids_and_jsonl_queue_are_deterministic(tmp_path):
    text = "Article premier : La circulaire n° 2019-07 est modifiée."
    first = _extract(text)
    second = _extract(text)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    first_receipt = write_candidate_review_queue(first, first_path)
    second_receipt = write_candidate_review_queue(reversed(second), second_path)

    assert tuple(item.uid for item in first) == tuple(item.uid for item in second)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_receipt == second_receipt
    rows = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]
    assert all(row["verification_status"] == "NEEDS_REVIEW" for row in rows)


def test_promotion_validator_fails_closed_and_names_missing_checks():
    candidate = next(
        item
        for item in _extract("Article 2 : La circulaire nÂ° 2019-07 est remplacÃ©e.")
        if item.candidate_type == CandidateType.LEGAL_ACTION
    )

    decision = validate_candidate_promotion(
        candidate,
        PromotionEvidence(
            source_hash_matches=True,
            page_matches=True,
            visual_confirmation=False,
            target_identity_unambiguous=False,
            provision_scope_unambiguous=True,
            affected_span_explicit=True,
            effective_state_recorded=False,
            predecessor_text_verified=False,
        ),
    )

    assert decision.status == VerificationStatus.NEEDS_REVIEW
    assert decision.reasons == (
        "visual_confirmation_missing",
        "target_identity_ambiguous",
        "effective_state_missing",
        "predecessor_text_unverified",
    )


def test_promotion_validator_only_verifies_complete_source_evidence():
    candidate = next(
        item
        for item in _extract("Article 2 : La circulaire nÂ° 2019-07 est remplacÃ©e.")
        if item.candidate_type == CandidateType.LEGAL_ACTION
    )

    decision = validate_candidate_promotion(
        candidate,
        PromotionEvidence(
            source_hash_matches=True,
            page_matches=True,
            visual_confirmation=True,
            target_identity_unambiguous=True,
            provision_scope_unambiguous=True,
            affected_span_explicit=True,
            effective_state_recorded=True,
            predecessor_text_verified=True,
        ),
    )

    assert decision.status == VerificationStatus.VERIFIED
    assert decision.reasons == ()
