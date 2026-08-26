from experiments.arabic_visual_answer_experiment import (
    enforce_routed_numeric_authority,
    prepare_routed_evidence,
    validate_visual_result_for_composition,
)
from experiments.arabic_visual_fallback import MODEL_ID, PROMPT_VERSION


def _record():
    return {
        "id": "case",
        "retrieved_evidence": [
            {
                "evidence_id": "E1",
                "source": "Risky.pdf",
                "page": 1,
                "text": "native 8102",
                "representations": ["native"],
            },
            {
                "evidence_id": "E2",
                "source": "Control.pdf",
                "page": 2,
                "text": "unchanged",
                "representations": ["native"],
            },
        ],
    }


def _route():
    return {
        "id": "case",
        "pages": [{"evidence_id": "E1", "source": "Risky.pdf", "page": 1}],
    }


def _visual(*, valid=True, complete=True):
    return {
        "source": "Risky.pdf",
        "page": 1,
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "source_pdf_sha256": "A" * 64,
        "image_sha256": "B" * 64,
        "validation_status": "valid" if valid else "invalid",
        "response": {
            "transcription": "visual 2018",
            "items": [
                {
                    "literal": "2018",
                    "kind": "date",
                    "context": "visual 2018",
                    "uncertain": False,
                }
            ],
            "uncertain_regions": [],
            "complete": complete,
        },
    }


def test_valid_visual_replaces_only_selected_evidence_without_concatenation():
    visual = _visual()
    evidence, action = prepare_routed_evidence(
        record=_record(),
        route=_route(),
        visual_pages={("risky.pdf", 1): visual},
    )
    assert action == "generate"
    assert evidence[0]["text"] == "visual 2018"
    assert "native 8102" not in evidence[0]["text"]
    assert evidence[0]["visual_verification"]["numeric_conflict"] is True
    assert evidence[1]["text"] == "unchanged"


def test_partial_provider_result_reaches_per_route_fail_closed_handling():
    validate_visual_result_for_composition(
        {"status": "partial_provider_unavailable", "pages": []}
    )


def test_invalid_or_incomplete_visual_fails_closed_for_the_whole_route():
    for visual in (_visual(valid=False), _visual(complete=False)):
        evidence, action = prepare_routed_evidence(
            record=_record(),
            route=_route(),
            visual_pages={("risky.pdf", 1): visual},
        )
        assert action == "fail_closed_visual_unavailable_invalid_or_uncertain"
        assert evidence == _record()["retrieved_evidence"]


def test_routed_numeric_authority_rejects_silent_digit_normalization():
    evidence = _record()["retrieved_evidence"]
    evidence[0]["text"] = "visual 7 July 2016"
    evidence[0]["visual_verification"] = {"provider": "Google Gemini API"}
    response = {
        "status": "answered",
        "answer": "10 July 2016",
        "claims": [{"text": "10 July 2016", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1", "source": "Risky.pdf", "page": 1}],
    }

    guarded, action = enforce_routed_numeric_authority(
        response=response,
        evidence=evidence,
        routed_evidence_ids={"E1"},
        language="ar",
    )

    assert action == "fail_closed_unsupported_numeric_literal"
    assert guarded["status"] == "insufficient_evidence"
    assert guarded["claims"] == []
    assert guarded["citations"] == []


def test_routed_numeric_authority_requires_validated_visual_provenance():
    evidence = _record()["retrieved_evidence"]
    evidence[0]["text"] = "native 7 July 2016"
    response = {
        "status": "answered",
        "answer": "7 July 2016",
        "claims": [{"text": "7 July 2016", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1", "source": "Risky.pdf", "page": 1}],
    }

    guarded, action = enforce_routed_numeric_authority(
        response=response,
        evidence=evidence,
        routed_evidence_ids={"E1"},
        language="ar",
    )

    assert action == "fail_closed_missing_visual_authority"
    assert guarded["status"] == "insufficient_evidence"


def test_routed_numeric_authority_accepts_supported_visual_literals():
    evidence = _record()["retrieved_evidence"]
    evidence[0]["text"] = "visual 7 July 2016"
    evidence[0]["visual_verification"] = {"provider": "Google Gemini API"}
    response = {
        "status": "answered",
        "answer": "7 July 2016",
        "claims": [{"text": "7 July 2016", "evidence_ids": ["E1"]}],
        "citations": [{"evidence_id": "E1", "source": "Risky.pdf", "page": 1}],
    }

    guarded, action = enforce_routed_numeric_authority(
        response=response,
        evidence=evidence,
        routed_evidence_ids={"E1"},
        language="ar",
    )

    assert guarded is response
    assert action == "generate"
