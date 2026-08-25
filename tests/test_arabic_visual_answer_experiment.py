from experiments.arabic_visual_answer_experiment import prepare_routed_evidence
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


def test_invalid_or_incomplete_visual_fails_closed_for_the_whole_route():
    for visual in (_visual(valid=False), _visual(complete=False)):
        evidence, action = prepare_routed_evidence(
            record=_record(),
            route=_route(),
            visual_pages={("risky.pdf", 1): visual},
        )
        assert action == "fail_closed_visual_unavailable_invalid_or_uncertain"
        assert evidence == _record()["retrieved_evidence"]
