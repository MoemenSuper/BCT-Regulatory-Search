import copy
import hashlib
import json

import pytest

from experiments.arabic_visual_fallback import (
    MODEL_ID,
    PROMPT_VERSION,
    build_routing_receipt,
    parse_visual_payload,
    route_visual_pages,
    validate_visual_cache_binding,
    visual_evidence,
)


def _suite():
    return {
        "cases": [
            {
                "id": "answered",
                "query": "متى بدأ التداول سنة 2018؟",
                "language": "ar",
                "expected_source": "GOLD.pdf",
                "expected_page": 99,
                "expected_answer": "GOLD",
            },
            {
                "id": "abstained",
                "query": "ما أجل السداد؟",
                "language": "ar",
                "expected_source": "OTHER-GOLD.pdf",
                "expected_page": 88,
            },
            {
                "id": "clarify",
                "query": "ما مبلغ المنحة؟",
                "language": "ar",
            },
            {
                "id": "future",
                "query": "ما شروط المنشور الذي سيصدر سنة 2027؟",
                "language": "ar",
            },
            {
                "id": "non_numeric_answer",
                "query": "ما شرط تمويل الاستيراد بالدينار التونسي؟",
                "language": "ar",
            },
        ]
    }


def _retrieved_result():
    def evidence(evidence_id, source, page):
        return {
            "evidence_id": evidence_id,
            "source": source,
            "page": page,
            "text": "native 8102",
            "representations": ["native"],
        }

    return {
        "records": [
            {
                "id": "answered",
                "retrieved_evidence": [
                    evidence("E1", "Risky.pdf", 1),
                    evidence("E2", "Risky.pdf", 2),
                ],
                "response": {
                    "status": "answered",
                    "citations": [{"evidence_id": "E2"}],
                },
            },
            {
                "id": "abstained",
                "retrieved_evidence": [
                    evidence("E1", "Risky.pdf", 3),
                    evidence("E2", "Risky.pdf", 4),
                    evidence("E3", "Risky.pdf", 5),
                ],
                "response": {"status": "insufficient_evidence", "citations": []},
            },
            {
                "id": "clarify",
                "retrieved_evidence": [evidence("E1", "Risky.pdf", 6)],
                "response": {"status": "clarification_needed", "citations": []},
            },
            {
                "id": "future",
                "retrieved_evidence": [evidence("E1", "Risky.pdf", 7)],
                "response": {"status": "insufficient_evidence", "citations": []},
            },
            {
                "id": "non_numeric_answer",
                "retrieved_evidence": [evidence("E1", "Risky.pdf", 8)],
                "response": {
                    "status": "answered",
                    "answer": "يجب أن تكون السلع ذات منشأ تونسي.",
                    "claims": [{"text": "السلع ذات منشأ تونسي."}],
                    "citations": [{"evidence_id": "E1"}],
                },
            },
        ]
    }


def _risk_result():
    return {
        "records": [
            {"source": "Risky.pdf", "requires_visual_fallback": True}
        ]
    }


def _visual_response(*, complete=True, uncertain=False):
    return {
        "transcription": "بدأ التداول سنة 2018 في 31 أوت 2018.",
        "items": [
            {
                "literal": "2018",
                "kind": "date",
                "context": "سنة 2018",
                "uncertain": uncertain,
            }
        ],
        "uncertain_regions": ["التاريخ"] if uncertain else [],
        "complete": complete,
    }


def test_routing_is_gold_blind_and_enforces_page_budget():
    suite = _suite()
    routes = route_visual_pages(
        suite=suite,
        retrieved_result=_retrieved_result(),
        risk_result=_risk_result(),
    )
    mutated = copy.deepcopy(suite)
    for case in mutated["cases"]:
        case["expected_source"] = "MUTATED.pdf"
        case["expected_page"] = -1
        case["expected_answer"] = "MUTATED"

    assert route_visual_pages(
        suite=mutated,
        retrieved_result=_retrieved_result(),
        risk_result=_risk_result(),
    ) == routes
    assert [route["id"] for route in routes] == ["answered", "abstained"]
    assert routes[0]["pages"] == [
        {
            "evidence_id": "E2",
            "source": "Risky.pdf",
            "page": 2,
            "reason": "answered_numeric_or_date_query_cited_risky_document",
        }
    ]
    assert [page["page"] for page in routes[1]["pages"]] == [3, 4]

    receipt = build_routing_receipt(
        suite=suite,
        retrieved_result=_retrieved_result(),
        risk_result=_risk_result(),
        input_hashes={"suite": "A" * 64},
    )
    assert receipt["counts"] == {
        "routed_cases": 2,
        "routed_pages": 3,
        "unique_pages": 3,
    }
    assert receipt["policy"]["gold_fields_used_for_routing"] == []


def test_visual_payload_requires_exact_contract_and_literal_traceability():
    assert parse_visual_payload(json.dumps(_visual_response()))["complete"] is True

    invalid = _visual_response()
    invalid["items"][0]["literal"] = "2020"
    with pytest.raises(ValueError, match="absent from transcription"):
        parse_visual_payload(json.dumps(invalid))


def test_visual_cache_binding_fails_closed():
    cache = {
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "source_pdf_sha256": "A" * 64,
        "page": 1,
        "image_sha256": "B" * 64,
        "response": _visual_response(),
    }
    validate_visual_cache_binding(
        cache,
        source_pdf_sha256="A" * 64,
        page=1,
        image_sha256="B" * 64,
    )
    cache["page"] = 2
    with pytest.raises(ValueError, match="binding differs"):
        validate_visual_cache_binding(
            cache,
            source_pdf_sha256="A" * 64,
            page=1,
            image_sha256="B" * 64,
        )


def test_uncertain_visual_output_is_not_usable_and_conflicts_are_audited():
    native = {
        "evidence_id": "E1",
        "source": "Risky.pdf",
        "page": 1,
        "text": "بدأ التداول سنة 8102.",
        "representations": ["native"],
    }
    cache = {
        "provider": "Google Gemini API",
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "source_pdf_sha256": "A" * 64,
        "image_sha256": "B" * 64,
        "response": _visual_response(uncertain=True),
    }
    assert visual_evidence(native, cache) is None

    cache["response"] = _visual_response()
    evidence = visual_evidence(native, cache)
    assert evidence is not None
    assert evidence["text"] == cache["response"]["transcription"]
    assert "native 8102" not in evidence["text"]
    assert evidence["visual_verification"]["numeric_conflict"] is True
    assert evidence["visual_verification"]["provider"] == "Google Gemini API"
    assert evidence["visual_verification"]["native_only_numbers"] == ["8102"]
    assert evidence["visual_verification"]["visual_only_numbers"] == ["2018", "31"]
    assert evidence["visual_verification"]["native_text_sha256"] == hashlib.sha256(
        native["text"].encode("utf-8")
    ).hexdigest().upper()
