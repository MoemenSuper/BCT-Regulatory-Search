import json

import pytest

from experiments.gemini_visual_transcription_experiment import (
    MODEL_ID,
    PROMPT_VERSION,
    build_generate_content_payload,
    extract_generate_content_response,
    validate_cached_page,
)


def _visual_response():
    return {
        "transcription": "يجب الإعلام قبل 24 ساعة.",
        "items": [
            {
                "literal": "24",
                "kind": "time",
                "context": "قبل 24 ساعة",
                "uncertain": False,
            }
        ],
        "uncertain_regions": [],
        "complete": True,
    }


def test_generate_content_response_is_locally_validated_and_usage_is_preserved():
    body = {
        "responseId": "response-1",
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": json.dumps(_visual_response(), ensure_ascii=False)}
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 5,
            "totalTokenCount": 125,
        },
    }
    parsed = extract_generate_content_response(body)
    assert parsed["response"] == _visual_response()
    assert parsed["response_id"] == "response-1"
    assert parsed["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "thinking_tokens": 5,
        "total_tokens": 125,
    }


def test_missing_text_response_fails_closed():
    with pytest.raises(ValueError, match="text part"):
        extract_generate_content_response({"candidates": []})


def test_rest_payload_uses_live_json_mime_enum():
    payload = build_generate_content_payload(b"png")
    assert (
        payload["generationConfig"]["responseFormat"]["text"]["mimeType"]
        == "APPLICATION_JSON"
    )


def test_gemini_cache_is_bound_to_provider_model_prompt_pdf_page_and_image():
    cache = {
        "provider": "Google Gemini API",
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "source_pdf_sha256": "A" * 64,
        "page": 2,
        "image_sha256": "B" * 64,
        "validation_status": "valid",
        "response": _visual_response(),
    }
    validate_cached_page(
        cache,
        source_pdf_sha256="A" * 64,
        page=2,
        image_sha256="B" * 64,
    )
    cache["model"] = "different-model"
    with pytest.raises(ValueError, match="binding differs"):
        validate_cached_page(
            cache,
            source_pdf_sha256="A" * 64,
            page=2,
            image_sha256="B" * 64,
        )
