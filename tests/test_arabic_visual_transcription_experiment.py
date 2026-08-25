import copy

import pytest

from experiments.arabic_visual_fallback import MODEL_ID, PROMPT_VERSION
from experiments.arabic_visual_transcription_experiment import (
    frozen_routed_pages,
    validate_cached_page,
)


def _routing_receipt():
    return {
        "status": "frozen_before_visual_calls",
        "policy": {
            "gold_fields_used_for_routing": [],
            "max_visual_pages_per_query": 2,
        },
        "counts": {"unique_pages": 2},
        "routes": [
            {
                "id": "one",
                "pages": [
                    {"source": "A.pdf", "page": 1},
                    {"source": "A.pdf", "page": 1},
                ],
            },
            {"id": "two", "pages": [{"source": "B.pdf", "page": 2}]},
        ],
    }


def _response():
    return {
        "transcription": "31 August 2018",
        "items": [
            {
                "literal": "31",
                "kind": "date",
                "context": "31 August 2018",
                "uncertain": False,
            }
        ],
        "uncertain_regions": [],
        "complete": True,
    }


def test_frozen_pages_are_deduplicated_and_budget_bound():
    assert frozen_routed_pages(_routing_receipt()) == [
        {"source": "A.pdf", "page": 1},
        {"source": "B.pdf", "page": 2},
    ]

    over_budget = _routing_receipt()
    over_budget["routes"][0]["pages"].append({"source": "C.pdf", "page": 3})
    with pytest.raises(ValueError, match="exceeds visual page budget"):
        frozen_routed_pages(over_budget)


def test_cached_page_binding_and_invalid_receipt_fail_closed():
    cache = {
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "source_pdf_sha256": "A" * 64,
        "page": 1,
        "image_sha256": "B" * 64,
        "validation_status": "valid",
        "response": _response(),
    }
    validate_cached_page(
        cache,
        source_pdf_sha256="A" * 64,
        page=1,
        image_sha256="B" * 64,
    )
    drifted = copy.deepcopy(cache)
    drifted["image_sha256"] = "C" * 64
    with pytest.raises(ValueError, match="binding differs"):
        validate_cached_page(
            drifted,
            source_pdf_sha256="A" * 64,
            page=1,
            image_sha256="B" * 64,
        )

    invalid = {**cache, "validation_status": "invalid"}
    invalid.pop("response")
    invalid["validation_error"] = "malformed"
    invalid["raw_response_sha256"] = "D" * 64
    validate_cached_page(
        invalid,
        source_pdf_sha256="A" * 64,
        page=1,
        image_sha256="B" * 64,
    )
    invalid.pop("raw_response_sha256")
    with pytest.raises(ValueError, match="lacks its audit receipt"):
        validate_cached_page(
            invalid,
            source_pdf_sha256="A" * 64,
            page=1,
            image_sha256="B" * 64,
        )
