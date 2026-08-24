import json

import pytest
from langchain_core.documents import Document

from experiments.artifacts import sha256_file
from experiments.provisional_validation_retrieval import (
    _failure_categories,
    load_frozen_validation,
)


def test_load_frozen_validation_verifies_dataset_hash(tmp_path):
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            [
                {
                    "id": "fr",
                    "query": "Question",
                    "language": "fr",
                    "relevant": True,
                    "expected_source": "Cir_2026_01_fr.pdf",
                    "expected_page": 1,
                    "evidence_quote": "Evidence",
                },
                {
                    "id": "ar",
                    "query": "سؤال",
                    "language": "ar",
                    "relevant": True,
                    "expected_source": "Cir_2026_02_ar.pdf",
                    "expected_page": 1,
                    "evidence_quote": "دليل",
                },
            ]
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "sets": {
                    "validation": {
                        "path": str(validation),
                        "sha256": sha256_file(validation),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cases, frozen = load_frozen_validation(protocol_path=protocol)

    assert len(cases) == 2
    assert frozen["sha256"] == sha256_file(validation)

    validation.write_text(validation.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_frozen_validation(protocol_path=protocol)


def test_failure_category_separates_candidate_and_ranking_failures():
    case = {
        "relevant": True,
        "expected_source": "Expected.pdf",
        "expected_page": 3,
    }
    wrong_source = {
        "document": Document(page_content="", metadata={"source": "Other.pdf", "page": 3})
    }
    wrong_page = {
        "document": Document(
            page_content="", metadata={"source": "Expected.pdf", "page": 2}
        )
    }
    right_page = {
        "document": Document(
            page_content="", metadata={"source": "Expected.pdf", "page": 3}
        )
    }

    assert _failure_categories(case, [wrong_source], None) == [
        "correct_document_missing_from_candidate_set"
    ]
    assert _failure_categories(case, [wrong_page], None) == [
        "correct_page_missing_from_candidate_set"
    ]
    assert _failure_categories(case, [right_page], 8) == [
        "correct_page_reranked_below_top_5"
    ]
    assert _failure_categories(case, [right_page], 5) == []
