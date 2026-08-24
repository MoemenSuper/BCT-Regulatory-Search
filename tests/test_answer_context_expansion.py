import json

import pytest

from experiments.answer_context_expansion import (
    build_context_expansion_suite,
    evidence_alphanumeric_identifiers,
    render_structured_page,
)


def test_render_structured_page_preserves_order_and_block_roles():
    page = {
        "page_number": 3,
        "blocks": [
            {"type": "heading", "text": "Ceramic importers"},
            {"type": "table", "text": "| CASA NOVA | 1605423P |"},
        ],
    }

    assert render_structured_page(page) == (
        "[HEADING]\nCeramic importers\n\n[TABLE]\n| CASA NOVA | 1605423P |"
    )


def test_evidence_identifier_inventory_uses_evidence_not_expected_answer():
    assert evidence_alphanumeric_identifiers(
        "CASA NOVA 1605423P, amount 100.000 DT and article 14"
    ) == ["1605423P"]


def test_build_context_suite_changes_only_selected_evidence(tmp_path):
    document_path = tmp_path / "document.json"
    document_path.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_number": 3,
                        "blocks": [
                            {"type": "heading", "text": "Ceramic importers"},
                            {"type": "table", "text": "CASA NOVA 1605423P"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    case = {
        "id": "case-1",
        "query": "Is CASA NOVA listed?",
        "language": "fr",
        "relevant": True,
        "expected_source": "Note.pdf",
        "expected_page": 3,
        "expected_answer": "NEVER COPY THIS INTO EVIDENCE",
        "evidence_quote": "CASA NOVA 1605423P",
        "answer_suite_role": "identifier",
    }
    suite = {"cases": [case, {**case, "id": "case-2"}]}
    manifest = {
        "records": {
            "folder/Note.pdf": {
                "source": "Note.pdf",
                "sha256": "A" * 64,
                "artifact": str(document_path),
            }
        }
    }

    result = build_context_expansion_suite(
        base_suite=suite,
        manifest=manifest,
        selected_ids={"case-1"},
        base_suite_sha256="B" * 64,
        manifest_sha256="C" * 64,
        prompt_version="bct-claim-linked-answer-v4",
    )

    assert [item["id"] for item in result["cases"]] == ["case-1"]
    expanded = result["cases"][0]
    assert expanded["query"] == case["query"]
    assert expanded["expected_answer"] == case["expected_answer"]
    assert "NEVER COPY" not in expanded["evidence_quote"]
    assert expanded["evidence_quote"].startswith("[HEADING]\nCeramic importers")
    assert expanded["context_expansion"]["block_count"] == 2
    assert result["answer_experiment"]["prompt_version"] == (
        "bct-claim-linked-answer-v4"
    )


def test_build_context_suite_rejects_unknown_case():
    with pytest.raises(ValueError, match="absent"):
        build_context_expansion_suite(
            base_suite={"cases": []},
            manifest={"records": {}},
            selected_ids={"missing"},
            base_suite_sha256="B" * 64,
            manifest_sha256="C" * 64,
        )


def test_full_relevant_suite_preserves_verified_excerpt_before_page_context(tmp_path):
    document_path = tmp_path / "document.json"
    document_path.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_number": 1,
                        "blocks": [{"type": "text", "text": "Full page condition"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    case = {
        "id": "case-1",
        "query": "Question",
        "language": "fr",
        "relevant": True,
        "expected_source": "Cir.pdf",
        "expected_page": 1,
        "expected_answer": "Answer",
        "evidence_quote": "Verified exact wording",
        "answer_suite_role": "control",
    }
    result = build_context_expansion_suite(
        base_suite={"cases": [case]},
        manifest={
            "records": {
                "Cir.pdf": {
                    "source": "Cir.pdf",
                    "sha256": "A" * 64,
                    "artifact": str(document_path),
                }
            }
        },
        selected_ids={"case-1"},
        base_suite_sha256="B" * 64,
        manifest_sha256="C" * 64,
        include_verified_excerpt=True,
    )

    assert result["suite_type"] == "full_relevant_gold_evidence_context_development"
    assert result["answer_experiment"]["experiment_id"] == (
        "claim-linked-full-page-relevant-development-v1"
    )
    assert result["cases"][0]["evidence_quote"] == (
        "[VERIFIED EXCERPT]\nVerified exact wording\n\n"
        "[FULL LABELED PAGE]\n[TEXT]\nFull page condition"
    )


def test_v5_context_suite_freezes_identifiers_from_verified_excerpt(tmp_path):
    document_path = tmp_path / "document.json"
    document_path.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_number": 3,
                        "blocks": [{"type": "table", "text": "CASA NOVA 1605423P"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    case = {
        "id": "case-1",
        "query": "Is CASA NOVA listed?",
        "language": "ar",
        "relevant": True,
        "expected_source": "Note.pdf",
        "expected_page": 3,
        "expected_answer": "Expected text is not consulted.",
        "evidence_quote": "CASA NOVA 1605423P",
        "answer_suite_role": "identifier",
    }

    result = build_context_expansion_suite(
        base_suite={"cases": [case]},
        manifest={
            "records": {
                "Note.pdf": {
                    "source": "Note.pdf",
                    "sha256": "A" * 64,
                    "artifact": str(document_path),
                }
            }
        },
        selected_ids={"case-1"},
        base_suite_sha256="B" * 64,
        manifest_sha256="C" * 64,
        prompt_version="bct-claim-linked-answer-v5",
    )

    assert result["cases"][0]["required_answer_literals"] == ["1605423P"]
