import json

import pytest

from experiments.final_voyage_qualification import FINAL_SYSTEM_PROMPT, _retrieve, runtime_cases


def test_runtime_cases_remove_all_gold_fields():
    cases = [
        {
            "id": "q1",
            "query": "question",
            "language": "fr",
            "category": "rule",
            "expected_source": "secret.pdf",
            "expected_page": 4,
            "expected_answer": "gold",
            "expected_sources": [{"source": "secret.pdf", "pages": [4]}],
        }
    ]

    assert runtime_cases(cases) == [
        {"id": "q1", "query": "question", "language": "fr", "category": "rule"}
    ]


def test_final_prompt_requires_safe_ambiguity_and_literal_handling():
    assert "clarification_needed" in FINAL_SYSTEM_PROMPT
    assert "current legal status" in FINAL_SYSTEM_PROMPT
    assert "Never repair or infer a number or date from a filename" in FINAL_SYSTEM_PROMPT
    assert "different instruments" in FINAL_SYSTEM_PROMPT


def test_final_runner_exposes_visual_composition_mode():
    from pathlib import Path
    source = Path(__import__("experiments.final_voyage_qualification", fromlist=["__file__"]).__file__).read_text(encoding="utf-8")
    assert "compose_visual(args) if args.visual_result else run(args)" in source


def test_retrieval_rejects_stale_checkpoint_before_provider_work(tmp_path):
    native = tmp_path / "native.jsonl"
    ocr = tmp_path / "ocr.jsonl"
    native.write_text("native", encoding="utf-8")
    ocr.write_text("ocr", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"binding": {"suite_sha256": "old"}, "records": []}), encoding="utf-8")
    matrix = type(
        "MatrixStub",
        (),
        {"native_chunks": native, "ocr_chunks": ocr},
    )()

    with pytest.raises(ValueError, match="binding differs"):
        _retrieve(
            matrix=matrix,
            cases=[],
            suite_hash="new",
            checkpoint_path=checkpoint,
        )
