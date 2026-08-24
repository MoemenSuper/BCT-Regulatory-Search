import json

import pytest

from experiments.artifacts import sha256_file
from experiments.validation_review_packet import render_review_packet


def test_review_packet_is_hash_bound_and_contains_no_model_output(tmp_path):
    validation_path = tmp_path / "validation.json"
    validation = [
        {
            "id": "case-1",
            "language": "fr",
            "category": "scope",
            "relevant": True,
            "expected_source": "Cir.pdf",
            "expected_page": 2,
            "query": "Question?",
            "expected_answer": "Answer",
            "evidence_quote": "Evidence",
        }
    ]
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit = {"inputs": {"validation_candidate_sha256": sha256_file(validation_path)}}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    packet = render_review_packet(
        validation_path=validation_path,
        audit_path=audit_path,
        validation=validation,
        audit=audit,
    )

    assert "Retrieval/model access: **not run**" in packet
    assert "Source: `Cir.pdf`, page `2`" in packet
    assert "model response" not in packet.casefold()


def test_review_packet_rejects_stale_audit(tmp_path):
    validation_path = tmp_path / "validation.json"
    validation_path.write_text("[]", encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")

    with pytest.raises((KeyError, ValueError)):
        render_review_packet(
            validation_path=validation_path,
            audit_path=audit_path,
            validation=[],
            audit={},
        )
