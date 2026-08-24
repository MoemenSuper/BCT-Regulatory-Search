import json

import pytest

from experiments.validation_candidate_audit import audit_validation_candidate


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _case(case_id, source, language, page=1):
    return {
        "id": case_id,
        "query": "Question",
        "language": language,
        "category": "test",
        "relevant": True,
        "expected_source": source,
        "expected_page": page,
        "evidence_quote": "required evidence text",
        "verification_status": "agent_curated_pending_independent_human_verification",
    }


def test_audit_accepts_bilingual_page_disjoint_candidate(tmp_path):
    development = [_case("dev-fr", "Dev_fr.pdf", "fr"), _case("dev-ar", "Dev_ar.pdf", "ar")]
    validation = [_case("val-fr", "Val_fr.pdf", "fr"), _case("val-ar", "Val_ar.pdf", "ar")]
    records = {}
    for source in ("Val_fr.pdf", "Val_ar.pdf"):
        artifact = _write(
            tmp_path / f"{source}.json",
            {"pages": [{"page_number": 1, "raw_text": "required evidence text"}]},
        )
        records[source] = {
            "source": source,
            "sha256": "A" * 64,
            "artifact": str(artifact),
        }
    result = audit_validation_candidate(
        development_path=_write(tmp_path / "development.json", development),
        validation_path=_write(tmp_path / "validation.json", validation),
        structured_manifest_path=_write(tmp_path / "manifest.json", {"records": records}),
    )

    assert result["status"].startswith("candidate_frozen")
    assert result["leakage_audit"]["exact_development_source_page_overlap"] == 0
    assert result["counts"]["relevant"] == 2


def test_audit_rejects_exact_development_page_overlap(tmp_path):
    cases = [_case("fr", "Same_fr.pdf", "fr"), _case("ar", "Same_ar.pdf", "ar")]
    records = {}
    for source in ("Same_fr.pdf", "Same_ar.pdf"):
        artifact = _write(
            tmp_path / f"{source}.json",
            {"pages": [{"page_number": 1, "raw_text": "required evidence text"}]},
        )
        records[source] = {"source": source, "sha256": "A" * 64, "artifact": str(artifact)}

    with pytest.raises(ValueError, match="occurs in development"):
        audit_validation_candidate(
            development_path=_write(tmp_path / "development.json", cases),
            validation_path=_write(tmp_path / "validation.json", cases),
            structured_manifest_path=_write(tmp_path / "manifest.json", {"records": records}),
        )
