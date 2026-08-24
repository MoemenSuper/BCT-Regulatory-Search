import json

import pytest

from experiments.validation_candidate_audit import audit_validation_candidate


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _case(case_id, source, language, page=1):
    return {
        "id": case_id,
        "query": f"Question {case_id}",
        "language": language,
        "category": "test",
        "relevant": True,
        "expected_source": source,
        "expected_page": page,
        "evidence_quote": f"required evidence text {case_id}",
        "verification_status": "agent_curated_pending_independent_human_verification",
    }


def test_audit_accepts_bilingual_page_disjoint_candidate(tmp_path):
    development = [_case("dev-fr", "Dev_fr.pdf", "fr"), _case("dev-ar", "Dev_ar.pdf", "ar")]
    validation = [_case("val-fr", "Val_fr.pdf", "fr"), _case("val-ar", "Val_ar.pdf", "ar")]
    records = {}
    for source in ("Val_fr.pdf", "Val_ar.pdf"):
        source_words = source.removesuffix(".pdf").replace("_", " ")
        artifact = _write(
            tmp_path / f"{source}.json",
            {"pages": [{"page_number": 1, "raw_text": f"required evidence text {source_words}"}]},
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


def test_audit_rejects_normalized_query_duplicate_on_different_pages(tmp_path):
    development = [_case("dev-fr", "Dev_fr.pdf", "fr"), _case("dev-ar", "Dev_ar.pdf", "ar")]
    validation = [_case("val-fr", "Val_fr.pdf", "fr"), _case("val-ar", "Val_ar.pdf", "ar")]
    validation[0]["query"] = development[0]["query"].upper()
    records = {}
    for source in ("Val_fr.pdf", "Val_ar.pdf"):
        artifact = _write(
            tmp_path / f"{source}.json",
            {"pages": [{"page_number": 1, "raw_text": "required evidence text"}]},
        )
        records[source] = {"source": source, "sha256": "A" * 64, "artifact": str(artifact)}

    with pytest.raises(ValueError, match="query duplicates"):
        audit_validation_candidate(
            development_path=_write(tmp_path / "development.json", development),
            validation_path=_write(tmp_path / "validation.json", validation),
            structured_manifest_path=_write(tmp_path / "manifest.json", {"records": records}),
        )


def _visual_audit_paths(tmp_path, visual_verification):
    development = [_case("dev-fr", "Dev_fr.pdf", "fr"), _case("dev-ar", "Dev_ar.pdf", "ar")]
    validation = [_case("val-fr", "Val_fr.pdf", "fr"), _case("val-ar", "Val_ar.pdf", "ar")]
    validation[1]["evidence_method"] = "visual_review"
    if visual_verification is not None:
        validation[1]["visual_verification"] = visual_verification
    records = {}
    for case in validation:
        source = case["expected_source"]
        raw_text = (
            "corrupted extraction"
            if source == "Val_ar.pdf"
            else case["evidence_quote"]
        )
        artifact = _write(
            tmp_path / f"{source}.json",
            {"pages": [{"page_number": 1, "raw_text": raw_text}]},
        )
        records[source] = {
            "source": source,
            "sha256": "B" * 64,
            "artifact": str(artifact),
        }
    return (
        _write(tmp_path / "development.json", development),
        _write(tmp_path / "validation.json", validation),
        _write(tmp_path / "manifest.json", {"records": records}),
    )


def test_audit_accepts_hashed_visual_verification_for_low_extracted_coverage(tmp_path):
    paths = _visual_audit_paths(tmp_path, {
        "rendered_page_sha256": "A" * 64,
        "note": "The rendered page visibly contains the complete evidence.",
    })
    result = audit_validation_candidate(
        development_path=paths[0],
        validation_path=paths[1],
        structured_manifest_path=paths[2],
    )

    assert result["evidence_audit"]["coverage_exception_case_ids"] == ["val-ar"]
    assert result["records"][1]["coverage_exception"] == (
        "verified_against_hashed_page_render"
    )


@pytest.mark.parametrize(
    "visual_verification",
    [
        None,
        {"rendered_page_sha256": "not-a-hash", "note": "Reviewed."},
        {"rendered_page_sha256": "A" * 64, "note": ""},
    ],
)
def test_audit_rejects_incomplete_visual_verification_for_low_coverage(
    tmp_path, visual_verification
):
    paths = _visual_audit_paths(tmp_path, visual_verification)

    with pytest.raises(ValueError, match="without a complete visual verification"):
        audit_validation_candidate(
            development_path=paths[0],
            validation_path=paths[1],
            structured_manifest_path=paths[2],
        )
