from experiments.targeted_stress_catalog import build_targeted_stress_catalog


def _case(case_id, language, category, source, *, relevant=True, method="text_extraction", quote="text"):
    return {
        "id": case_id,
        "language": language,
        "category": category,
        "expected_source": source,
        "expected_page": 1 if relevant else None,
        "relevant": relevant,
        "evidence_method": method,
        "evidence_quote": quote,
    }


def _record(case_id, rank=1, failures=None, primary=None):
    return {
        "id": case_id,
        "result": {"exact_page_rank": rank},
        "failure_categories": failures or [],
        "primary_failure_category": primary,
    }


def _manifest():
    return {
        "records": {
            "a": {"source": "Note_A_ar.pdf", "pages": 25, "artifact": "unused-a"},
            "b": {"source": "Note_B_ar.pdf", "pages": 2, "artifact": "unused-b"},
            "c": {"source": "Cir_C_fr.pdf", "pages": 3, "artifact": "unused-c"},
        }
    }


def test_catalog_membership_and_deterministic_controls(monkeypatch):
    evaluation = [
        _case("version", "ar", "rate", "Note_A_ar.pdf"),
        _case("context", "fr", "definition", "Cir_C_fr.pdf"),
        _case("ar-control", "ar", "rate", "Note_B_ar.pdf", quote="code ABC-12"),
        _case("fr-control", "fr", "definition", "Cir_C_fr.pdf", method="visual_review"),
        _case("negative", "ar", "ambiguity", "", relevant=False),
    ]
    result = {
        "records": [
            _record("version", 6, ["wrong_temporal_or_document_version"], "correct_page_missing"),
            _record("context", 6, ["chunk_boundary_or_context_problem"], "chunk_boundary_or_context_problem"),
            _record("ar-control"),
            _record("fr-control"),
            _record("negative", rank=None),
        ]
    }
    pages = {
        ("note_a_ar.pdf", 1): {"blocks": [{"type": "table"}]},
        ("note_b_ar.pdf", 1): {"blocks": [{"type": "paragraph"}]},
        ("cir_c_fr.pdf", 1): {"blocks": [{"type": "paragraph"}]},
    }
    page_counts = {"note_a_ar.pdf": 25, "note_b_ar.pdf": 2, "cir_c_fr.pdf": 3}
    monkeypatch.setattr(
        "experiments.targeted_stress_catalog._load_structured_pages",
        lambda manifest: (pages, page_counts),
    )

    first = build_targeted_stress_catalog(evaluation, result, _manifest())
    second = build_targeted_stress_catalog(evaluation, result, _manifest())

    assert first == second
    assert first["suite_counts"] == {
        "table_pages": 1,
        "visual_non_table": 1,
        "temporal_near_duplicate": 2,
        "long_documents": 1,
        "context_dependence": 2,
        "ambiguity_abstention": 1,
        "identifiers": 1,
    }
    version_roles = {
        case["id"]: case["role"]
        for case in first["suites"]["temporal_near_duplicate"]["cases"]
    }
    assert version_roles == {"version": "failure", "ar-control": "control"}
    context_roles = {
        case["id"]: case["role"]
        for case in first["suites"]["context_dependence"]["cases"]
    }
    assert context_roles == {"context": "failure", "fr-control": "control"}


def test_catalog_rejects_result_missing_an_evaluation_id(monkeypatch):
    monkeypatch.setattr(
        "experiments.targeted_stress_catalog._load_structured_pages",
        lambda manifest: ({}, {}),
    )
    evaluation = [_case("missing", "ar", "rate", "Note_A_ar.pdf")]

    try:
        build_targeted_stress_catalog(evaluation, {"records": []}, _manifest())
    except ValueError as error:
        assert "missing evaluation IDs" in str(error)
    else:
        raise AssertionError("expected a missing-ID validation error")
