import json

import pytest

from experiments.evaluation_protocol import freeze_protocol


def _case(case_id, source, language="fr"):
    return {
        "id": case_id,
        "query": f"Question {case_id}",
        "language": language,
        "category": "definition",
        "relevant": True,
        "expected_source": source,
        "expected_page": 1,
        "evidence_quote": "Evidence independently verified from the source page.",
    }


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _bilingual(role, *cases):
    """Pad concise fixtures so every frozen split can report both languages."""
    values = list(cases)
    present = {case["language"] for case in values if case["relevant"]}
    number = {"dev": 901, "val": 902, "holdout": 903}[role]
    for language in {"fr", "ar"} - present:
        values.append(
            _case(
                f"{role}-language-control-{language}",
                f"CB_2099_{number}_{language}.pdf",
                language=language,
            )
        )
    return values


def test_freeze_records_hashes_exposure_and_legacy_overlap(tmp_path):
    development = _write(
        tmp_path / "development.json",
        _bilingual("dev", _case("dev-1", "Cir_2024_01_fr.pdf")),
    )
    validation = _write(
        tmp_path / "validation.json",
        _bilingual("val", _case("val-1", "Cir_2024_01_fr.pdf")),
    )
    holdout = _write(
        tmp_path / "holdout.json",
        _bilingual("holdout", _case("holdout-1", "Note_2026_02_ar.pdf", language="ar")),
    )
    output = tmp_path / "protocol.json"

    protocol = freeze_protocol(
        development=development,
        validation=validation,
        holdout=holdout,
        output=output,
        frozen_at="2026-08-24T00:00:00+00:00",
    )

    assert output.exists()
    assert protocol["sets"]["development"]["exposure"] == "inspected_development"
    assert protocol["sets"]["validation"]["exposure"] == "periodic_aggregate_and_failure_review"
    assert protocol["sets"]["final_holdout"]["exposure"] == "final_aggregate_only"
    assert protocol["sets"]["development"]["sha256"]
    assert protocol["sets"]["final_holdout"]["languages"] == {"ar": 1, "fr": 1}
    assert protocol["leakage_audit"]["development_validation"]["exact_sources"] == [
        "Cir_2024_01_fr.pdf"
    ]
    assert protocol["leakage_audit"]["validation_holdout"]["families"] == []


def test_freeze_rejects_validation_holdout_family_leakage(tmp_path):
    development = _write(
        tmp_path / "development.json",
        _bilingual("dev", _case("dev-1", "Cir_2024_09_fr.pdf")),
    )
    validation = _write(
        tmp_path / "validation.json",
        _bilingual("val", _case("val-1", "Note_2023_07_ar.pdf", language="ar")),
    )
    holdout = _write(
        tmp_path / "holdout.json",
        _bilingual("holdout", _case("holdout-1", "Note_2026_07_ar.pdf", language="ar")),
    )

    with pytest.raises(ValueError, match="document families"):
        freeze_protocol(
            development=development,
            validation=validation,
            holdout=holdout,
            output=tmp_path / "protocol.json",
            frozen_at="2026-08-24T00:00:00+00:00",
        )


def test_freeze_parses_compact_circular_names_and_fails_closed_on_unknown_sources(tmp_path):
    development = _write(
        tmp_path / "development.json",
        _bilingual("dev", _case("dev-1", "Cir_2024_09_fr.pdf")),
    )
    validation = _write(
        tmp_path / "validation.json",
        _bilingual("val", _case("val-1", "Cir202204fr.pdf")),
    )
    overlapping_holdout = _write(
        tmp_path / "overlapping-holdout.json",
        _bilingual("holdout", _case("holdout-1", "Cir_2026_04_fr.pdf")),
    )

    with pytest.raises(ValueError, match="document families"):
        freeze_protocol(
            development=development,
            validation=validation,
            holdout=overlapping_holdout,
            output=tmp_path / "protocol.json",
            frozen_at="2026-08-24T00:00:00+00:00",
        )


def test_freeze_rejects_development_holdout_family_leakage_by_default(tmp_path):
    development = _write(
        tmp_path / "development.json",
        _bilingual("dev", _case("dev-1", "Note_2023_07_ar.pdf", language="ar")),
    )
    validation = _write(
        tmp_path / "validation.json",
        _bilingual("val", _case("val-1", "Cir_2024_01_fr.pdf")),
    )
    holdout = _write(
        tmp_path / "holdout.json",
        _bilingual("holdout", _case("holdout-1", "Note_2026_07_ar.pdf", language="ar")),
    )

    with pytest.raises(ValueError, match="Development and final holdout"):
        freeze_protocol(
            development=development,
            validation=validation,
            holdout=holdout,
            output=tmp_path / "protocol.json",
            frozen_at="2026-08-24T00:00:00+00:00",
        )


def test_freeze_treats_french_and_arabic_editions_as_one_family(tmp_path):
    development = _write(
        tmp_path / "development.json",
        _bilingual("dev", _case("dev-1", "Cir_2024_09_fr.pdf")),
    )
    validation = _write(
        tmp_path / "validation.json",
        _bilingual("val", _case("val-1", "Note_2023_07_fr.pdf")),
    )
    holdout = _write(
        tmp_path / "holdout.json",
        _bilingual("holdout", _case("holdout-1", "Note_2026_07_ar.pdf", language="ar")),
    )

    with pytest.raises(ValueError, match="document families"):
        freeze_protocol(
            development=development,
            validation=validation,
            holdout=holdout,
            output=tmp_path / "protocol.json",
            frozen_at="2026-08-24T00:00:00+00:00",
        )


def test_freeze_rejects_family_override_for_recognized_source(tmp_path):
    development = _write(
        tmp_path / "development.json",
        _bilingual("dev", _case("dev-1", "Cir_2024_09_fr.pdf")),
    )
    validation_case = _case("val-1", "Note_2023_07_fr.pdf")
    validation_case["document_family"] = "unrelated"
    validation = _write(tmp_path / "validation.json", _bilingual("val", validation_case))
    holdout = _write(
        tmp_path / "holdout.json",
        _bilingual("holdout", _case("holdout-1", "Note_2026_08_ar.pdf", language="ar")),
    )

    with pytest.raises(ValueError, match="cannot override"):
        freeze_protocol(
            development=development,
            validation=validation,
            holdout=holdout,
            output=tmp_path / "protocol.json",
            frozen_at="2026-08-24T00:00:00+00:00",
        )

    unknown_holdout = _write(
        tmp_path / "unknown-holdout.json",
        _bilingual("holdout", _case("holdout-2", "Decision_alpha_ar.pdf", language="ar")),
    )
    with pytest.raises(ValueError, match="document_family"):
        freeze_protocol(
            development=development,
            validation=_write(
                tmp_path / "safe-validation.json",
                _bilingual("val", _case("val-2", "Note_2023_07_ar.pdf", language="ar")),
            ),
            holdout=unknown_holdout,
            output=tmp_path / "protocol.json",
            frozen_at="2026-08-24T00:00:00+00:00",
        )


def test_freeze_rejects_empty_malformed_or_non_bilingual_splits(tmp_path):
    valid_development = _write(
        tmp_path / "development.json",
        _bilingual("dev", _case("dev-1", "Cir_2024_09_fr.pdf")),
    )
    valid_validation = _write(
        tmp_path / "validation.json",
        _bilingual("val", _case("val-1", "Note_2023_07_fr.pdf")),
    )
    valid_holdout = _write(
        tmp_path / "holdout.json",
        _bilingual("holdout", _case("holdout-1", "Note_2026_08_ar.pdf", language="ar")),
    )

    invalid_values = [
        ([], "must not be empty"),
        ([{**_case("bad-language", "Cir_2025_10_fr.pdf"), "language": "french"}], "language"),
        ([{**_case("bad-relevant", "Cir_2025_10_fr.pdf"), "relevant": 1}], "Boolean"),
        ([_case("fr-only", "Cir_2025_10_fr.pdf")], "French and Arabic"),
    ]
    for index, (invalid, message) in enumerate(invalid_values):
        invalid_development = _write(tmp_path / f"invalid-{index}.json", invalid)
        with pytest.raises(ValueError, match=message):
            freeze_protocol(
                development=invalid_development,
                validation=valid_validation,
                holdout=valid_holdout,
                output=tmp_path / "protocol.json",
            )

    assert valid_development.exists()


def test_freeze_rejects_whitespace_only_document_family(tmp_path):
    unknown = _case("holdout-unknown", "Decision_alpha_ar.pdf", language="ar")
    unknown["document_family"] = "   "
    with pytest.raises(ValueError, match="document_family must be non-empty"):
        freeze_protocol(
            development=_write(
                tmp_path / "development.json",
                _bilingual("dev", _case("dev-1", "Cir_2024_09_fr.pdf")),
            ),
            validation=_write(
                tmp_path / "validation.json",
                _bilingual("val", _case("val-1", "Note_2023_07_fr.pdf")),
            ),
            holdout=_write(tmp_path / "holdout.json", _bilingual("holdout", unknown)),
            output=tmp_path / "protocol.json",
        )
