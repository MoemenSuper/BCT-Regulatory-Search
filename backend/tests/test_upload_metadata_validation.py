import pytest
from fastapi import HTTPException

from app import _validate_upload_metadata


def test_upload_metadata_requires_all_five_fields_and_normalizes_type():
    metadata = _validate_upload_metadata(
        title="Circulaire relative aux banques",
        publication_date="2024-03-01",
        document_type="Circulaire",
        category="Réglementation bancaire",
        document_number="2024-03",
    )

    assert metadata["type"] == "circulaire"
    assert metadata["document_number"] == "2024-03"


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("title", "  ", "title is required."),
        ("publication_date", "2024-02-30", "publication_date is invalid."),
        ("document_type", "decision", "document_type must be circulaire or note."),
        ("category", "x", "category must contain at least 2 characters."),
        ("document_number", "reference", "document_number must contain a digit."),
    ],
)
def test_upload_metadata_rejects_invalid_required_values(field, value, detail):
    values = {
        "title": "Titre réglementaire",
        "publication_date": "2024-03-01",
        "document_type": "note",
        "category": "Prudential",
        "document_number": "2024-03",
    }
    values[field] = value

    with pytest.raises(HTTPException, match=detail):
        _validate_upload_metadata(**values)
