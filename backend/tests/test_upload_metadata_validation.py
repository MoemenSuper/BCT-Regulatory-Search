"""Upload no longer requires administrator catalog fields."""

from inspect import signature

from app import _install_ingestion_routes
from fastapi import FastAPI


def test_documents_upload_accepts_file_only():
    app = FastAPI()
    _install_ingestion_routes(app)
    route = next(route for route in app.routes if getattr(route, "path", None) == "/documents" and "POST" in getattr(route, "methods", set()))
    params = signature(route.endpoint).parameters
    assert "file" in params
    assert "title" not in params
    assert "publication_date" not in params
    assert "document_type" not in params
    assert "category" not in params
    assert "document_number" not in params
