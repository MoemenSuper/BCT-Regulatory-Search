from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import (DocItemLabel, TableItem, TextItem)

from ingestion.models import Block, Page, StructuredDocument


def detect_language(pdf_path: Path) -> str:
    filename = pdf_path.stem.lower()

    if filename.endswith("_fr"):
        return "fr"

    if filename.endswith("_ar"):
        return "ar"

    return "unknown"

class DoclingPdfLoader:

    def __init__(self):
        self.converter = DocumentConverter()

    def load(self, file_path: str) -> StructuredDocument:
        pdf_path = Path(file_path)

        result = self.converter.convert(pdf_path)

        docling_document = result.document