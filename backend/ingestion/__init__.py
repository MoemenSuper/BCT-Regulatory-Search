"""Ingestion: from a PDF to searchable chunks (the other half of the core, next to rag/).

The path of one PDF: extract.py reads each page (the PDF's own text, laid out by Docling; images
and scans read by OCR / a vision model in local_visual.py), chunk.py cuts the pages into chunks
with their context header, index.py stores them (Chroma + chunk files) in a new version of the
index. pipeline.py, registry.py and enrichment.py run the upload queue around these steps.
"""

from .models import Block, Page, StructuredDocument

__all__ = ["Block", "Page", "StructuredDocument"]
