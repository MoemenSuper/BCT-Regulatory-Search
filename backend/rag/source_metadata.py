"""Shared helpers for source page numbers and PDF filenames."""

from pathlib import Path


def normalize_page(metadata):
    label = metadata.get("page_label")
    if label is not None:
        try:
            return int(label)
        except (TypeError, ValueError):
            return label
    page = metadata.get("page")
    if type(page) is not int:
        return None
    # StructuredDocument chunks record physical PDF pages starting at one.
    # Legacy PyPDF/Chroma documents instead store a zero-based page index.
    return page if "pages" in metadata else page + 1


def safe_pdf_filename(
    name: str,
    *,
    require_exact_basename: bool = False,
    ensure_pdf_suffix: bool = False,
    max_length: int = 240,
) -> str:
    value = Path(name).name.strip().replace("\x00", "")
    if not value or value in {".", ".."}:
        raise ValueError("Invalid PDF filename")
    if require_exact_basename and value != name:
        raise ValueError("Invalid source filename")
    if any(ord(character) < 32 for character in value):
        raise ValueError("PDF filename contains control characters")
    if ensure_pdf_suffix:
        if not value.casefold().endswith(".pdf"):
            value += ".pdf"
    elif not value.casefold().endswith(".pdf"):
        raise ValueError("Source must be a PDF")
    if len(value) > max_length:
        raise ValueError("PDF filename is too long")
    return value
