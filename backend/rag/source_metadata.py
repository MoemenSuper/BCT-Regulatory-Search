"""Two small helpers shared by search and answers: the page number of a chunk, and a PDF filename
made safe to use as a path (no folders, no control characters, must end in .pdf)."""

from pathlib import Path


def normalize_page(metadata):
    """The page number to show and cite, counting from 1 (None if the chunk has no page)."""
    label = metadata.get("page_label")  # the page number printed on the page, when known
    if label is not None:
        try:
            return int(label)
        except (TypeError, ValueError):
            return label
    page = metadata.get("page")
    if type(page) is not int:
        return None
    # Chunks from today's ingestion (they have "pages") already count from 1. Older chunks made
    # with PyPDF counted from 0, so they get +1.
    return page if "pages" in metadata else page + 1


def safe_pdf_filename(
    name: str,
    *,
    require_exact_basename: bool = False,
    ensure_pdf_suffix: bool = False,
    max_length: int = 240,
) -> str:
    """name reduced to a bare PDF filename, or ValueError if it cannot be made safe."""
    # Path(...).name drops any folder part, so "../../etc/x.pdf" becomes "x.pdf".
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
