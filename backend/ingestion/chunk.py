from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document

from rag.answer_evidence import IMAGE_READING, IMAGE_WORDS

from .models import Block, Page, StructuredDocument


CHUNKER_VERSION = "bct-structure-context-v3"
_MAX_CHARS = 1000
# No piece shorter than this: a lone heading, title or page number embeds as little more than its
# context header, and such scraps took 7-14% of the meaning search's top 10 for unrelated questions.
_MIN_PIECE = 150

# How a piece is described in its context header (French: the corpus language).
_ELEMENT = {"article": "article", "table": "tableau", "list_item": "liste", "paragraph": "texte", "caption": "légende"}


def _sentence_start(text: str, position: int, floor: int) -> int:
    """Where the line or sentence holding `position` starts (not before `floor`), else the word.

    The next chunk starts there, so a sentence cut at the end of one chunk is whole in the next:
    "Un ratio de solvabilité qui ne peut pas être inférieur à 10 %" is never split into
    "... être inféri" and "ur à 10 %".
    """
    cut = max(text.rfind("\n", floor, position), text.rfind(". ", floor, position), text.rfind("؛", floor, position))
    if cut < 0:
        cut = text.rfind(" ", floor, position)
    return cut + 1 if cut >= 0 else position


def _split(text: str, max_chars: int = _MAX_CHARS, overlap: int = 200, *, by_lines: bool = False) -> list[str]:
    """Pieces of at most max_chars, cut at a line or sentence end, overlapping by about `overlap`.
    by_lines (a table): cut only between lines, so a row is never split ("MARS. 2023" ends no sentence)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + 1 if by_lines else start + max_chars // 2, end)
            if boundary < 0 or not by_lines:
                boundary = max(
                    boundary,
                    text.rfind(". ", start + max_chars // 2, end),
                    text.rfind("؛", start + max_chars // 2, end),
                )
            if boundary < 0:  # one long sentence: at least end between two words
                boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary >= 0:
                end = boundary + 1
        piece = text[start:end].strip()
        line_start = text.rfind("\n", 0, start) + 1
        # A chunk that starts in the middle of a marked line keeps the line's mark: image words
        # stay uncitable, and a picture reading stays known as a reading.
        for mark in (IMAGE_WORDS, IMAGE_READING):
            if piece and line_start < start and text.startswith(mark, line_start):
                piece = f"{mark} {piece}"
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        if by_lines and text.rfind("\n", start, end) >= 0:
            start = end  # rows: the next piece starts with the next row, no overlap
            continue
        # The overlap goes back to the start of its sentence, at most twice the usual overlap.
        start = max(_sentence_start(text, end - overlap, max(start + 1, end - 2 * overlap)), start + 1)
    return pieces


def _structured_pieces(page: Page) -> list[tuple[str, list[Block]]] | None:
    """The page cut along its blocks: [(verbatim text, blocks it covers)].

    A piece never crosses into a new heading or article, and a long block (a table, a long
    article) is split on its own, a table only between rows. A scrap under _MIN_PIECE characters
    (a lone heading, a page number) joins the next piece, or the one before at the end of the page.
    None when the blocks do not rebuild the page text exactly (a page read visually): that page
    keeps the plain cut by length.
    """
    if not page.blocks or "\n".join(block.text for block in page.blocks).strip() != page.raw_text.strip():
        return None
    pieces: list[tuple[str, list[Block]]] = []
    current: list[Block] = []

    def flush() -> None:
        if current:
            pieces.append(("\n".join(block.text for block in current).strip(), list(current)))
            current.clear()

    for block in page.blocks:
        if not block.text.strip():
            continue
        if block.type in ("heading", "article") and any(b.type != "heading" for b in current):
            flush()
        if current and sum(len(b.text) + 1 for b in current) + len(block.text) > _MAX_CHARS:
            flush()
        if len(block.text) > _MAX_CHARS:
            parts = _split(block.text, by_lines=block.type == "table")
            if current and all(b.type == "heading" for b in current):
                # Its heading opens the first part (still the page text verbatim: heading, then block).
                heads = "\n".join(b.text for b in current)
                pieces.append((f"{heads}\n{parts[0]}", current + [block]))
                current.clear()
                parts = parts[1:]
            flush()
            pieces.extend((part, [block]) for part in parts)
            continue
        current.append(block)
    flush()
    return _without_scraps([(text, blocks) for text, blocks in pieces if text])


def _without_scraps(pieces: list[tuple[str, list[Block]]]) -> list[tuple[str, list[Block]]]:
    """Join each piece under _MIN_PIECE characters to the next one (a heading to its content), or
    to the previous one at the end of the page. Only across a block boundary, so the joined text is
    still the page text verbatim (two parts of one long block overlap and are never joined)."""
    joined: list[tuple[str, list[Block]]] = []
    for text, blocks in pieces:
        if joined and len(joined[-1][0]) < _MIN_PIECE and joined[-1][1][-1] is not blocks[0]:
            before, before_blocks = joined.pop()
            text, blocks = f"{before}\n{text}", before_blocks + blocks
        joined.append((text, blocks))
    if len(joined) > 1 and len(joined[-1][0]) < _MIN_PIECE and joined[-2][1][-1] is not joined[-1][1][0]:
        (before, before_blocks), (text, blocks) = joined[-2], joined[-1]
        joined[-2:] = [(f"{before}\n{text}", before_blocks + blocks)]
    return joined


def _document_title(document: StructuredDocument) -> str:
    """The document's own name from its first pages: its first heading ("CIRCULAIRE AUX BANQUES
    N°2018-10", "Rapport annuel 2025") and its "Objet" line; the administrator's title too when it
    is more than the file name."""
    parts: list[str] = []
    admin = str((document.metadata.get("administrator_metadata") or {}).get("title") or "").strip()
    if admin and admin.casefold() != Path(document.filename).stem.casefold():
        parts.append(admin)
    first_heading = objet = ""
    for page in document.pages[:2]:
        for block in page.blocks:
            line = " ".join(block.text.split())
            if not first_heading and block.type == "heading":
                first_heading = line
            if not objet and line.casefold().startswith(("objet", "الموضوع")):
                objet = line
    parts += [value[:160] for value in (first_heading, objet) if value]
    return " · ".join(dict.fromkeys(parts))


def _context(document: StructuredDocument, title: str, page: Page, blocks: list[Block] | None) -> tuple[str, str]:
    """(context header, section path) of a piece: document, title, section, element, page."""
    path: list[str] = []
    in_title = set(title.split(" · "))
    for block in blocks if blocks is not None else page.blocks:
        for heading in block.heading_path:
            # The document's own name is said once, in the first line: a short piece made of its
            # title words twice would match any question on the topic, whatever the piece says.
            if heading not in path and heading[:160] not in in_title:
                path.append(heading)
    elements = [] if blocks is None else [_ELEMENT[b.type] for b in blocks if b.type in _ELEMENT]
    section = " › ".join(path)
    # A table with no title of its own keeps the one before it: say it comes from an earlier page.
    earlier = [b.metadata.get("section_page") or page.page_number for b in blocks or [] if b.type == "table"]
    if section and earlier and min(earlier) < page.page_number:
        section += f" (suite, titre page {min(earlier)})"
    first = " · ".join(value for value in (document.filename, title) if value)
    second = " · ".join(value for value in (section, ", ".join(dict.fromkeys(elements)), f"page {page.page_number}") if value)
    return f"{first}\n{second}", section


def _chunk_id(document: StructuredDocument, *, representation: str, page: int, ordinal: int, text: str) -> str:
    value = "\0".join(
        [
            document.content_sha256 or "",
            representation,
            str(page),
            str(ordinal),
            CHUNKER_VERSION,
            text,
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata(document: StructuredDocument, page, *, representation: str, ordinal: int, text: str,
              context: str, section: str) -> dict:
    admin = document.metadata.get("administrator_metadata") or {}
    doc_kind = str(document.metadata.get("doc_kind") or admin.get("doc_kind") or "regulatory")
    authority = str(document.metadata.get("authority") or admin.get("authority") or "primary")
    meta = {
        "source": document.filename,
        "page": page.page_number,
        "pages": [page.page_number],
        "language": document.language,
        "representation": representation,
        "extraction_method": page.extraction_method,
        "heading_path": section,
        "context": context,
        "flat_part": ordinal,
        "chunk_index": ordinal,
        "chunker_version": CHUNKER_VERSION,
        "document_sha256": document.content_sha256,
        "chunk_id": _chunk_id(
            document,
            representation=representation,
            page=page.page_number,
            ordinal=ordinal,
            text=text,
        ),
        "quality_flags": ",".join(page.quality_flags),
        "doc_kind": doc_kind,
        "authority": authority,
    }
    related = str(document.metadata.get("related_to") or admin.get("related_to") or "").strip()
    if related:
        meta["related_to"] = related
    return meta


def build_runtime_chunks(document: StructuredDocument) -> tuple[list[Document], list[Document]]:
    """Return canonical page chunks and additive Gemini Arabic visual chunks.

    Every chunk keeps its page text verbatim and carries a context header in its metadata (document,
    title, section, element, page) that search reads with it: a piece of a table or of an article
    is found through the name of the table or article it belongs to.
    """
    title = _document_title(document)
    primary: list[Document] = []
    visual: list[Document] = []
    for page in document.pages:
        structured = _structured_pieces(page)
        pieces = structured if structured is not None else [(piece, None) for piece in _split(page.raw_text)]
        for ordinal, (piece, blocks) in enumerate(pieces):
            context, section = _context(document, title, page, blocks)
            primary.append(
                Document(
                    page_content=piece,
                    metadata=_metadata(document, page, representation="native", ordinal=ordinal, text=piece,
                                       context=context, section=section),
                )
            )

        visual_text = str(page.metadata.get("visual_text") or "").strip()
        if (
            document.language == "ar"
            and page.extraction_method != "vlm"
            and visual_text
            and page.metadata.get("visual_complete") is True
        ):
            context, section = _context(document, title, page, None)
            for ordinal, piece in enumerate(_split(visual_text)):
                visual.append(
                    Document(
                        page_content=piece,
                        metadata={
                            **_metadata(document, page, representation="arabic_ocr_secondary", ordinal=ordinal,
                                        text=piece, context=context, section=section),
                            "extraction_method": "vlm",
                            "visual_model": page.metadata.get("visual_model"),
                        },
                    )
                )
    if not primary:
        raise ValueError("Ingestion produced no searchable chunks")
    return primary, visual
