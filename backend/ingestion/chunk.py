from __future__ import annotations

import hashlib

from langchain_core.documents import Document

from .models import StructuredDocument


CHUNKER_VERSION = "bct-page-local-1000-200-v1"


def _split(text: str, max_chars: int = 1000, overlap: int = 200) -> list[str]:
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
            boundary = max(
                text.rfind("\n", start + max_chars // 2, end),
                text.rfind(". ", start + max_chars // 2, end),
                text.rfind("؛", start + max_chars // 2, end),
            )
            if boundary >= 0:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


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


def _metadata(document: StructuredDocument, page, *, representation: str, ordinal: int, text: str) -> dict:
    headings = []
    for block in page.blocks:
        for heading in block.heading_path:
            if heading not in headings:
                headings.append(heading)
    admin = document.metadata.get("administrator_metadata") or {}
    doc_kind = str(document.metadata.get("doc_kind") or admin.get("doc_kind") or "regulatory")
    authority = str(document.metadata.get("authority") or admin.get("authority") or "primary")
    meta = {
        "source": document.filename,
        "page": page.page_number,
        "page_end": page.page_number,
        "pages": [page.page_number],
        "language": document.language,
        "representation": representation,
        "extraction_method": page.extraction_method,
        "heading_path": " > ".join(headings),
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
        "quality_score": page.quality_score,
        "quality_flags": ",".join(page.quality_flags),
        "doc_kind": doc_kind,
        "authority": authority,
        "has_chart": bool(page.metadata.get("has_chart")),
    }
    page_image = str(page.metadata.get("page_image_path") or "").strip()
    if page_image:
        meta["page_image_path"] = page_image
    related = str(document.metadata.get("related_to") or admin.get("related_to") or "").strip()
    if related:
        meta["related_to"] = related
    return meta


def build_runtime_chunks(document: StructuredDocument) -> tuple[list[Document], list[Document]]:
    """Return canonical page chunks and additive Gemini Arabic visual chunks."""
    primary: list[Document] = []
    visual: list[Document] = []
    for page in document.pages:
        for ordinal, piece in enumerate(_split(page.raw_text)):
            primary.append(
                Document(
                    page_content=piece,
                    metadata=_metadata(
                        document,
                        page,
                        representation="native",
                        ordinal=ordinal,
                        text=piece,
                    ),
                )
            )

        visual_text = str(page.metadata.get("visual_text") or "").strip()
        if (
            document.language == "ar"
            and page.extraction_method != "vlm"
            and visual_text
            and page.metadata.get("visual_complete") is True
        ):
            for ordinal, piece in enumerate(_split(visual_text)):
                visual.append(
                    Document(
                        page_content=piece,
                        metadata={
                            **_metadata(
                                document,
                                page,
                                representation="arabic_ocr_secondary",
                                ordinal=ordinal,
                                text=piece,
                            ),
                            "extraction_method": "vlm",
                            "visual_model": page.metadata.get("visual_model"),
                        },
                    )
                )
    if not primary:
        raise ValueError("Ingestion produced no searchable chunks")
    return primary, visual
