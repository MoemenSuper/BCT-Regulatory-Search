from dataclasses import dataclass, field

from langchain_core.documents import Document

from ingestion.models import Block, StructuredDocument


@dataclass
class _ChunkGroup:
    heading_path: list[str]
    blocks: list[Block] = field(default_factory=list)


def _render_group(group: _ChunkGroup) -> str:
    headings = list(dict.fromkeys(group.heading_path))
    body = [block.text.strip() for block in group.blocks if block.text.strip()]
    return "\n".join(headings + body).strip()


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        pieces.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [piece for piece in pieces if piece]


def structure_aware_chunks(document: StructuredDocument, max_chars: int = 1000, overlap: int = 200) -> list[Document]:
    """Group related blocks by legal hierarchy before applying size limits."""
    groups: list[_ChunkGroup] = []
    current: _ChunkGroup | None = None
    for page in document.pages:
        for block in page.blocks:
            starts_group = current is None or block.heading_path != current.heading_path or block.type == "table"
            if starts_group:
                current = _ChunkGroup(heading_path=list(block.heading_path))
                groups.append(current)
            current.blocks.append(block)
            if block.type == "table":
                current = None

    chunks: list[Document] = []
    for group_index, group in enumerate(groups):
        rendered = _render_group(group)
        if not rendered:
            continue
        pages = sorted({block.page_number for block in group.blocks})
        methods = sorted({str(block.metadata.get("extraction_method", "native")) for block in group.blocks})
        block_types = sorted({block.type for block in group.blocks})
        for part_index, part in enumerate(_split_text(rendered, max_chars, overlap)):
            chunks.append(Document(page_content=part, metadata={
                "source": document.filename,
                "page": pages[0],
                "page_end": pages[-1],
                "pages": pages,
                "heading_path": " > ".join(group.heading_path),
                "block_types": ",".join(block_types),
                "extraction_methods": ",".join(methods),
                "structure_group": group_index,
                "structure_part": part_index,
                "language": document.language,
            }))
    return chunks
