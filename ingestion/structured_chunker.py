from dataclasses import dataclass, field

from langchain_core.documents import Document

from ingestion.models import Block, StructuredDocument


@dataclass
class _ChunkGroup:
    heading_path: list[str]
    blocks: list[Block] = field(default_factory=list)


def _split_ranges(text: str, max_chars: int, overlap: int) -> list[tuple[int, int, str]]:
    if len(text) <= max_chars:
        return [(0, len(text), text)]
    pieces: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        stripped = text[start:end].strip()
        if stripped:
            pieces.append((start, end, stripped))
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _render_body(group: _ChunkGroup) -> tuple[str, list[tuple[int, int, int]]]:
    headings = set(group.heading_path)
    body = ""
    spans: list[tuple[int, int, int]] = []
    for block in group.blocks:
        text = block.text.strip()
        if not text or text in headings:
            continue
        separator = "\n" if body else ""
        start = len(body) + len(separator)
        body += separator + text
        spans.append((start, len(body), block.page_number))
    return body, spans


def structure_aware_chunks(document: StructuredDocument, max_chars: int = 1000, overlap: int = 200) -> list[Document]:
    """Group legal blocks while assigning pages only to text present in each chunk."""
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
        prefix = "\n".join(dict.fromkeys(group.heading_path)).strip()
        body, spans = _render_body(group)
        if not body and prefix:
            body = prefix
            first_page = group.blocks[0].page_number
            spans = [(0, len(body), first_page)]
            prefix = ""
        if not body:
            continue
        content_limit = max(1, max_chars - len(prefix) - (1 if prefix else 0))
        effective_overlap = min(overlap, max(content_limit - 1, 0))
        methods = sorted({str(block.metadata.get("extraction_method", "native")) for block in group.blocks})
        block_types = sorted({block.type for block in group.blocks})
        for part_index, (start, end, part) in enumerate(_split_ranges(body, content_limit, effective_overlap)):
            pages = sorted({page for span_start, span_end, page in spans if span_start < end and span_end > start})
            if not pages:
                pages = [group.blocks[0].page_number]
            text = f"{prefix}\n{part}" if prefix else part
            chunks.append(Document(page_content=text, metadata={
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
