from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


BlockType = Literal["heading", "article", "paragraph", "list_item"]
ExtractionMethod = Literal["native", "vlm"]


@dataclass
class Block:
    type: BlockType
    text: str
    page_number: int
    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    page_number: int
    raw_text: str
    quality_score: float = 1.0
    extraction_method: ExtractionMethod = "native"
    quality_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)


@dataclass
class StructuredDocument:
    filename: str
    language: Literal["fr", "ar", "unknown"] = "unknown"
    document_number: str | None = None
    publication_date: str | None = None
    content_sha256: str | None = None
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
