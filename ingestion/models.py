from dataclasses import asdict, dataclass, field
from typing import Any, Literal


BlockType = Literal[
    "heading",
    "article",
    "paragraph",
    "list_item",
    "table",
    "formula",
    "image",
    "other",
]

@dataclass
class Block:
    type: BlockType
    text: str

    # Where this block came from:
    page_number: int

    # Example:
    # ["Titre II", "Chapitre 3", "Article 14"]
    heading_path: list[str] = field(default_factory=list)

    # Later useful for tables, coordinates, article numbers, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    page_number: int

    # What the original extractor produced
    raw_text: str

    # 0.0 = extraction is unusable
    # 1.0 = extraction looks excellent
    quality_score: float = 1.0

    # native = extracted from PDF text
    # ocr/vlm = fallback was required
    extraction_method: Literal["native", "ocr", "vlm"] = "native"

    quality_flags: list[str] = field(default_factory=list)

    blocks: list[Block] = field(default_factory=list)


@dataclass
class StructuredDocument:
    filename: str

    language: Literal["fr", "ar", "unknown"] =  "unknown"

    document_number: str | None = None
    publication_date: str | None = None

    pages: list[Page] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

