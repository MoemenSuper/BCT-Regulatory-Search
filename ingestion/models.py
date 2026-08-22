from dataclasses import dataclass, field
from typing import Literal


BlockType = Literal[
    "heading",
    "article",
    "paragraph"
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
    metadata: dict = field(default_factory=dict)


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

    blocks: list[Block] = field(default_factory=list)


@dataclass
class StructuredDocument:
    filename: str

    language: Literal["fr", "ar", "unknown"] =  "unknown"

    document_number: str | None = None
    publication_date: str | None = None

    pages: list[Page] = field(default_factory=list)
        


if __name__ == "__main__":
    block = Block(
        type="article",
        text="Le taux de rémunération de l'épargne est fixé à 6%.",
        page_number=2,
        heading_path=[
            "Article 36 (alinéa premier nouveau)"
        ],
    )

    page = Page(
        page_number=2,
        raw_text=block.text,
        blocks=[block],
    )

    document = StructuredDocument(
        filename="Cir_2026_01_fr.pdf",
        language="fr",
        document_number="2026-1",
        publication_date="2026-01-05",
        pages=[page],
    )

    print(document)