from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


BlockType = Literal["heading", "article", "paragraph", "list_item", "table", "caption"]
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


# What a visual reader (local OCR/VLM or Gemini) returns for one page image.
class SensitiveLiteral(BaseModel):
    literal: str
    kind: Literal["number", "date", "time", "percentage", "amount", "identifier", "other"]
    context: str
    uncertain: bool = False


class VisualPage(BaseModel):
    transcription: str = Field(description="Faithful verbatim transcription of all visible text in reading order.")
    items: list[SensitiveLiteral] = Field(default_factory=list)
    uncertain_regions: list[str] = Field(default_factory=list)
    complete: bool
    contains_chart: bool = Field(
        default=False,
        description="True when the page shows a chart, graph, plot, or similar figure (image or drawing).",
    )
    chart_notes: str = Field(
        default="",
        description="Visible chart title, legend, axis labels, and readable data values only; empty if no chart.",
    )
