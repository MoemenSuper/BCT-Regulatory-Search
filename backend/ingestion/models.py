"""The shapes a PDF takes while it is read: a StructuredDocument has Pages, a Page has Blocks
(a heading, a paragraph, a table...). Every way of reading a page fills the same shapes, so the
chunker does not care how the text was obtained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


BlockType = Literal["heading", "article", "paragraph", "list_item", "table", "caption"]
# "native": the PDF's own text layer; "vlm": text read from the page image by a vision model.
ExtractionMethod = Literal["native", "vlm"]


@dataclass
class Block:
    """One piece of a page: its type, its exact text, and the headings it sits under."""

    type: BlockType
    text: str
    page_number: int
    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Page:
    """One PDF page: its full text, how it was read, quality warnings, and its blocks."""

    page_number: int  # physical page, counting from 1
    raw_text: str
    quality_score: float = 1.0
    extraction_method: ExtractionMethod = "native"
    quality_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)


@dataclass
class StructuredDocument:
    """A whole PDF after reading: its identity (name, language, content hash) and its pages."""

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
    """A value where one wrong character changes the meaning (a number, a date, an amount)."""

    literal: str
    kind: Literal["number", "date", "time", "percentage", "amount", "identifier", "other"]
    context: str
    uncertain: bool = False


class VisualPage(BaseModel):
    """The reading of one page image. The Field descriptions are also sent to Gemini as the
    format to fill, so they are written as instructions."""

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
