import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import (DocItemLabel, TableItem, TextItem)

from ingestion.models import Block, Page, StructuredDocument


def detect_language(pdf_path: Path) -> str:
    filename = pdf_path.stem.lower()

    if filename.endswith("_fr"):
        return "fr"

    if filename.endswith("_ar"):
        return "ar"

    return "unknown"


ARTICLE_PATTERN_FR = re.compile(
    r"^\s*"
    r"(Article\s+"
    r"(?:premier|1er|\d+(?:\s*(?:bis|ter|quater))?)"
    r"(?:\s*\([^)]*\))?"
    r")"
    r"\s*[:\-–—]?\s*"
    r"(.*)$",
    re.IGNORECASE,
)


def split_french_article(text: str) -> tuple[str, str] | None:
    match = ARTICLE_PATTERN_FR.match(text)

    if not match:
        return None

    article_heading = match.group(1).strip()
    article_body = match.group(2).strip()

    return article_heading, article_body

class DoclingPdfLoader:

    def __init__(self):
        self.converter = DocumentConverter()

    def load(self, file_path: str) -> StructuredDocument:
        pdf_path = Path(file_path)

        result = self.converter.convert(pdf_path)

        docling_document = result.document
        language = detect_language(pdf_path)

        pages_by_number: dict[int, Page] = {}

        #We use this to track articles that include more than 1 paragph and/or extend to other pages
        current_article: str | None = None
        for item,level in docling_document.iterate_items():
            if not isinstance(item, TextItem):
                continue
            if not item.prov:
                continue

            page_number = item.prov[0].page_no

            if page_number not in pages_by_number:
                pages_by_number[page_number] = Page(page_number=page_number, raw_text="")

            page = pages_by_number[page_number]

            article_parts = None

            if language == "fr":
                article_parts = split_french_article(item.text)

            if article_parts:
                article_heading, article_body = article_parts
                # Remember This article:
                current_article = article_heading

                
                block = Block(
                    type="article",
                    text=article_body,
                    page_number=page_number,
                    heading_path=[article_heading],
                    metadata={
                        "docling_label": item.label.value,
                        "hierarchy_level": level,
                        "article_heading": article_heading,
                    },
                )

                page.blocks.append(block)
                page.raw_text += item.text + "\n"

                continue
            # Figure out the block type
            if item.label in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}:
                block_type = "heading"
            elif item.label == DocItemLabel.LIST_ITEM:
                block_type = "list_item"
            elif item.label == DocItemLabel.FORMULA:
                block_type = "formula"

            else:
                block_type = "paragraph"


            heading_path = []
            if current_article and block_type in {"paragraph", "list_item", "formula"}:
            
                heading_path = [current_article]
    
            #Build the block
            block = Block(
                type=block_type,
                text = item.text,
                page_number=page_number,
                heading_path=heading_path,
                metadata={
                    "docling_label": item.label.value,
                    "hierarchy_level": level,
                },
            )

            page.blocks.append(block)
            page.raw_text += item.text + "\n"

            

        pages = [
            pages_by_number[number]
            for number in sorted(pages_by_number)
        ]

        return StructuredDocument(
            filename=pdf_path.name,
            language=language,
            pages=pages,
        )


if __name__ == "__main__":
    loader = DoclingPdfLoader()

    document = loader.load(
        r"C:\Users\Moemen Super\BCT-Regulatory-Search\documents\Circulaires et notes 2026\Cir_2026_01_fr.pdf"
    )

    for page in document.pages:
        for block in page.blocks:

            if block.type == "article":
                print(
                    f"[ARTICLE] "
                    f"{block.heading_path[-1]} -> "
                    f"{block.text}"
                )

            else:
                print(
                    f"[{block.type}] "
                    f"{block.metadata['docling_label']}: "
                    f"{block.text}"
                )