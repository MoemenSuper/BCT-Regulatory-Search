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

class DoclingPdfLoader:

    def __init__(self):
        self.converter = DocumentConverter()

    def load(self, file_path: str) -> StructuredDocument:
        pdf_path = Path(file_path)

        result = self.converter.convert(pdf_path)

        docling_document = result.document
        language = detect_language(pdf_path)

        pages_by_number: dict[int, Page] = {}

        for item,level in docling_document.iterate_items():
            if not isinstance(item, TextItem):
                continue
            if not item.prov:
                continue

            page_number = item.prov[0].page_no

            if page_number not in pages_by_number:
                pages_by_number = Page(page_number=page_number, raw_text="")

            page = pages_by_number[page_number]

            # Figure out the block type
            if item.label in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}:
                block_type = "heading"
            elif item.label == DocItemLabel.LIST_ITEM:
                block_type = "list_item"
            elif item.label == DocItemLabel.FORMULA:
                block_type = "formula"

            else:
                block_type = "paragraph"

            #Build the block
            block = Block(
                type=block_type,
                text = item.text,
                page_number=page_number,
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
        "documents/Cir_2026_01_fr.pdf"
    )

    for page in document.pages:
        print(f"\n===== PAGE {page.page_number} =====")

        for block in page.blocks:
            print(
                f"[{block.type}] "
                f"{block.metadata['docling_label']}: "
                f"{block.text}"
            )