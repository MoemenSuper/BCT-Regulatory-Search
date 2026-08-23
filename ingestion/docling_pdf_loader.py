import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrAutoOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, TableItem, TextItem

from ingestion.legal_structure import HierarchyState, StructureType, recognizer_for_language
from ingestion.models import Block, Page, StructuredDocument
from ingestion.page_quality import assess_page_quality


def detect_language(pdf_path: Path) -> str:
    filename = pdf_path.stem.lower()
    if filename.endswith("_fr"):
        return "fr"
    if filename.endswith("_ar"):
        return "ar"
    return "unknown"


def _converter(*, do_ocr: bool, force_full_page_ocr: bool = False) -> DocumentConverter:
    options = PdfPipelineOptions()
    options.do_ocr = do_ocr
    options.do_table_structure = True
    if do_ocr:
        options.ocr_options = OcrAutoOptions(force_full_page_ocr=force_full_page_ocr)
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })


def _item_page_number(item: TextItem | TableItem) -> int | None:
    if not item.prov:
        return None
    return item.prov[0].page_no


def _extract_blocks(docling_document, extraction_method: str) -> dict[int, list[Block]]:
    blocks_by_page: dict[int, list[Block]] = {}
    for item, level in docling_document.iterate_items():
        if not isinstance(item, (TextItem, TableItem)):
            continue
        page_number = _item_page_number(item)
        if page_number is None:
            continue

        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc=docling_document).strip()
            block_type = "table"
            label = "table"
            metadata = {"table_format": "markdown"}
        else:
            text = item.text.strip()
            label = item.label.value
            metadata = {}
            if item.label in {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}:
                block_type = "heading"
            elif item.label == DocItemLabel.LIST_ITEM:
                block_type = "list_item"
            elif item.label == DocItemLabel.FORMULA:
                block_type = "formula"
            else:
                block_type = "paragraph"

        if not text:
            continue
        metadata.update({
            "docling_label": label,
            "hierarchy_level": level,
            "extraction_method": extraction_method,
        })
        blocks_by_page.setdefault(page_number, []).append(Block(
            type=block_type,
            text=text,
            page_number=page_number,
            metadata=metadata,
        ))
    return blocks_by_page


def _apply_legal_hierarchy(document: StructuredDocument) -> None:
    hierarchy = HierarchyState()
    recognize = recognizer_for_language(document.language)
    for page in document.pages:
        for block in page.blocks:
            structure = recognize(block.text) if block.type != "table" else None
            if structure:
                hierarchy.update(structure)
                block.heading_path = hierarchy.heading_path()
                block.metadata["structure_type"] = structure.type.value
                block.metadata["structure_heading"] = structure.heading
                if structure.type == StructureType.ARTICLE:
                    block.type = "article"
                    block.text = structure.body or structure.heading
                else:
                    block.type = "heading"
                    block.text = structure.heading
            else:
                block.heading_path = hierarchy.heading_path()


class DoclingPdfLoader:
    """Two-pass loader: native Docling first, OCR only for suspect pages."""

    def __init__(self) -> None:
        self.native_converter = _converter(do_ocr=False)
        self.ocr_converter = _converter(do_ocr=True, force_full_page_ocr=True)

    def _ocr_page(self, pdf_path: Path, page_index: int) -> list[Block]:
        with TemporaryDirectory(prefix="bct-structured-ocr-") as temp_dir:
            source = fitz.open(pdf_path)
            single_page = fitz.open()
            single_page.insert_pdf(source, from_page=page_index, to_page=page_index)
            temp_path = Path(temp_dir) / f"page-{page_index + 1}.pdf"
            single_page.save(temp_path)
            single_page.close()
            source.close()
            result = self.ocr_converter.convert(temp_path)
            blocks = _extract_blocks(result.document, "ocr").get(1, [])
            for block in blocks:
                block.page_number = page_index + 1
            return blocks

    def load(self, file_path: str | Path) -> StructuredDocument:
        pdf_path = Path(file_path)
        native_result = self.native_converter.convert(pdf_path)
        native_blocks = _extract_blocks(native_result.document, "native")

        with fitz.open(pdf_path) as pdf:
            page_count = pdf.page_count

        pages: list[Page] = []
        fallback_attempts: list[dict] = []
        for page_number in range(1, page_count + 1):
            blocks = native_blocks.get(page_number, [])
            raw_text = "\n".join(block.text for block in blocks)
            native_quality = assess_page_quality(raw_text, len(blocks))
            chosen_blocks = blocks
            extraction_method = "native"
            final_quality = native_quality

            if native_quality.requires_fallback:
                ocr_blocks = self._ocr_page(pdf_path, page_number - 1)
                ocr_text = "\n".join(block.text for block in ocr_blocks)
                ocr_quality = assess_page_quality(ocr_text, len(ocr_blocks))
                rescued = bool(ocr_blocks) and ocr_quality.score > native_quality.score
                fallback_attempts.append({
                    "page": page_number,
                    "native_score": native_quality.score,
                    "ocr_score": ocr_quality.score,
                    "rescued": rescued,
                    "native_flags": native_quality.flags,
                })
                if rescued:
                    chosen_blocks = ocr_blocks
                    raw_text = ocr_text
                    extraction_method = "ocr"
                    final_quality = ocr_quality

            for block in chosen_blocks:
                block.metadata["extraction_method"] = extraction_method
            pages.append(Page(
                page_number=page_number,
                raw_text=raw_text,
                quality_score=final_quality.score,
                extraction_method=extraction_method,
                quality_flags=final_quality.flags,
                blocks=chosen_blocks,
                metadata={
                    "native_quality_score": native_quality.score,
                    "native_quality_flags": native_quality.flags,
                    "native_raw_text": "\n".join(block.text for block in blocks),
                },
            ))

        document = StructuredDocument(
            filename=pdf_path.name,
            language=detect_language(pdf_path),
            pages=pages,
            metadata={
                "native_extractor": "docling",
                "fallback": "docling_ocr",
                "fallback_attempts": fallback_attempts,
            },
        )
        _apply_legal_hierarchy(document)
        return document
