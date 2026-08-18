import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMode,
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.documents import Document


LANGUAGE_BY_SUFFIX = {
    "_ar": ("ar", "arabic"),
    "_fr": ("fr", "latin"),
}


class DoclingPdfLoader:
    """Convert BCT PDFs into one LangChain Document per PDF page."""

    def __init__(self) -> None:
        self._converters: dict[tuple[str, bool], DocumentConverter] = {}

    def load(self, file_path: str, *, force_ocr: bool = False) -> list[Document]:
        pdf_path = Path(file_path)
        language, ocr_language = self._detect_language(pdf_path)
        converter = self._get_converter(
            ocr_language=ocr_language,
            force_ocr=force_ocr,
        )
        result = converter.convert(pdf_path)
        page_count = len(result.document.pages)
        ocr_mode = (
            OcrMode.FULL_PAGE.value
            if force_ocr
            else OcrMode.PDF_AWARE_LAYOUT_REGIONS.value
        )

        pages = []
        for page_number in range(1, page_count + 1):
            page_content = result.document.export_to_markdown(page_no=page_number)
            pages.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": str(pdf_path),
                        "file_name": pdf_path.name,
                        "page": page_number - 1,
                        "page_label": str(page_number),
                        "total_pages": page_count,
                        "language": language,
                        "extraction_method": "docling",
                        "ocr_backend": "onnxruntime",
                        "ocr_mode": ocr_mode,
                    },
                )
            )

        return pages

    def _get_converter(
        self,
        *,
        ocr_language: str,
        force_ocr: bool,
    ) -> DocumentConverter:
        converter_key = (ocr_language, force_ocr)
        if converter_key not in self._converters:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.ocr_options = RapidOcrOptions(
                lang=[ocr_language],
                backend="onnxruntime",
                mode=(
                    OcrMode.FULL_PAGE
                    if force_ocr
                    else OcrMode.PDF_AWARE_LAYOUT_REGIONS
                ),
            )
            self._converters[converter_key] = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )

        return self._converters[converter_key]

    @staticmethod
    def _detect_language(pdf_path: Path) -> tuple[str, str]:
        stem = pdf_path.stem.lower()
        for suffix, languages in LANGUAGE_BY_SUFFIX.items():
            if stem.endswith(suffix):
                return languages

        raise ValueError(
            f"Cannot determine PDF language from file name: {pdf_path.name}. "
            "Expected a name ending in _ar.pdf or _fr.pdf."
        )
