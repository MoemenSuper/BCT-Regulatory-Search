from langchain_core.documents import Document

from ingestion.chunk import build_runtime_chunks
from ingestion.models import Block, Page, StructuredDocument


def _page(number: int, text: str, *, method="native", visual_text="", complete=False):
    return Page(
        page_number=number,
        raw_text=text,
        extraction_method=method,
        blocks=[Block(type="paragraph", text=text, page_number=number)],
        metadata={"visual_text": visual_text, "visual_complete": complete, "visual_model": "gemini-3.7-flash"},
    )


def test_chunks_never_cross_physical_pages():
    document = StructuredDocument(
        filename="Cir_2026_01_fr.pdf",
        language="fr",
        content_sha256="abc",
        pages=[_page(1, "A" * 1400), _page(2, "B" * 1400)],
    )
    primary, visual = build_runtime_chunks(document)
    assert visual == []
    assert {chunk.metadata["page"] for chunk in primary} == {1, 2}
    assert all(chunk.metadata["page"] == chunk.metadata["page_end"] for chunk in primary)
    assert all(not ("A" in chunk.page_content and "B" in chunk.page_content) for chunk in primary)


def test_clean_arabic_native_page_gets_additive_gemini_representation():
    document = StructuredDocument(
        filename="Note_2026_01_ar.pdf",
        language="ar",
        content_sha256="abc",
        pages=[_page(1, "native Arabic extraction", visual_text="النص المرئي", complete=True)],
    )
    primary, visual = build_runtime_chunks(document)
    assert len(primary) == 1
    assert len(visual) == 1
    assert visual[0].metadata["representation"] == "arabic_ocr_secondary"
    assert visual[0].metadata["extraction_method"] == "vlm"


def test_gemini_primary_page_is_not_embedded_twice():
    document = StructuredDocument(
        filename="Note_2026_02_ar.pdf",
        language="ar",
        content_sha256="abc",
        pages=[_page(1, "النص المرئي", method="vlm", visual_text="النص المرئي", complete=True)],
    )
    primary, visual = build_runtime_chunks(document)
    assert len(primary) == 1
    assert visual == []
    assert primary[0].metadata["extraction_method"] == "vlm"
