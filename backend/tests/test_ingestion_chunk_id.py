from langchain_core.documents import Document

from ingestion.index import _document_chunk_id


def test_document_chunk_id_uses_metadata_when_present():
    doc = Document(page_content="hello", metadata={"chunk_id": "abc"})
    assert _document_chunk_id(doc) == "abc"


def test_document_chunk_id_synthesizes_for_legacy_rows():
    doc = Document(
        page_content="legacy text",
        metadata={"source": "Cir_2016_01_fr.pdf", "page": 1, "representation": "native", "flat_part": 0},
    )
    assert len(_document_chunk_id(doc)) == 64
    assert _document_chunk_id(doc) == _document_chunk_id(doc)
