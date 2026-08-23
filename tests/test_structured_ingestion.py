from ingestion.legal_structure import (
    HierarchyState,
    StructureType,
    recognize_arabic_structure,
    recognize_french_structure,
)
from ingestion.models import Block, Page, StructuredDocument
from ingestion.page_quality import assess_page_quality
from ingestion.structured_chunker import structure_aware_chunks


def test_french_article_splits_heading_from_body():
    match = recognize_french_structure("Article 14 bis : Les banques doivent déclarer.")

    assert match is not None
    assert match.type == StructureType.ARTICLE
    assert match.heading == "Article 14 bis"
    assert match.body == "Les banques doivent déclarer."


def test_arabic_article_is_recognized():
    match = recognize_arabic_structure("الفصل 12 - يجب على البنك التصريح")

    assert match is not None
    assert match.type == StructureType.ARTICLE
    assert match.heading == "الفصل 12"
    assert match.body == "يجب على البنك التصريح"


def test_hierarchy_resets_children_when_chapter_changes():
    state = HierarchyState(title="Titre I", chapter="Chapitre 1", article="Article 2")
    chapter = recognize_french_structure("Chapitre 2 Dispositions nouvelles")

    assert chapter is not None
    state.update(chapter)

    assert state.heading_path() == ["Titre I", "Chapitre 2 Dispositions nouvelles"]


def test_empty_native_page_requires_fallback():
    quality = assess_page_quality("", 0)

    assert quality.requires_fallback is True
    assert quality.flags == ["no_native_text"]


def test_chunk_preserves_article_context_and_page_span():
    document = StructuredDocument(
        filename="Cir_test_fr.pdf",
        language="fr",
        pages=[
            Page(1, "", blocks=[Block("article", "Début", 1, ["Titre I", "Article 4"])]),
            Page(2, "", blocks=[Block("paragraph", "Suite", 2, ["Titre I", "Article 4"])]),
        ],
    )

    chunks = structure_aware_chunks(document)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Titre I\nArticle 4\nDébut\nSuite"
    assert chunks[0].metadata["pages"] == [1, 2]
    assert chunks[0].metadata["source"] == "Cir_test_fr.pdf"


def test_table_is_a_standalone_retrievable_chunk():
    document = StructuredDocument(
        filename="table.pdf",
        pages=[Page(3, "", blocks=[
            Block("paragraph", "Avant", 3, ["Article 7"]),
            Block("table", "| A | B |\n|---|---|\n| 1 | 2 |", 3, ["Article 7"]),
            Block("paragraph", "Après", 3, ["Article 7"]),
        ])],
    )

    chunks = structure_aware_chunks(document)

    assert len(chunks) == 3
    assert chunks[1].metadata["block_types"] == "table"


def test_split_chunk_page_span_only_covers_text_in_that_piece():
    document = StructuredDocument(
        filename="long-annex.pdf",
        pages=[
            Page(1, "", blocks=[Block("paragraph", "page-one " * 20, 1, ["Annexe"])]),
            Page(2, "", blocks=[Block("paragraph", "page-two " * 20, 2, ["Annexe"])]),
        ],
    )

    chunks = structure_aware_chunks(document, max_chars=100, overlap=10)

    page_one_only = [chunk for chunk in chunks if "page-one" in chunk.page_content and "page-two" not in chunk.page_content]
    page_two_only = [chunk for chunk in chunks if "page-two" in chunk.page_content and "page-one" not in chunk.page_content]
    assert page_one_only and all(chunk.metadata["pages"] == [1] for chunk in page_one_only)
    assert page_two_only and all(chunk.metadata["pages"] == [2] for chunk in page_two_only)
