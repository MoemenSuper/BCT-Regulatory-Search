from experiments.retrieval_ablations import (
    _merge_candidates,
    _normalized_document,
    redundancy_deduplicate,
    rrf_select,
    structured_baseline_chunks,
    structured_hierarchy_chunks,
)
from ingestion.models import Block, Page, StructuredDocument


def _candidate(text, source, dense_rank=None, bm25_rank=None):
    from langchain_core.documents import Document
    ranks = {"test": {}}
    if dense_rank:
        ranks["test"]["dense"] = dense_rank
    if bm25_rank:
        ranks["test"]["bm25"] = bm25_rank
    return {"document": Document(page_content=text, metadata={"source": source, "page": 1}), "representations": {"test"}, "ranks": ranks}


def test_baseline_chunks_keep_ocr_page_metadata_without_heading_in_text():
    document = StructuredDocument(filename="Cir_2026_01_fr.pdf", language="fr", pages=[Page(page_number=3, raw_text="A" * 1200, extraction_method="ocr")])
    chunks = structured_baseline_chunks([document])
    assert len(chunks) == 2
    assert chunks[0].metadata["page"] == 3
    assert chunks[0].metadata["extraction_methods"] == "ocr"
    assert "heading_path" in chunks[0].metadata


def test_metadata_hierarchy_keeps_article_but_strips_title_and_chapter():
    document = StructuredDocument(filename="Cir_2026_01_fr.pdf", language="fr", pages=[Page(page_number=1, raw_text="body", blocks=[Block(type="paragraph", text="Texte utile", page_number=1, heading_path=["TITRE I", "CHAPITRE 2", "Article 62"])])])
    chunk = structured_hierarchy_chunks([document], metadata_only=True)[0]
    assert chunk.page_content.startswith("Article 62")
    assert "TITRE I" not in chunk.page_content
    assert chunk.metadata["heading_path"] == "TITRE I > CHAPITRE 2 > Article 62"


def test_rrf_uses_both_rank_sources_and_dedup_preserves_other_documents():
    dense_only = _candidate("A", "one.pdf", dense_rank=1)
    both = _candidate("B", "one.pdf", dense_rank=2, bm25_rank=1)
    selected = rrf_select([dense_only, both], budget=1)
    assert selected[0]["document"].page_content == "B"
    kept, removed = redundancy_deduplicate([_candidate("same legal text", "one.pdf"), _candidate("same legal text", "one.pdf"), _candidate("same legal text", "other.pdf")])
    assert removed == 1
    assert [candidate["document"].metadata["source"] for candidate in kept] == ["one.pdf", "other.pdf"]


def test_candidate_union_deduplicates_chroma_and_json_page_metadata_shapes():
    dense = _candidate("same", "one.pdf", dense_rank=1)
    dense["document"].metadata["pages"] = "1"
    sparse = _candidate("same", "one.pdf", bm25_rank=1)
    sparse["document"].metadata["pages"] = [1]
    merged = _merge_candidates([[dense], [sparse]])
    assert len(merged) == 1
    assert merged[0]["ranks"] == {"test": {"dense": 1, "bm25": 1}}


def test_normalized_document_uses_filename_source_identity():
    document = _candidate("text", "C:/corpus/Note_2026_01_ar.pdf")["document"]
    assert _normalized_document(document).metadata["source"] == "Note_2026_01_ar.pdf"
