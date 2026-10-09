"""Every chunk carries its context (document, title, section, element, page) for search, cut along
the page's structure, while its text stays the page text verbatim (answers quote it)."""
from ingestion.chunk import build_runtime_chunks
from ingestion.extract import classify_blocks
from ingestion.models import Block, Page, StructuredDocument
from retrieval_selection import _join


def _page(number: int, items: list[tuple[str, str]]) -> Page:
    blocks = [Block(type="paragraph", text=text, page_number=number, metadata={"layout_kind": kind})
              for text, kind in items]
    return Page(page_number=number, raw_text="\n".join(text for text, _kind in items), blocks=blocks)


def _document(pages: list[Page], filename: str = "Cir_2018_10_fr.pdf") -> StructuredDocument:
    hierarchy = None
    for page in pages:
        hierarchy = classify_blocks(page.blocks, "fr", hierarchy)
    return StructuredDocument(filename=filename, language="fr", content_sha256="a" * 64, pages=pages)


ROW = "DEPOTS MONETAIRES — " + "; ".join(f"(en milliers de dinars) / MARS. 202{i}: 4{i}391" for i in range(9))


def test_a_circular_piece_names_its_document_article_and_page():
    page = _page(2, [
        ("CIRCULAIRE AUX BANQUES N°2018-10", "heading"),
        ("Objet : Ratio « Crédits/Dépôts »", "paragraph"),
        ("Article 3 : Les banques doivent adresser l'état chaque trimestre à la Banque Centrale de Tunisie, "
         "selon le modèle joint en annexe, au plus tard vingt jours après la fin du trimestre.", "paragraph"),
        ("Article 4 : La présente circulaire entre en vigueur à compter de sa date de publication et "
         "s'applique aux situations arrêtées à la fin du premier trimestre de l'année.", "paragraph"),
    ])
    chunks, _ = build_runtime_chunks(_document([page]))

    article3 = next(c for c in chunks if "Article 3" in c.page_content)
    assert "Article 4" not in article3.page_content  # a piece never crosses into the next article
    context = article3.metadata["context"]
    for expected in ("Cir_2018_10_fr.pdf", "CIRCULAIRE AUX BANQUES N°2018-10", "Objet : Ratio", "Article 3", "article", "page 2"):
        assert expected in context, expected


def test_a_table_piece_far_from_its_title_still_names_the_table_and_keeps_rows_whole():
    rows = "\n".join(ROW.replace("DEPOTS MONETAIRES", f"LIGNE {n}") for n in range(6))
    page = _page(42, [("IV-2. SITUATION MENSUELLE DES ETABLISSEMENTS DE LEASING", "heading"), (rows, "table")])
    chunks, _ = build_runtime_chunks(_document([page], "bsf223_fr.pdf"))

    table_pieces = [c for c in chunks if "LIGNE" in c.page_content]
    assert len(table_pieces) > 1  # long table, split...
    for piece in table_pieces:  # ...between rows only, each piece knowing its table
        assert all(line.startswith(("LIGNE ", "IV-2.")) for line in piece.page_content.splitlines())
        assert "IV-2. SITUATION MENSUELLE DES ETABLISSEMENTS DE LEASING" in piece.metadata["context"]
        assert "tableau" in piece.metadata["context"]


def test_the_pieces_rebuild_the_page_text_exactly():
    long_paragraph = " ".join(f"Phrase numéro {n} du rapport annuel sur la politique monétaire." for n in range(40))
    page = _page(7, [("Chapitre 2 Inflation", "heading"), (long_paragraph, "paragraph"), (ROW, "table"),
                     ("- une copie du certificat", "list_item")])
    chunks, _ = build_runtime_chunks(_document([page], "RA_2025_fr.pdf"))

    rebuilt = chunks[0].page_content
    for chunk in chunks[1:]:
        rebuilt = _join(rebuilt, chunk.page_content)
    assert rebuilt == page.raw_text
    assert all("Chapitre 2 Inflation" in c.metadata["context"] for c in chunks)


def test_keyword_search_finds_a_piece_by_its_context():
    from bm25 import create_bm25, retrieve_bm25

    page = _page(42, [("IV-2. SITUATION DES ETABLISSEMENTS DE LEASING", "heading"), ("x " * 600 + "\n" + ROW, "table")])
    chunks, _ = build_runtime_chunks(_document([page], "bsf223_fr.pdf"))
    row_piece = next(c for c in chunks if "DEPOTS MONETAIRES" in c.page_content)
    assert "LEASING" not in row_piece.page_content  # the words are only in its header...

    bm25 = create_bm25(chunks)
    assert retrieve_bm25("dépôts monétaires leasing", bm25, chunks, k=1) == [row_piece]  # ...and still found


def test_a_question_naming_a_section_reads_that_section_first():
    from langchain_core.documents import Document
    from retrieval_selection import named_section_labels, prefer_named_section_hits

    banks = (Document(page_content="DEPOTS MONETAIRES — AVR. 2023: 27137218",
                      metadata={"context": "bsf223_fr.pdf\nII-1-B. SITUATION DES BANQUES · tableau · page 30"}), 0.9)
    leasing = (Document(page_content="DEPOTS MONETAIRES — AVR. 2023: 48730",
                        metadata={"context": "bsf223_fr.pdf\nIV-2. SITUATION DES ETABLISSEMENTS DE LEASING · tableau · page 42"}), 0.8)
    question = "Dans le tableau IV-2, quelles sont les valeurs des dépôts monétaires en avril 2023 ?"

    assert named_section_labels(question) == ["IV-2"]
    assert named_section_labels("la circulaire 2025-07 et l'article 3") == ["article 3"]  # 2025-07 is no section
    assert prefer_named_section_hits([banks, leasing], question) == [leasing, banks]
    assert prefer_named_section_hits([banks, leasing], "dépôts monétaires en avril 2023") == [banks, leasing]


def test_no_piece_is_a_scrap():
    """A lone heading or page number embeds as its header alone and lands near unrelated questions."""
    body = "Le taux d'intérêt des dépôts à terme est fixé librement entre la banque et son client. " * 3
    page = _page(5, [("Chapitre 2 Inflation", "heading"), (body, "paragraph"), ("5", "paragraph")])
    chunks, _ = build_runtime_chunks(_document([page], "RA_2025_fr.pdf"))
    assert all(len(c.page_content) >= 150 for c in chunks)
    assert chunks[0].page_content.startswith("Chapitre 2 Inflation\nLe taux")  # the heading stays with its text
