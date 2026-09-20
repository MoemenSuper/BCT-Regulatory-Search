"""Unit tests for JSONL supersession pin + ingest extract (no Voyage / Groq)."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from jsonl_supersession import (
    SupersessionEdge,
    extract_edges_from_page_text,
    instruments_from_text,
    merge_supersession_edges_for_ingest,
    pin_supersession_edges,
    select_edges,
    write_edges,
)


def _edge(**kwargs):
    base = dict(
        source_instrument="cir:2019:7",
        source_file="Cir_2019_07_fr.pdf",
        source_page=2,
        action="ABROGATE",
        target_instrument="cir:2018:7",
        target_article="2",
        quote="Les dispositions de l'article 2 de la circulaire 2018-07 sont abrogées.",
    )
    base.update(kwargs)
    return SupersessionEdge(**base)


def test_select_edges_matches_target_instrument():
    edges = [_edge()]
    picked = select_edges(
        edges,
        "L'article 2 de la circulaire 2018-07 est-il encore en vigueur ?",
        limit=1,
    )
    assert len(picked) == 1
    assert picked[0].source_file == "Cir_2019_07_fr.pdf"


def test_select_edges_ignores_ordinary_query_without_instrument():
    edges = [_edge()]
    assert select_edges(edges, "Quelles sont les obligations de change ?", limit=2) == []


def test_select_edges_from_hits_when_old_circular_in_top_results():
    from jsonl_supersession import select_edges_from_hits

    edges = [_edge()]
    old = Document(
        page_content="taux de change applicables",
        metadata={"source": "Cir_2018_07_fr.pdf", "page": 4, "pages": [4]},
    )
    other = Document(
        page_content="autre sujet",
        metadata={"source": "Cir_2016_01_fr.pdf", "page": 1, "pages": [1]},
    )
    picked = select_edges_from_hits(edges, [(old, 2.0), (other, 1.0)], limit=2)
    assert len(picked) == 1
    assert picked[0].source_instrument == "cir:2019:7"
    assert picked[0].target_instrument == "cir:2018:7"


def test_pin_topical_query_prefers_successor_over_superseded_hit():
    """User asks about a topic; classic retrieve returns the old circular; pin successor."""
    edges = [
        _edge(
            action="REPLACE",
            target_article=None,
            quote="La circulaire 2018-07 est abrogée et remplacée.",
        )
    ]
    old = Document(
        page_content="horaires de travail applicables aux banques",
        metadata={"source": "Cir_2018_07_fr.pdf", "page": 3, "pages": [3]},
    )
    noise = Document(
        page_content="autre circulaire",
        metadata={"source": "Cir_2015_02_fr.pdf", "page": 1, "pages": [1]},
    )

    def lookup(edge):
        return edge.source_file, edge.source_page, edge.quote

    pinned = pin_supersession_edges(
        [(old, 5.0), (noise, 1.0)],
        "Quels sont les horaires de travail des banques ?",
        edges,
        page_lookup=lookup,
    )
    assert Path(str(pinned[0][0].metadata["source"])).name == "Cir_2019_07_fr.pdf"
    # Fully replaced instrument is demoted below successor + unrelated live hits
    sources = [Path(str(doc.metadata["source"])).name for doc, _ in pinned]
    assert sources.index("Cir_2019_07_fr.pdf") < sources.index("Cir_2018_07_fr.pdf")
    assert sources.index("Cir_2015_02_fr.pdf") < sources.index("Cir_2018_07_fr.pdf")


def test_pin_amend_keeps_old_pages_but_fronts_successor():
    edges = [
        _edge(
            action="AMEND",
            source_instrument="cir:2020:3",
            source_file="Cir_2020_03_fr.pdf",
            source_page=2,
            target_instrument="cir:2016:8",
            target_article="5",
            quote="L'article 5 de la circulaire 2016-08 est modifié.",
        )
    ]
    old = Document(
        page_content="durée du travail",
        metadata={"source": "Cir_2016_08_fr.pdf", "page": 5, "pages": [5]},
    )

    def lookup(edge):
        return edge.source_file, edge.source_page, edge.quote

    pinned = pin_supersession_edges(
        [(old, 3.0)],
        "Quelle est la durée du travail applicable ?",
        edges,
        page_lookup=lookup,
    )
    assert Path(str(pinned[0][0].metadata["source"])).name == "Cir_2020_03_fr.pdf"
    # AMEND is partial — old circular stays in the pack (not demoted away)
    assert any(
        Path(str(doc.metadata["source"])).name == "Cir_2016_08_fr.pdf"
        for doc, _ in pinned
    )


def test_messy_user_query_parses_instrument():
    assert "cir:2018:7" in instruments_from_text("circ 2018 07 toujours valable ?")
    assert "cir:2017:8" in instruments_from_text(
        "En compliance on me demande si 2017-08 est encore applicable aujourd'hui."
    )


def test_pin_boosts_declaring_page_to_front():
    edges = [_edge()]
    other = Document(
        page_content="marché des changes",
        metadata={"source": "Cir_2016_01_fr.pdf", "page": 2, "pages": [2]},
    )
    ranked = [(other, 1.0)]

    def lookup(edge):
        return edge.source_file, edge.source_page, edge.quote

    pinned = pin_supersession_edges(
        ranked,
        "circ 2018-07 toujours valable ?",
        edges,
        page_lookup=lookup,
    )
    assert Path(str(pinned[0][0].metadata["source"])).name == "Cir_2019_07_fr.pdf"
    assert pinned[0][0].metadata["retrieval_source"] == "jsonl_supersession"


def test_extract_edges_from_abrogation_page():
    text = (
        "Article 10 : Les dispositions de l'article 2 de la circulaire n°2018-07 "
        "du 14 octobre 2018 sont abrogées et remplacées par les dispositions suivantes."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2019_07_fr.pdf", page_number=2, text=text
    )
    assert edges
    assert edges[0].source_instrument == "cir:2019:7"
    assert edges[0].target_instrument == "cir:2018:7"
    assert edges[0].action == "REPLACE"
    assert edges[0].target_article == "2"
    assert "abrogées et remplacées" in edges[0].quote


def test_extract_amends_action():
    text = (
        "Article 3 : Les dispositions de l'article 5 de la circulaire n°2016-08 "
        "sont modifiées comme suit."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2020_03_fr.pdf", page_number=2, text=text
    )
    assert edges
    assert edges[0].action == "AMEND"
    assert edges[0].target_instrument == "cir:2016:8"
    assert edges[0].target_article == "5"


def test_vu_citation_modifiee_par_never_creates_amends_edge():
    """Cir_2026_04 p.1 cites 94-14 as modified by 2025-13 — not 2026-04 AMENDS 2025-13."""
    text = (
        "Vu la circulaire aux intermédiaires agréés n°94-14 du 14 septembre 1994, "
        "relative au règlement financier des importations et des exportations de "
        "marchandises, telle que modifiée par les textes subséquents et notamment la "
        "circulaire n° 2025-13 du 27 octobre 2025, "
        "Vu la correspondance du Ministère du Commerce."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2026_04_fr.pdf", page_number=1, text=text
    )
    assert edges == []


def test_real_cir_2026_04_page1_vu_block_produces_no_supersession_edge():
    """Regression on the live PDF recital that previously false-fired AMENDS."""
    text = (
        "Tunis, le 26 mars 2026\n"
        "CIRCULAIRE AUX INTERMEDIAIRES AGREES N° 2026-4\n"
        "Objet : Conditions de financement de l'importation de produits non prioritaires.\n"
        "Le Gouverneur de la Banque Centrale de Tunisie,\n"
        "Vu le code des changes et du commerce extérieur promulgué par la loi n°76-18 "
        "du 21 janvier 1976, tel que modifié par les textes subséquents et notamment "
        "le décret-loi n°2011-98 du 24 octobre 2011,\n"
        "Vu la circulaire aux intermédiaires agréés n°94-14 du 14 septembre 1994, "
        "relative au règlement financier des importations et des exportations de "
        "marchandises, telle que modifiée par les textes subséquents et notamment la "
        "circulaire n° 2025-13 du 27 octobre 2025,\n"
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2026_04_fr.pdf", page_number=1, text=text
    )
    assert edges == []
    assert not any(
        e.target_instrument == "cir:2025:13" and e.action in {"AMEND", "ABROGATE", "REPLACE"}
        for e in edges
    )


def test_present_circular_annule_et_remplace_names_target():
    text = (
        "Décide :\n"
        "Article 2- La présente circulaire annule et remplace toutes dispositions "
        "antérieures contraires, notamment la circulaire aux établissements de crédit "
        "n° 2007-18 du 5 juillet 2007 et entre en vigueur à partir de la date de sa "
        "publication."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2024_14_fr.pdf", page_number=1, text=text
    )
    assert len(edges) == 1
    assert edges[0].source_instrument == "cir:2024:14"
    assert edges[0].target_instrument == "cir:2007:18"
    assert edges[0].action == "REPLACE"
    assert "annule et remplace" in edges[0].quote.casefold()


def test_vu_preamble_ignored_when_operative_article_follows():
    text = (
        "Vu la circulaire n°2007-18 du 5 juillet 2007, telle que modifiée par les "
        "textes subséquents et notamment la circulaire n°2014-04,\n"
        "Décide :\n"
        "Article premier : Les dispositions de l'article 2 de la circulaire n°2018-07 "
        "du 14 octobre 2018 sont abrogées et remplacées comme suit."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2019_07_fr.pdf", page_number=2, text=text
    )
    assert edges
    assert all(e.target_instrument == "cir:2018:7" for e in edges)
    assert not any(e.target_instrument in {"cir:2007:18", "cir:2014:4"} for e in edges)


def test_annex_est_modifiee_par_ajout_is_amend_not_citation():
    text = (
        "Article 7\n"
        "L'annexe I à la circulaire n°2017-06 relative au reporting comptable, "
        "prudentiel et statistique à la Banque Centrale de Tunisie est modifiée par "
        "l'ajout de deux déclarations au domaine 4."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2023_05_fr.pdf", page_number=5, text=text
    )
    assert edges
    assert edges[0].source_instrument == "cir:2023:5"
    assert edges[0].target_instrument == "cir:2017:6"
    assert edges[0].action == "AMEND"


def test_article_premier_marker_without_decide():
    text = (
        "Article premier : Les dispositions de l'article 2 de la circulaire n°2018-07 "
        "sont abrogées."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2019_07_fr.pdf", page_number=2, text=text
    )
    assert edges
    assert edges[0].action == "ABROGATE"
    assert edges[0].target_instrument == "cir:2018:7"


def test_article_1_marker_without_decide():
    text = (
        "Article 1\n"
        "Les dispositions de la circulaire n°2016-08 du 1er janvier 2016 sont abrogées."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2017_01_fr.pdf", page_number=1, text=text
    )
    assert edges
    assert edges[0].target_instrument == "cir:2016:8"
    assert edges[0].action == "ABROGATE"


def test_est_remplacee_passive_operative():
    text = (
        "Article 3 : La circulaire n°2015-02 du 10 mars 2015 est remplacée par les "
        "dispositions de la présente circulaire."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2020_01_fr.pdf", page_number=2, text=text
    )
    assert edges
    assert edges[0].source_instrument == "cir:2020:1"
    assert edges[0].target_instrument == "cir:2015:2"
    assert edges[0].action == "REPLACE"


def test_sont_abrogees_passive_operative():
    text = (
        "Décide :\n"
        "Article 4 : Les dispositions de l'article 9 de la circulaire n°2011-05 "
        "sont abrogées."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2018_03_fr.pdf", page_number=3, text=text
    )
    assert edges
    assert edges[0].action == "ABROGATE"
    assert edges[0].target_instrument == "cir:2011:5"
    assert edges[0].target_article == "9"


def test_pdf_line_break_between_article_and_premier():
    """PDF extraction often splits 'Article' / 'premier' across lines."""
    text = (
        "Vu la circulaire n°2001-01 telle que modifiée par la circulaire n°2002-02,\n"
        "Article\n"
        "premier : Les dispositions de la circulaire n°2010-11 sont abrogées."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2012_01_fr.pdf", page_number=1, text=text
    )
    assert edges
    assert all(e.target_instrument == "cir:2010:11" for e in edges)
    assert not any(e.target_instrument in {"cir:2001:1", "cir:2002:2"} for e in edges)


def test_pdf_line_breaks_inside_annule_et_remplace_clause():
    text = (
        "Décide :\n"
        "Article 2-\n"
        "La présente circulaire annule et\n"
        "remplace toutes dispositions antérieures contraires, notamment la\n"
        "circulaire aux banques n°\n"
        "72-56 du 7 août 1972."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2025_12_fr.pdf", page_number=4, text=text
    )
    assert edges
    assert edges[0].action == "REPLACE"
    assert edges[0].target_instrument == "cir:1972:56"


def test_no_operative_amendment_means_no_edge():
    text = (
        "Décide :\n"
        "Article premier : Les intermédiaires agréés constituent des dépôts "
        "couvrant la totalité de la valeur des importations envisagées."
    )
    edges = extract_edges_from_page_text(
        filename="Cir_2026_04_fr.pdf", page_number=2, text=text
    )
    assert edges == []


def test_rebuild_writes_edges_and_resolve_path_uses_them(tmp_path: Path):
    from jsonl_supersession import (
        rebuild_supersession_edges_from_documents,
        resolve_edges_path,
        load_edges,
        clear_supersession_cache,
    )

    docs = tmp_path / "documents"
    docs.mkdir()
    # Minimal stand-in PDF is heavy; drive rebuild via monkeypatched page reader.
    pdf = docs / "Cir_2024_14_fr.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    text = (
        "Décide :\n"
        "Article 2- La présente circulaire annule et remplace toutes dispositions "
        "antérieures contraires, notamment la circulaire n°2007-18 du 5 juillet 2007."
    )

    import jsonl_supersession as mod

    original = mod._pdf_page_texts
    mod._pdf_page_texts = lambda _path: [(1, text)]
    try:
        active = tmp_path / "versions" / "v1"
        active.mkdir(parents=True)
        out = active / "supersession_edges.jsonl"
        report = rebuild_supersession_edges_from_documents(docs, out)
        clear_supersession_cache()
        assert report["edges"] >= 1
        resolved = resolve_edges_path(active)
        assert resolved == out
        edges = load_edges(resolved)
        assert any(
            e.source_instrument == "cir:2024:14" and e.target_instrument == "cir:2007:18"
            for e in edges
        )
    finally:
        mod._pdf_page_texts = original


def test_ingest_merge_replaces_edges_for_same_pdf(tmp_path: Path):
    active = tmp_path / "active"
    staged = tmp_path / "staged"
    active.mkdir()
    staged.mkdir()
    write_edges(
        active / "supersession_edges.jsonl",
        [
            _edge(quote="old quote"),
            _edge(
                source_instrument="cir:2020:3",
                source_file="Cir_2020_03_fr.pdf",
                source_page=1,
                target_instrument="cir:2016:8",
                target_article=None,
                quote="other circular stays",
            ),
        ],
    )

    class Page:
        def __init__(self, page_number, raw_text):
            self.page_number = page_number
            self.raw_text = raw_text
            self.metadata = {}

    pages = [
        Page(
            2,
            "Les dispositions de l'article 2 de la circulaire n°2018-07 sont abrogées.",
        )
    ]
    report = merge_supersession_edges_for_ingest(
        active_before=active,
        staged_version=staged,
        filename="Cir_2019_07_fr.pdf",
        pages=pages,
        asset_root=tmp_path,
    )
    assert report["from_pdf"] >= 1
    assert (staged / "supersession_edges.jsonl").is_file()
    from jsonl_supersession import load_edges

    merged = load_edges(staged / "supersession_edges.jsonl")
    sources = {Path(e.source_file).name for e in merged}
    assert "Cir_2020_03_fr.pdf" in sources
    assert "Cir_2019_07_fr.pdf" in sources
    # old Cir_2019_07 edges replaced by fresh extract
    nineteen = [e for e in merged if Path(e.source_file).name == "Cir_2019_07_fr.pdf"]
    assert nineteen
    assert all(e.quote != "old quote" for e in nineteen)
