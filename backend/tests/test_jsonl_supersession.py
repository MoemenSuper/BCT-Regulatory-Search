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
