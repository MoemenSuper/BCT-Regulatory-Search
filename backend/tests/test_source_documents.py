from pathlib import Path

import pytest

from source_documents import SourceDocumentResolver, _word_highlight_rects, locate_quote, render_page_png


@pytest.fixture
def source_root(tmp_path: Path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    pdf_path = tmp_path / "Cir_2026_01_fr.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "Article 12 - Banque Centrale de Tunisie")
    page.insert_text((72, 125), "Le delai est de trente jours pour presenter des observations.")
    document.save(pdf_path)
    document.close()
    monkeypatch.setenv("BCT_SOURCE_DOCUMENT_ROOTS", str(tmp_path))
    monkeypatch.delenv("BCT_DOCUMENTS_DIR", raising=False)
    monkeypatch.delenv("BCT_INGESTION_DB", raising=False)
    return pdf_path


def test_source_resolver_accepts_only_basename(source_root: Path):
    resolver = SourceDocumentResolver()
    resolved = resolver.resolve(source_root.name)
    assert resolved.path == source_root.resolve()
    with pytest.raises(ValueError):
        resolver.resolve(f"../{source_root.name}")


@pytest.mark.parametrize("status", ["enriching", "ready_degraded"])
def test_source_resolver_opens_ledger_pdfs_that_are_live_but_still_being_read(tmp_path: Path, monkeypatch, status):
    from ingestion.registry import IngestionRegistry

    pymupdf = pytest.importorskip("pymupdf")
    stored = tmp_path / "immutable" / "dette2024.pdf"
    stored.parent.mkdir()
    document = pymupdf.open()
    document.new_page().insert_text((72, 100), "Dette exterieure 2024")
    document.save(stored)
    document.close()
    database = tmp_path / "ingestion.sqlite3"
    registry = IngestionRegistry(database)
    registry.start("a" * 64, "dette2024.pdf", str(stored))
    registry.ready("a" * 64, stored_path=str(stored), asset_version="v1", report={}, status=status)
    registry.close()
    monkeypatch.setenv("BCT_INGESTION_DB", str(database))
    monkeypatch.setenv("BCT_SOURCE_DOCUMENT_ROOTS", str(tmp_path / "empty"))
    monkeypatch.delenv("BCT_DOCUMENTS_DIR", raising=False)
    assert SourceDocumentResolver().resolve("dette2024.pdf").path == stored.resolve()


def test_an_upload_still_in_the_queue_opens_but_a_citation_opens_the_indexed_version(tmp_path: Path, monkeypatch):
    """The admin page links every upload; a newer upload of the same name waiting in the queue
    must not replace the indexed version a citation points to."""
    from ingestion.registry import IngestionRegistry

    pymupdf = pytest.importorskip("pymupdf")
    paths = {}
    for name in ("indexed", "queued"):
        paths[name] = tmp_path / name / "bsf223_fr.pdf"
        paths[name].parent.mkdir()
        document = pymupdf.open()
        document.new_page().insert_text((72, 100), name)
        document.save(paths[name])
        document.close()
    database = tmp_path / "ingestion.sqlite3"
    monkeypatch.setenv("BCT_INGESTION_DB", str(database))
    monkeypatch.setenv("BCT_SOURCE_DOCUMENT_ROOTS", str(tmp_path / "empty"))
    monkeypatch.delenv("BCT_DOCUMENTS_DIR", raising=False)
    registry = IngestionRegistry(database)
    registry.queue("q" * 64, "bsf223_fr.pdf", str(paths["queued"]), {})
    assert SourceDocumentResolver().resolve("bsf223_fr.pdf").path == paths["queued"].resolve()

    registry.start("i" * 64, "bsf223_fr.pdf", str(paths["indexed"]))
    registry.ready("i" * 64, stored_path=str(paths["indexed"]), asset_version="v1", report={})
    registry.close()
    assert SourceDocumentResolver().resolve("bsf223_fr.pdf").path == paths["indexed"].resolve()


def test_rendered_source_page_contains_real_highlight(source_root: Path):
    resolver = SourceDocumentResolver()
    resolved = resolver.resolve(source_root.name)
    png, headers = render_page_png(
        resolved,
        page_number=1,
        quote="Le delai est de trente jours pour presenter des observations.",
        scale=1.2,
    )
    assert png.startswith(b"\x89PNG")
    assert int(headers["X-BCT-Highlight-Matches"]) >= 1
    assert headers["X-BCT-Total-Pages"] == "1"


def test_arabic_quote_is_located_despite_visual_word_order():
    pytest.importorskip("pymupdf")
    # PyMuPDF `get_text("words", sort=True)` returns each line left-to-right, so an
    # Arabic line comes back with its logical word order reversed.
    logical = "تونس، في 30 أكتوبر 2025 مذكرة إلى البنوك عدد 175 لسنة 2025".split()
    line_one, line_two = logical[:5], logical[5:]

    def visual_words(words, y0, y1):
        # (x0, y0, x1, y1, word, block_no, line_no, word_no): the first logical
        # word sits rightmost, and PyMuPDF lists the line leftmost-first.
        placed = [
            (400 - 40 * index, y0, 430 - 40 * index, y1, word, 0, int(y0), index)
            for index, word in enumerate(words)
        ]
        return placed[::-1]

    class FakePage:
        def get_text(self, kind, sort=False):
            assert kind == "words"
            return visual_words(line_one, 100, 112) + visual_words(line_two, 120, 132)

    rects = _word_highlight_rects(FakePage(), " ".join(logical))
    assert len(rects) == 2  # one merged highlight per line
    assert _word_highlight_rects(FakePage(), "مذكرة إلى البنوك") != []
    assert _word_highlight_rects(FakePage(), "نص غير موجود في هذه الصفحة") == []


def test_long_quote_is_located_in_pieces_within_a_bounded_time():
    pytest.importorskip("pymupdf")
    import time

    # A repetitive page is the fuzzy matcher's worst case; a 2000-char request quote must not
    # pin a worker for minutes, and a long real quote is still highlighted piece by piece.
    page_words = ("de la banque centrale " * 150).split()
    line = [(10.0 * i, 100.0 + i // 20 * 12, 10.0 * i + 8, 112.0 + i // 20 * 12, w, 0, i // 20, i)
            for i, w in enumerate(page_words)]

    class FakePage:
        def get_text(self, kind, sort=False):
            return list(line)

    started = time.monotonic()
    assert _word_highlight_rects(FakePage(), ("la de " * 330)[:2000]) is not None
    # "…" splits a quote into segments; they share one budget instead of one each.
    assert locate_quote(FakePage(), " … ".join(["la banque de la "] * 120)[:2000]) is not None
    assert time.monotonic() - started < 10
    assert _word_highlight_rects(FakePage(), " ".join(page_words[:100])) != []
