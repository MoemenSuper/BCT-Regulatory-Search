from pathlib import Path

import pytest

from source_documents import SourceDocumentResolver, _word_highlight_rects, render_page_png


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
