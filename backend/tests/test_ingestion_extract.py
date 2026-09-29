from pathlib import Path

import pytest

from ingestion.extract import PdfExtractor
from ingestion.gemini_visual import VisualPage


class FakeVisualTranscriber:
    model = "gemini-3.7-flash"

    def __init__(self, *, transcription: str, complete: bool = True):
        self.transcription = transcription
        self.complete = complete
        self.calls = []

    def transcribe(self, *, image_png: bytes, source_pdf_sha256: str, page_number: int, **_):
        self.calls.append((source_pdf_sha256, page_number, len(image_png)))
        return VisualPage(
            transcription=self.transcription,
            items=[],
            uncertain_regions=[] if self.complete else ["unreadable region"],
            complete=self.complete,
        )


def _make_pdf(path: Path, text: str | None = None):
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 100), text)
    else:
        page.draw_rect(pymupdf.Rect(72, 72, 300, 200), fill=(0, 0, 0))  # scanned-looking, not blank
    document.save(path)
    document.close()


def test_clean_french_pdf_stays_native_without_visual_call(tmp_path: Path, monkeypatch):
    path = tmp_path / "Cir_2026_01_fr.pdf"
    _make_pdf(
        path,
        "Article 12. La Banque Centrale de Tunisie peut retirer un agrement lorsque les conditions ne sont plus remplies.",
    )
    fake = FakeVisualTranscriber(transcription="unused")

    document = PdfExtractor(visual_transcriber=fake).extract(path)

    assert document.language == "fr"
    assert document.pages[0].page_number == 1
    assert document.pages[0].extraction_method == "native"
    assert fake.calls == []


def test_arabic_page_with_numbers_keeps_native_text_and_adds_the_visual_reading(tmp_path: Path, monkeypatch):
    path = tmp_path / "Note_2026_01_ar.pdf"
    # The filename is the trusted language hint. Latin glyphs keep the generated
    # fixture independent of system Arabic fonts while exercising the Arabic policy.
    _make_pdf(path, "BCT regulatory text 2026 with enough native content for a clean extraction page.")
    fake = FakeVisualTranscriber(transcription="النص المرئي الكامل ١١ أكتوبر ٢٠٢٦")

    document = PdfExtractor(visual_transcriber=fake).extract(path)
    page = document.pages[0]

    assert document.language == "ar"
    assert page.extraction_method == "native"
    assert page.metadata["visual_complete"] is True
    assert "١١ أكتوبر ٢٠٢٦" in page.metadata["visual_text"]
    assert len(fake.calls) == 1


def test_blank_scanned_page_uses_complete_gemini_transcription_as_primary(tmp_path: Path, monkeypatch):
    path = tmp_path / "Note_2026_02_ar.pdf"
    _make_pdf(path)
    fake = FakeVisualTranscriber(transcription="الفصل الأول\nالمبلغ ١٠٠٠ دينار")

    document = PdfExtractor(visual_transcriber=fake).extract(path)
    page = document.pages[0]

    assert page.extraction_method == "vlm"
    assert page.raw_text == "الفصل الأول\nالمبلغ ١٠٠٠ دينار"
    assert "native_replaced_by_visual" in page.quality_flags


def test_page_with_garbled_native_digits_is_replaced_by_gemini_transcription(tmp_path: Path, monkeypatch):
    path = tmp_path / "Cir_2016_04_fr.pdf"
    # A broken font map keeps the text readable but reverses digits; the header
    # then contradicts the trusted filename, which routes the page to Gemini.
    _make_pdf(path, "CIRCULAIRE AUX BANQUES n° 6112-04 du 15 septembre 6112. Ligne de financement de 31 millions.")
    fake = FakeVisualTranscriber(transcription="CIRCULAIRE AUX BANQUES n° 2016-04 du 15 septembre 2016. Ligne de financement de 31 millions.")

    page = PdfExtractor(visual_transcriber=fake).extract(path).pages[0]

    assert len(fake.calls) == 1
    assert page.extraction_method == "vlm"
    assert "2016-04" in page.raw_text
    assert "native_digits_unreliable:source_header_conflict" in page.quality_flags
    assert "native_replaced_by_visual" in page.quality_flags
    assert not any(flag.startswith("gemini_digits_unreliable") for flag in page.quality_flags)
    assert "6112" in page.metadata["native_raw_text"]


def test_blank_cover_page_does_not_abort_document(tmp_path: Path, monkeypatch):
    """Born-digital stats PDFs often have empty/image covers; ingest must continue."""
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "RA_blank_cover_fr.pdf"
    document = pymupdf.open()
    document.new_page()  # blank cover
    page = document.new_page()
    page.insert_text((72, 100), "Chapitre 1. Situation economique avec assez de texte natif pour rester native.")
    document.save(path)
    document.close()

    monkeypatch.setenv("BCT_VISUAL_BACKEND", "off")
    structured = PdfExtractor(visual_transcriber=None).extract(path)
    assert len(structured.pages) == 2
    assert "native_unusable_retained" in structured.pages[0].quality_flags
    assert structured.pages[1].extraction_method == "native"
    assert "Situation economique" in structured.pages[1].raw_text


def test_render_page_png_caps_pixel_budget(tmp_path: Path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    from ingestion.extract import render_page_png

    path = tmp_path / "wide.pdf"
    document = pymupdf.open()
    # Oversized page; uncapped 2× would be far above the default pixel budget.
    page = document.new_page(width=2000, height=2800)
    page.insert_text((72, 100), "stats table")
    document.save(path)
    document.close()

    monkeypatch.setenv("BCT_PAGE_RENDER_MAX_EDGE", "800")
    monkeypatch.setenv("BCT_PAGE_RENDER_MAX_PIXELS", "500000")
    # Re-import env caps by calling helper after env set — helper reads env at call time
    # via module-level constants captured at import. Force re-read:
    import ingestion.extract as extract_mod

    monkeypatch.setattr(extract_mod, "_MAX_RENDER_EDGE_PX", 800.0)
    monkeypatch.setattr(extract_mod, "_MAX_RENDER_PIXELS", 500_000.0)

    with pymupdf.open(path) as pdf:
        png = render_page_png(pdf.load_page(0), preferred_scale=2.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) < 400_000


def test_repair_lam_alef_puts_lam_before_zero_width_alef():
    from ingestion.extract import repair_lam_alef

    def char(c, width):
        return {"c": c, "bbox": (10.0, 0.0, 10.0 + width, 5.0)}

    # PyMuPDF order for "خلال": خ ا(zero-width) ل ل  ->  خ ل ا ل
    chars = [char("خ", 4), char("ا", 0.0), char("ل", 3), char("ل", 3)]
    assert repair_lam_alef(chars) == "خلال"
    # A real alef (with width) before lam is left alone: "الل" in "الله".
    assert repair_lam_alef([char("ا", 2), char("ل", 3), char("ل", 3), char("ه", 3)]) == "الله"


def test_native_blocks_fix_lam_alef_on_real_corpus_note():
    pymupdf = pytest.importorskip("pymupdf")
    from ingestion.extract import _native_blocks

    matches = sorted(Path(__file__).resolve().parents[2].joinpath("documents").rglob("Note_2017_19_ar.pdf"))
    if not matches:
        pytest.skip("BCT corpus not present")
    with pymupdf.open(matches[0]) as pdf:
        text = "\n".join(block.text for page in pdf for block in _native_blocks(page, page.number + 1))
    assert "خالل" not in text and "خلال" in text


def test_docling_layout_blocks_become_the_page_text_and_are_cached(tmp_path: Path, monkeypatch):
    import ingestion.docling_layout as layout
    from answer_evidence import IMAGE_WORDS

    path = tmp_path / "Rapport_2025_fr.pdf"
    _make_pdf(path, "texte natif")
    calls = []

    def blocks(pdf_path, raw_cache=None):
        calls.append(pdf_path)
        return {1: ["Tableau 4", "Encadrement Moyen — Effectif: 293", f"{IMAGE_WORDS} 56 de 30 à 34 ans 67"]}

    monkeypatch.setattr(layout, "page_blocks", blocks)
    cache = tmp_path / "docling-layout.json"
    for _ in range(2):  # the re-extraction after enrichment reads the cache
        page = PdfExtractor(visual_transcriber=None).extract(path, layout_cache=cache).pages[0]
    assert len(calls) == 1
    assert page.raw_text == f"Tableau 4\nEncadrement Moyen — Effectif: 293\n{IMAGE_WORDS} 56 de 30 à 34 ans 67"


def test_docling_table_is_kept_only_when_its_rows_match_the_pdf_lines():
    from ingestion.docling_layout import rows_match_pdf

    lines = ["Encadrement Supérieur 354 42,5", "Encadrement Moyen 293 35,1"]
    assert rows_match_pdf([["354", "42,5"], ["293", "35,1"]], lines)
    assert not rows_match_pdf([["354", "35,1"]], lines)  # TableFormer put a cell in the wrong row


def test_chunk_starting_inside_chart_words_keeps_the_mark():
    from answer_evidence import IMAGE_WORDS
    from ingestion.chunk import _split

    text = "Titre\n" + IMAGE_WORDS + " " + " ".join(f"{i} de {i} ans" for i in range(200))
    pieces = _split(text)
    assert len(pieces) > 1 and all(piece.startswith(IMAGE_WORDS) for piece in pieces[1:])


def test_docling_column_headers_keep_years_apart():
    from ingestion.docling_layout import column_headers

    # Tableau 1.1 of the Conjoncture note: a spanning "update" header over two year columns.
    rows = [["Désignation", "Année", "Actualisations du mois d'avril 2026", "Actualisations du mois d'avril 2026"],
            ["Désignation", "2025", "2026", "2027"]]
    assert column_headers(rows) == ["Désignation", "Année / 2025", "Actualisations du mois d'avril 2026 / 2026",
                                    "Actualisations du mois d'avril 2026 / 2027"]


def test_image_region_is_read_visually_and_put_back_in_place(tmp_path: Path, monkeypatch):
    """A readable page with a box the PDF text layer cannot fill (a table or text that is an image):
    only that box is read, and its reading lands between the paragraphs around it."""
    import ingestion.docling_layout as layout
    from ingestion.extract import ImageRegions

    before = "La Banque Centrale de Tunisie a recruté de nouveaux agents au cours de l'année 2025."
    after = "La répartition de l'effectif par grade est présentée dans le tableau ci-dessus."
    path = tmp_path / "Rapport_2025_fr.pdf"
    _make_pdf(path, before)
    monkeypatch.setattr(layout, "page_blocks", lambda _pdf, _raw=None: {1: [before, {"image": [72, 120, 300, 300]}, after]})
    calls = []

    class Reader:
        model = "local-test"

        def transcribe(self, *, image_png, image_region, **_):
            calls.append((image_region, len(image_png)))
            return VisualPage(transcription="Effectif total — 2025: 834", complete=True)

    page = PdfExtractor(visual_transcriber=Reader()).extract(path).pages[0]
    assert [region for region, _size in calls] == [True]  # the box only, not the whole page
    assert page.raw_text == f"{before}\nEffectif total — 2025: 834\n{after}"

    # Deferred (upload): pending until the background reader stores the region's reading.
    pending = PdfExtractor().extract(path, visual_results={}).pages[0]
    assert "visual_pending" in pending.quality_flags
    assert pending.metadata["visual_plan"]["image_regions"] is True
    assert pending.raw_text == f"{before}\n{after}"
    read = PdfExtractor().extract(path, visual_results={1: ImageRegions(texts=["Effectif total — 2025: 834"], complete=True)})
    assert read.pages[0].raw_text == page.raw_text


def test_table_cells_stored_as_separate_pdf_lines_are_regrouped_into_rows():
    pymupdf = pytest.importorskip("pymupdf")
    from ingestion.docling_layout import _visual_rows

    cells = [(pymupdf.Rect(x, y, x + 20, y + 9), text) for x, y, text in [
        (300, 100, "3,4"), (80, 100, "Monde"), (200, 101, "3,5"), (80, 115, "Chine"), (200, 115, "5,0")]]
    assert _visual_rows(cells) == ["Monde 3,5 3,4", "Chine 5,0"]
