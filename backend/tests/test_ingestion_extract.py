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
    monkeypatch.setenv("BCT_GEMINI_VISUAL", "1")
    monkeypatch.setenv("BCT_GEMINI_ARABIC_MODE", "all")

    document = PdfExtractor(visual_transcriber=fake).extract(path)

    assert document.language == "fr"
    assert document.pages[0].page_number == 1
    assert document.pages[0].extraction_method == "native"
    assert fake.calls == []


def test_arabic_policy_keeps_native_text_and_adds_complete_gemini_visual_text(tmp_path: Path, monkeypatch):
    path = tmp_path / "Note_2026_01_ar.pdf"
    # The filename is the trusted language hint. Latin glyphs keep the generated
    # fixture independent of system Arabic fonts while exercising the Arabic policy.
    _make_pdf(path, "BCT regulatory text 2026 with enough native content for a clean extraction page.")
    fake = FakeVisualTranscriber(transcription="النص المرئي الكامل ١١ أكتوبر ٢٠٢٦")
    monkeypatch.setenv("BCT_GEMINI_VISUAL", "1")
    monkeypatch.setenv("BCT_GEMINI_ARABIC_MODE", "all")
    monkeypatch.delenv("BCT_ALLOW_DEGRADED_INGESTION", raising=False)

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
    monkeypatch.setenv("BCT_GEMINI_VISUAL", "1")
    monkeypatch.setenv("BCT_GEMINI_ARABIC_MODE", "all")
    monkeypatch.delenv("BCT_ALLOW_DEGRADED_INGESTION", raising=False)

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
    monkeypatch.setenv("BCT_GEMINI_VISUAL", "1")
    monkeypatch.delenv("BCT_ALLOW_DEGRADED_INGESTION", raising=False)

    page = PdfExtractor(visual_transcriber=fake).extract(path).pages[0]

    assert len(fake.calls) == 1
    assert page.extraction_method == "vlm"
    assert "2016-04" in page.raw_text
    assert "native_digits_unreliable:source_header_conflict" in page.quality_flags
    assert "native_replaced_by_visual" in page.quality_flags
    assert not any(flag.startswith("gemini_digits_unreliable") for flag in page.quality_flags)
    assert "6112" in page.metadata["native_raw_text"]


def test_required_arabic_visual_failure_fails_closed(tmp_path: Path, monkeypatch):
    path = tmp_path / "Note_2026_03_ar.pdf"
    _make_pdf(path, "BCT regulatory text with enough native content to avoid the native quality fallback.")
    fake = FakeVisualTranscriber(transcription="جزء غير مكتمل", complete=False)
    monkeypatch.setenv("BCT_GEMINI_VISUAL", "1")
    monkeypatch.setenv("BCT_GEMINI_ARABIC_MODE", "all")
    monkeypatch.delenv("BCT_ALLOW_DEGRADED_INGESTION", raising=False)

    with pytest.raises(ValueError, match="requires complete visual extraction"):
        PdfExtractor(visual_transcriber=fake).extract(path)


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


def test_chart_page_with_rich_native_still_runs_visual(tmp_path: Path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    from ingestion.extract import ChartSignals, PdfExtractor
    from ingestion.gemini_visual import VisualPage

    path = tmp_path / "Bulletin_rich_chart_fr.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Figure 1. Exportations 2024. Serie A 12. Serie B 18. Source: BCT. "
        "Legende complete avec assez de texte natif pour porter la page sans OCR.",
    )
    document.save(path)
    document.close()

    class FakeVisual:
        model = "gemini-test"
        calls = 0

        def transcribe(self, **_):
            self.calls += 1
            return VisualPage(
                transcription="Figure 1. Exportations 2024",
                items=[],
                uncertain_regions=[],
                complete=True,
                contains_chart=True,
                chart_notes="Serie C (bars): 21",
            )

    fake = FakeVisual()
    monkeypatch.setenv("BCT_GEMINI_VISUAL", "1")
    monkeypatch.setenv("BCT_GEMINI_CHART_VISION", "1")
    monkeypatch.setattr(
        "ingestion.extract.page_chart_signals",
        lambda _page: ChartSignals(
            image_count=1,
            drawing_cluster_count=1,
            max_image_area_ratio=0.2,
            max_drawing_area_ratio=0.2,
            suspect=True,
        ),
    )
    structured = PdfExtractor(visual_transcriber=fake).extract(path)
    assert fake.calls == 1
    assert "chart_suspect" in structured.pages[0].quality_flags
    assert "chart_notes_merged" in structured.pages[0].quality_flags
    assert "Serie C (bars): 21" in structured.pages[0].raw_text
    tmp = structured.pages[0].metadata.get("page_image_tmp")
    assert tmp and Path(tmp).is_file()
    Path(tmp).unlink(missing_ok=True)


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


def test_persist_page_images_moves_temp_spill(tmp_path: Path):
    from ingestion.models import Page, StructuredDocument
    from ingestion.pipeline import _persist_page_images

    spill = tmp_path / "spill.png"
    spill.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    structured = StructuredDocument(
        filename="Bulletin.pdf",
        pages=[Page(page_number=1, raw_text="chart", metadata={"page_image_tmp": str(spill)})],
    )
    written = _persist_page_images(structured, tmp_path / "immutable")
    assert written == 1
    assert not spill.exists()
    dest = Path(structured.pages[0].metadata["page_image_path"])
    assert dest.is_file()
    assert "page_image_tmp" not in structured.pages[0].metadata
