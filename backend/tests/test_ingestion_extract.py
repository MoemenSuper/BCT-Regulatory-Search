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

    def transcribe(self, *, image_png: bytes, source_pdf_sha256: str, page_number: int):
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
    assert "native_replaced_by_gemini" in page.quality_flags


def test_required_arabic_visual_failure_fails_closed(tmp_path: Path, monkeypatch):
    path = tmp_path / "Note_2026_03_ar.pdf"
    _make_pdf(path, "BCT regulatory text with enough native content to avoid the native quality fallback.")
    fake = FakeVisualTranscriber(transcription="جزء غير مكتمل", complete=False)
    monkeypatch.setenv("BCT_GEMINI_VISUAL", "1")
    monkeypatch.setenv("BCT_GEMINI_ARABIC_MODE", "all")
    monkeypatch.delenv("BCT_ALLOW_DEGRADED_INGESTION", raising=False)

    with pytest.raises(ValueError, match="requires complete Gemini visual extraction"):
        PdfExtractor(visual_transcriber=fake).extract(path)
