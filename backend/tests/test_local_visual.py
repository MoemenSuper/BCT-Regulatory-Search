from pathlib import Path

from ingestion.gemini_visual import VisualPage
from ingestion.local_visual import LocalVisualRouter, build_visual_transcriber, visual_backend_name


class _FakeEngine:
    def __init__(self, name: str, text: str, *, contains_chart: bool = False, chart_notes: str = ""):
        self.model = name
        self.name = name
        self.text = text
        self.contains_chart = contains_chart
        self.chart_notes = chart_notes
        self.calls = 0

    def transcribe(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return VisualPage(
            transcription=self.text,
            items=[],
            uncertain_regions=[],
            complete=True,
            contains_chart=self.contains_chart,
            chart_notes=self.chart_notes,
        )


def test_visual_backend_defaults_local_for_hybrid(monkeypatch):
    monkeypatch.delenv("BCT_VISUAL_BACKEND", raising=False)
    monkeypatch.setenv("BCT_DEFAULT_PROFILE", "local_hybrid")
    assert visual_backend_name() == "local"
    monkeypatch.setenv("BCT_DEFAULT_PROFILE", "cloud")
    assert visual_backend_name() == "gemini"


def test_build_visual_transcriber_off(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("BCT_VISUAL_BACKEND", "off")
    assert build_visual_transcriber(tmp_path) is None


def test_local_router_sends_arabic_to_easyocr(tmp_path: Path):
    easy = _FakeEngine("easy", "النص العربي")
    paddle = _FakeEngine("paddle", "chart table")
    router = LocalVisualRouter(tmp_path, easy=easy, paddle=paddle)

    page = router.transcribe(
        image_png=b"png",
        source_pdf_sha256="abc",
        page_number=1,
        language="ar",
        image_region=False,
    )

    assert page.transcription == "النص العربي"
    assert easy.calls == 1
    assert paddle.calls == 0


def test_local_router_sends_charts_to_paddle(tmp_path: Path):
    easy = _FakeEngine("easy", "ar")
    paddle = _FakeEngine("paddle", "Tableau 1\n1.2", contains_chart=True, chart_notes="1.2")
    router = LocalVisualRouter(tmp_path, easy=easy, paddle=paddle)

    page = router.transcribe(
        image_png=b"png",
        source_pdf_sha256="abc",
        page_number=2,
        language="fr",
        image_region=True,
    )

    assert page.transcription.startswith("Tableau")
    assert easy.calls == 0
    assert paddle.calls == 1


def test_local_router_arabic_chart_merges_paddle_notes(tmp_path: Path):
    easy = _FakeEngine("easy", "منشور عربي")
    paddle = _FakeEngine("paddle", "ignored body", contains_chart=True, chart_notes="85,5%")
    router = LocalVisualRouter(tmp_path, easy=easy, paddle=paddle)

    page = router.transcribe(
        image_png=b"png",
        source_pdf_sha256="abc",
        page_number=3,
        language="ar",
        image_region=True,
    )

    assert "منشور عربي" in page.transcription
    assert "85,5%" in page.transcription
    assert page.contains_chart is True
    assert easy.calls == 1
    assert paddle.calls == 1


def test_paddle_falls_back_to_cpu_when_gpu_worker_cannot_start(tmp_path, monkeypatch):
    from ingestion import local_visual

    monkeypatch.setenv("BCT_PADDLE_DEVICE", "gpu")
    visual = local_visual.PaddleVlVisual(tmp_path)
    devices = []

    class FakeProc:
        stdin = stdout = None

        def poll(self):
            return None

    def start():
        devices.append(visual._device)
        if visual._device == "gpu":
            raise RuntimeError("PaddleOCR-VL worker failed to start on gpu (exit=1)")
        visual._proc = FakeProc()

    monkeypatch.setattr(visual, "_start_worker", start)
    with visual._ensure_lock():
        pass
    try:
        visual._ask({"cmd": "predict"})
    except AssertionError:
        pass  # FakeProc has no pipes; only the device choice matters here
    assert devices == ["gpu", "cpu"]


def test_paddle_gpu_cap_setting(monkeypatch):
    from ingestion.local_visual import _paddle_gpu_cap_mb

    monkeypatch.setenv("BCT_PADDLE_GPU_MEMORY_MB", "0")
    assert _paddle_gpu_cap_mb() is None
    monkeypatch.setenv("BCT_PADDLE_GPU_MEMORY_MB", "4096")
    assert _paddle_gpu_cap_mb() == 4096


def test_paddle_on_cpu_refuses_to_start_without_enough_memory(tmp_path, monkeypatch):
    import pytest

    from ingestion import local_visual

    monkeypatch.setattr(local_visual, "_available_memory_gb", lambda: 5.0)
    reader = local_visual.PaddleVlVisual(tmp_path)
    reader._device = "cpu"
    with pytest.raises(RuntimeError, match="needs about 8 GB of free memory; 5.0 GB free"):
        reader._start_worker()
