"""Local visual ingest: EasyOCR for Arabic page text, PaddleOCR-VL for image regions and other pages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from langfuse import get_client

from .gemini_visual import VisualPage

ENGINE_EASY = "easyocr-ar-v1"
ENGINE_PADDLE = "paddleocr-vl-1.6-v1"


def _cache_get(cache_dir: Path, binding: dict) -> VisualPage | None:
    key = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    if cached.get("binding") != binding:
        raise ValueError("visual cache binding mismatch")
    return VisualPage.model_validate(cached["response"])


def _cache_put(cache_dir: Path, binding: dict, page: VisualPage) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"binding": binding, "response": page.model_dump()}, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _ocr_generation(engine: str, page_number: int, image_png: bytes):
    return get_client().start_as_current_observation(
        name="transcribe-page",
        as_type="generation",
        model=engine,
        input={"page": page_number, "image_bytes": len(image_png)},
    )


def _traced(generation, page: VisualPage, **metadata) -> VisualPage:
    generation.update(
        output={
            "transcription": page.transcription[:2000],
            "transcription_chars": len(page.transcription),
            "chart_notes": (page.chart_notes or "")[:1000],
            "contains_chart": page.contains_chart,
        },
        metadata=metadata,
    )
    return page


def _binding(engine: str, image_png: bytes, pdf_sha: str, page_number: int) -> dict:
    return {
        "engine": engine,
        "pdf": pdf_sha.lower(),
        "page": int(page_number),
        "image": hashlib.sha256(image_png).hexdigest(),
    }


class EasyOcrVisual:
    model = ENGINE_EASY

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            from hardware import torch_device

            # GPU when the machine has one (EasyOCR runs on CUDA or Apple MPS); BCT_EASYOCR_GPU=0 forces CPU.
            gpu = torch_device() in {"cuda", "mps"} and os.environ.get("BCT_EASYOCR_GPU", "1") == "1"
            self._reader = easyocr.Reader(["ar"], gpu=gpu, verbose=False)
        return self._reader

    def transcribe(self, *, image_png: bytes, source_pdf_sha256: str, page_number: int, **_) -> VisualPage:
        with _ocr_generation(self.model, page_number, image_png) as generation:
            binding = _binding(self.model, image_png, source_pdf_sha256, page_number)
            hit = _cache_get(self.cache_dir, binding)
            if hit is not None:
                return _traced(generation, hit, cache_hit=True)
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                handle.write(image_png)
                path = handle.name
            try:
                cold = self._reader is None
                rows = self._get_reader().readtext(path, detail=0, paragraph=True)
            finally:
                Path(path).unlink(missing_ok=True)
            text = "\n".join(rows) if isinstance(rows, list) else str(rows)
            text = text.strip()
            page = VisualPage(transcription=text, items=[], uncertain_regions=[], complete=True, contains_chart=False)
            _cache_put(self.cache_dir, binding, page)
            return _traced(generation, page, cache_hit=False, model_cold_start=cold)


def _available_memory_gb() -> float | None:
    """Memory the machine (or container VM) can still hand out; None where it cannot be read."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return None


def _paddle_gpu_cap_mb() -> int | None:
    """Hard Paddle allocator cap so the OCR worker cannot starve the API's search models of VRAM.

    ``BCT_PADDLE_GPU_MEMORY_MB`` wins (0 = no cap); otherwise total VRAM minus 1 GB, never below
    2 GB. PaddleOCR-VL 1.6 needs ~7 GB (measured: 6 GB cap fails while loading, 7 GB reads a
    chart page in ~80 s), so smaller cards fail to start on GPU and fall back to CPU.
    None when no CUDA device is visible.
    """
    configured = (os.environ.get("BCT_PADDLE_GPU_MEMORY_MB") or "").strip()
    if configured:
        return int(configured) or None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        total_mb = torch.cuda.get_device_properties(0).total_memory // 2**20
    except Exception:
        return None
    return max(2048, int(total_mb) - 1024)


class PaddleVlVisual:
    """PaddleOCR-VL via a sibling process so Torch/EasyOCR can share the API process on Windows."""

    model = ENGINE_PADDLE

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._proc = None
        self._lock = None
        self._device = os.environ.get("BCT_PADDLE_DEVICE", "gpu")
        self.killed = False

    def _ensure_lock(self):
        if self._lock is None:
            import threading

            self._lock = threading.Lock()
        return self._lock

    def _paddle_python(self) -> str:
        """Prefer a torch-free GPU Paddle venv — same-process Torch+Paddle cuDNN clashes on Windows."""
        import sys

        configured = (os.environ.get("BCT_PADDLE_PYTHON") or "").strip()
        if configured and Path(configured).is_file():
            return configured
        repo = Path(__file__).resolve().parents[2]
        for relative in (
            Path(".tmp") / "paddleocr-vl-bench" / ".venv" / "Scripts" / "python.exe",
            Path(".tmp") / "paddleocr-vl-bench" / ".venv" / "bin" / "python",
        ):
            candidate = repo / relative
            if candidate.is_file():
                return str(candidate)
        return sys.executable

    def _start_worker(self):
        import subprocess
        import sys

        if self._device == "cpu":
            free_gb, needed_gb = _available_memory_gb(), float(os.environ.get("BCT_PADDLE_MIN_FREE_GB", "8"))
            if free_gb is not None and free_gb < needed_gb:
                # Loading the 0.9B model on CPU on a small machine (an 8 GB Docker Desktop) runs it
                # out of memory. The page stays readable from its PDF text and is marked unread.
                raise RuntimeError(
                    f"PaddleOCR-VL on CPU needs about {needed_gb:g} GB of free memory; {free_gb:.1f} GB free"
                )
        self.killed = False
        env = os.environ.copy()
        env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
        # Grow GPU memory on demand instead of pre-reserving a large pool beside the search models.
        env.setdefault("FLAGS_allocator_strategy", "auto_growth")
        cap = _paddle_gpu_cap_mb() if self._device.startswith("gpu") else None
        if cap:
            env.setdefault("FLAGS_gpu_memory_limit_mb", str(cap))
        env["BCT_PADDLE_DEVICE"] = self._device
        env["PYTHONUTF8"] = "1"
        env["BCT_PARENT_PID"] = str(os.getpid())
        # Avoid inheriting work-venv site-packages that pull Torch into the Paddle worker.
        env.pop("PYTHONPATH", None)
        worker = Path(__file__).resolve().parent / "paddle_vl_worker.py"
        python = self._paddle_python()
        with get_client().start_as_current_observation(
            name="start-paddle-worker",
            input={"python": python, "device": self._device, "gpu_memory_limit_mb": env.get("FLAGS_gpu_memory_limit_mb")},
        ) as span:
            self._proc = subprocess.Popen(
                [python, "-u", str(worker)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # avoid pipe-full deadlock while models load
                text=True,
                encoding="utf-8",
                env=env,
                bufsize=1,
            )
            assert self._proc.stdout is not None
            ready = self._proc.stdout.readline()
            if not ready:
                try:
                    exit_code = self._proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    exit_code = "killed"
                self._proc = None
                raise RuntimeError(f"PaddleOCR-VL worker failed to start on {self._device} ({python}, exit={exit_code})")
            payload = json.loads(ready)
            if not payload.get("ok"):
                raise RuntimeError(f"PaddleOCR-VL worker init failed: {payload}")
            span.update(output={"pid": self._proc.pid, "ready": payload})

    def _ask(self, request: dict) -> dict:
        with self._ensure_lock():
            if self._proc is None or self._proc.poll() is not None:
                try:
                    self._start_worker()
                except RuntimeError as error:
                    if not self._device.startswith("gpu"):
                        raise
                    # Not enough free VRAM (small GPU, or search models resident): same model on CPU.
                    get_client().create_event(
                        name="paddle-cpu-fallback", level="WARNING", status_message=str(error)[:1000]
                    )
                    self._device = "cpu"
                    self._start_worker()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                exit_code = self._proc.poll()
                self._proc = None
                raise RuntimeError(f"PaddleOCR-VL worker died (exit={exit_code})")
            return json.loads(line)

    def kill(self) -> None:
        """Free the GPU now; the in-flight predict fails with 'worker died' and the next call restarts it."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self.killed = True
        if os.name == "nt":
            import subprocess

            # The venv python.exe is a launcher; the process holding VRAM is its child.
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
        else:
            proc.kill()

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def transcribe(self, *, image_png: bytes, source_pdf_sha256: str, page_number: int, **_) -> VisualPage:
        with _ocr_generation(self.model, page_number, image_png) as generation:
            binding = _binding(self.model, image_png, source_pdf_sha256, page_number)
            hit = _cache_get(self.cache_dir, binding)
            if hit is not None:
                return _traced(generation, hit, cache_hit=True)
            import shutil
            import tempfile

            tmp = Path(tempfile.mkdtemp(prefix="bct-paddle-vl-"))
            png_path = tmp / "page.png"
            out_dir = tmp / "out"
            out_dir.mkdir()
            try:
                png_path.write_bytes(image_png)
                cold = self._proc is None or self._proc.poll() is not None
                result = self._ask({"cmd": "predict", "png": str(png_path), "out_dir": str(out_dir)})
                if not result.get("ok"):
                    raise RuntimeError(result.get("error") or "PaddleOCR-VL worker error")
                page = VisualPage(
                    transcription=str(result.get("transcription") or "").strip(),
                    items=[],
                    uncertain_regions=[],
                    complete=True,
                    contains_chart=bool(result.get("contains_chart")),
                    chart_notes=str(result.get("chart_notes") or ""),
                )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            _cache_put(self.cache_dir, binding, page)
            return _traced(generation, page, cache_hit=False, model_cold_start=cold)


class LocalVisualRouter:
    """Route Arabic page text to EasyOCR; image regions (tables, charts, text in images) and French
    pages to PaddleOCR-VL. An Arabic image region gets both: EasyOCR text plus PaddleOCR-VL structure."""

    def __init__(self, cache_dir: str | Path, *, easy=None, paddle=None) -> None:
        root = Path(cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.cache_dir = root
        self._easy = easy or EasyOcrVisual(root / "easyocr")
        self._paddle = paddle or PaddleVlVisual(root / "paddleocr_vl")
        self.model = "local-visual-router"
        self.last_model = self.model

    def transcribe(
        self,
        *,
        image_png: bytes,
        source_pdf_sha256: str,
        page_number: int,
        language: str = "fr",
        image_region: bool = False,
        **_,
    ) -> VisualPage:
        arabic = (language or "").casefold().startswith("ar")
        if arabic and not image_region:
            self.last_model = self._easy.model
            return self._easy.transcribe(
                image_png=image_png, source_pdf_sha256=source_pdf_sha256, page_number=page_number
            )
        if arabic and image_region:
            self.last_model = self._easy.model
            body = self._easy.transcribe(
                image_png=image_png, source_pdf_sha256=source_pdf_sha256, page_number=page_number
            )
            try:
                charts = self._paddle.transcribe(
                    image_png=image_png, source_pdf_sha256=source_pdf_sha256, page_number=page_number
                )
            except Exception as error:
                if getattr(self._paddle, "killed", False):
                    raise  # interrupted for chat: re-read the whole page later
                get_client().create_event(
                    name="skip-chart-notes",
                    level="WARNING",
                    status_message=f"{type(error).__name__}: {error}",
                    metadata={"page": page_number},
                )
                return body
            self.last_model = f"{self._easy.model}+{self._paddle.model}"
            notes = (charts.chart_notes or charts.transcription or "").strip()
            if notes and notes not in body.transcription:
                return body.model_copy(
                    update={
                        "chart_notes": notes,
                        "contains_chart": True,
                        "transcription": f"{body.transcription}\n\n{notes}".strip(),
                    }
                )
            return body.model_copy(update={"contains_chart": True, "chart_notes": notes})
        self.last_model = self._paddle.model
        return self._paddle.transcribe(
            image_png=image_png, source_pdf_sha256=source_pdf_sha256, page_number=page_number
        )

    def kill(self) -> None:
        self._paddle.kill()

    def close(self) -> None:
        """Release OCR models (Paddle worker process, EasyOCR/Torch cache) when the queue is idle."""
        self._paddle.close()
        if getattr(self._easy, "_reader", None) is not None:
            self._easy._reader = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def visual_backend_name(profile: str | None = None) -> str:
    configured = (os.environ.get("BCT_VISUAL_BACKEND") or "").strip().casefold()
    if configured:
        return configured
    active = (profile or os.environ.get("BCT_DEFAULT_PROFILE") or "local_hybrid").strip().casefold()
    return "gemini" if active == "cloud" else "local"


def build_visual_transcriber(cache_dir: str | Path, *, profile: str | None = None):
    """Return Gemini or local router, or None when visual ingest is off."""
    backend = visual_backend_name(profile)
    if backend in {"off", "0", "none", "false"}:
        return None
    if os.environ.get("BCT_GEMINI_VISUAL", "1") != "1" and backend == "gemini":
        return None
    if backend == "gemini":
        from .gemini_visual import GeminiVisualTranscriber

        return GeminiVisualTranscriber(cache_dir)
    # local (default for local / local_hybrid)
    if os.environ.get("BCT_GEMINI_VISUAL", "1") == "0" and not (os.environ.get("BCT_VISUAL_BACKEND") or "").strip():
        # legacy kill-switch with no explicit backend still disables visual
        return None
    root = Path(cache_dir)
    # Prefer dedicated local cache; fall back beside gemini-cache.
    local_root = Path(os.environ.get("BCT_VISUAL_CACHE", str(root.parent / "visual-cache" if root.name == "gemini-cache" else root)))
    return LocalVisualRouter(local_root)
