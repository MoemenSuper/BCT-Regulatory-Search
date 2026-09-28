"""PaddleOCR-VL worker process (GPU). Kept separate from Torch/EasyOCR to avoid Windows cuDNN clashes."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def _ensure_nvidia_path() -> None:
    try:
        import site

        roots = [Path(p) for p in site.getsitepackages()]
    except Exception:
        roots = [Path(sys.prefix) / "Lib" / "site-packages"]
    bins = []
    for root in roots:
        bins.extend(str(p) for p in sorted(root.glob("nvidia/*/bin")))
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins + [os.environ.get("PATH", "")])


def _block_modelscope_torch() -> None:
    """Stop ModelScope from importing real Torch (cuDNN clash with paddlepaddle-gpu on Windows)."""
    import importlib.util as iutil

    real_find_spec = iutil.find_spec

    def find_spec(name, package=None):  # noqa: ANN001
        if name == "torch" or (isinstance(name, str) and name.startswith("torch.")):
            return None
        return real_find_spec(name, package)

    iutil.find_spec = find_spec  # type: ignore[assignment]


def _first_to_go_when_memory_runs_out() -> None:
    """Linux: if memory runs out anyway, the kernel kills this reader, not the API that serves chat."""
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass  # not Linux, or not allowed: nothing to adjust


def _exit_with_parent() -> None:
    """A hard-killed API must not leave this process holding GPU memory (next worker would crash)."""
    parent = int(os.environ.get("BCT_PARENT_PID") or 0)
    if not parent:
        return
    import threading
    import time

    def watch() -> None:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x00100000, False, parent)  # SYNCHRONIZE
            if handle:
                kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        else:
            while os.getppid() == parent:
                time.sleep(2)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def main() -> int:
    _first_to_go_when_memory_runs_out()
    _exit_with_parent()
    _ensure_nvidia_path()
    _block_modelscope_torch()
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
    device = os.environ.get("BCT_PADDLE_DEVICE", "gpu")

    import paddle
    from paddleocr import PaddleOCRVL

    if device.startswith("gpu") and not (paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count()):
        device = "cpu"  # CPU-only machine or CPU wheel: slower, same model
    paddle.device.set_device(device if device.startswith("gpu") or device == "cpu" else "gpu")
    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        use_chart_recognition=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        device=device,
    )
    sys.stdout.write(json.dumps({"ok": True, "device": str(paddle.device.get_device())}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req.get("cmd")
        if cmd == "quit":
            sys.stdout.write(json.dumps({"ok": True, "quit": True}) + "\n")
            sys.stdout.flush()
            return 0
        if cmd != "predict":
            sys.stdout.write(json.dumps({"ok": False, "error": f"unknown cmd {cmd}"}) + "\n")
            sys.stdout.flush()
            continue
        png_path = Path(req["png"])
        out_dir = Path(req["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            outputs = list(pipeline.predict(str(png_path)))
            for res in outputs:
                if hasattr(res, "save_to_markdown"):
                    res.save_to_markdown(save_path=str(out_dir))
            text = "\n\n".join(
                path.read_text(encoding="utf-8", errors="replace") for path in sorted(out_dir.rglob("*.md"))
            )
            flat = re.sub(r"<[^>]+>", " ", text)
            flat = re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", " ", flat)).strip()
            has_chart = bool(re.search(r"(?i)graph|graphique|chart|tableau|table", text))
            sys.stdout.write(
                json.dumps(
                    {
                        "ok": True,
                        "transcription": flat or text.strip(),
                        "contains_chart": has_chart,
                        "chart_notes": flat if has_chart else "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        except Exception as exc:  # noqa: BLE001 — worker boundary
            sys.stdout.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
