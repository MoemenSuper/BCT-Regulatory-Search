"""The server's log file: every question, upload and failure leaves a line an admin can send.

<data dir>/logs/bct.log, rotated at 5 MB (5 old files kept), and the same lines on the console
(`docker compose logs app`). Crashes are caught everywhere: in a request (app.py gives the user
a reference to find the line), in a background thread, and at startup (uvicorn logs here too).
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path

# Libraries that log every HTTP call or model load at INFO; uvicorn.access logs every request.
_QUIET = ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "huggingface_hub", "groq",
          "uvicorn.access")


def log_file(data_dir: Path) -> Path:
    return data_dir / "logs" / "bct.log"


def configure_logging(data_dir: Path) -> Path:
    path = log_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (
        logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    for name in _QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
    _log_uncaught_errors()
    return path


def _log_uncaught_errors() -> None:
    """A crash outside a request (background indexing thread, startup) lands in the log file too."""
    crash = logging.getLogger("crash")

    def in_thread(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not SystemExit:
            crash.error("Uncaught error in thread %s", getattr(args.thread, "name", "?"),
                        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    threading.excepthook = in_thread
    sys.excepthook = lambda kind, value, trace: crash.error("Uncaught error", exc_info=(kind, value, trace))
