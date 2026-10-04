"""The server's log file: every question, upload and failure leaves a line an admin can send.

<data dir>/logs/bct.log, rotated at 5 MB (5 old files kept), and the same lines on the console
(`docker compose logs app`).
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

# Libraries that log every HTTP call or model load at INFO.
_QUIET = ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "huggingface_hub", "langfuse", "groq")


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
    return path
