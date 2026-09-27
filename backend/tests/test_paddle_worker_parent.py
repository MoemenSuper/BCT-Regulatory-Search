"""The PaddleOCR-VL worker exits when the API that spawned it dies (no orphan holding GPU memory)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

INGESTION = Path(__file__).resolve().parents[1] / "ingestion"


def test_worker_exits_when_parent_is_killed():
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time; sys.path.insert(0, sys.argv[1]); import paddle_vl_worker as w; "
            "w._exit_with_parent(); time.sleep(60)",
            str(INGESTION),
        ],
        env={**os.environ, "BCT_PARENT_PID": str(parent.pid)},
    )
    try:
        parent.kill()
        parent.wait(10)
        assert child.wait(timeout=15) == 0
    finally:
        for process in (parent, child):
            if process.poll() is None:
                process.kill()
