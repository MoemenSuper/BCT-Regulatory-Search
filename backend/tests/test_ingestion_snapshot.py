import json
from pathlib import Path

from ingestion.pipeline import _read_snapshot


def test_read_snapshot_coerces_legacy_list(tmp_path: Path):
    active = tmp_path / "active"
    active.mkdir()
    (active / "snapshot.json").write_text(json.dumps(["legacy", "bindings"]), encoding="utf-8")
    assert _read_snapshot(active) == {}


def test_read_snapshot_keeps_dict(tmp_path: Path):
    active = tmp_path / "active"
    active.mkdir()
    (active / "snapshot.json").write_text(json.dumps({"local_primary": "x"}), encoding="utf-8")
    assert _read_snapshot(active)["local_primary"] == "x"


def test_read_snapshot_missing(tmp_path: Path):
    assert _read_snapshot(tmp_path / "missing") == {}
