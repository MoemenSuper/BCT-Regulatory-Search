"""Baked Voyage assets must seed an empty Docker assets volume on first boot."""

from pathlib import Path
import json

import docker_serve


def _write_baked(baked: Path) -> None:
    version = baked / "versions" / "baked-active"
    version.mkdir(parents=True)
    (version / "native.jsonl").write_text(
        json.dumps({"page_content": "hello", "metadata": {"source": "Cir_2020_01_fr.pdf", "page": 1}}) + "\n",
        encoding="utf-8",
    )
    (version / "arabic_ocr_secondary.jsonl").write_text("", encoding="utf-8")
    (version / "snapshot.json").write_text(json.dumps({"documents": 1}), encoding="utf-8")
    (version / "supersession_edges.jsonl").write_text("", encoding="utf-8")
    (version / "indexes").mkdir()
    (baked / "ACTIVE.json").write_text(
        json.dumps({"active_version": "versions/baked-active", "activated_at": "test"}),
        encoding="utf-8",
    )


def test_ensure_runtime_assets_seeds_from_baked_corpus(tmp_path: Path, monkeypatch):
    baked = tmp_path / "baked"
    assets = tmp_path / "assets"
    _write_baked(baked)
    monkeypatch.setenv("BCT_BAKED_ASSETS_DIR", str(baked))

    docker_serve.ensure_runtime_assets(assets)

    assert (assets / "ACTIVE.json").is_file()
    active = docker_serve.resolve_active_assets(assets)
    assert (active / "native.jsonl").read_text(encoding="utf-8").strip()
    assert '"documents": 1' in (active / "snapshot.json").read_text(encoding="utf-8")


def test_ensure_runtime_assets_keeps_existing_volume(tmp_path: Path, monkeypatch):
    baked = tmp_path / "baked"
    assets = tmp_path / "assets"
    _write_baked(baked)
    assets.mkdir()
    (assets / "ACTIVE.json").write_text(
        json.dumps({"active_version": ".", "activated_at": "already"}),
        encoding="utf-8",
    )
    (assets / "native.jsonl").write_text("keep-me\n", encoding="utf-8")
    monkeypatch.setenv("BCT_BAKED_ASSETS_DIR", str(baked))

    docker_serve.ensure_runtime_assets(assets)

    assert (assets / "native.jsonl").read_text(encoding="utf-8") == "keep-me\n"


def test_ensure_runtime_assets_falls_back_to_empty_stub(tmp_path: Path, monkeypatch):
    assets = tmp_path / "assets"
    monkeypatch.setenv("BCT_BAKED_ASSETS_DIR", str(tmp_path / "missing-baked"))

    docker_serve.ensure_runtime_assets(assets)

    assert (assets / "native.jsonl").is_file()
    assert (assets / "snapshot.json").is_file()
