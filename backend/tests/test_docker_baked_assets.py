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


def _old_volume(assets: Path) -> None:
    assets.mkdir()
    (assets / "ACTIVE.json").write_text(
        json.dumps({"active_version": ".", "activated_at": "already"}),
        encoding="utf-8",
    )
    (assets / "native.jsonl").write_text("keep-me\n", encoding="utf-8")


def test_ensure_runtime_assets_replaces_an_empty_stub_volume(tmp_path: Path, monkeypatch):
    """An image without a baked index left empty stubs: every question found nothing."""
    baked = tmp_path / "baked"
    assets = tmp_path / "assets"
    monkeypatch.setenv("BCT_BAKED_ASSETS_DIR", str(tmp_path / "missing-baked"))
    docker_serve.ensure_runtime_assets(assets)
    _write_baked(baked)
    monkeypatch.setenv("BCT_BAKED_ASSETS_DIR", str(baked))

    assert docker_serve.ensure_runtime_assets(assets) is True

    active = docker_serve.resolve_active_assets(assets)
    assert "hello" in (active / "native.jsonl").read_text(encoding="utf-8")


def test_ensure_runtime_assets_replaces_an_older_index(tmp_path: Path, monkeypatch):
    baked = tmp_path / "baked"
    assets = tmp_path / "assets"
    _write_baked(baked)
    _old_volume(assets)
    monkeypatch.setenv("BCT_BAKED_ASSETS_DIR", str(baked))

    assert docker_serve.ensure_runtime_assets(assets) is True

    active = docker_serve.resolve_active_assets(assets)
    assert "hello" in (active / "native.jsonl").read_text(encoding="utf-8")
    assert (assets / docker_serve.SEEDED_FROM).read_text(encoding="utf-8") == "versions/baked-active"
    # The old index is set aside, not deleted.
    [kept] = list(assets.glob("replaced-*"))
    assert (kept / "native.jsonl").read_text(encoding="utf-8") == "keep-me\n"

    # Next boot with the same image: nothing to do.
    assert docker_serve.ensure_runtime_assets(assets) is False
    assert len(list(assets.glob("replaced-*"))) == 1


def test_ensure_runtime_assets_falls_back_to_empty_stub(tmp_path: Path, monkeypatch):
    assets = tmp_path / "assets"
    monkeypatch.setenv("BCT_BAKED_ASSETS_DIR", str(tmp_path / "missing-baked"))

    docker_serve.ensure_runtime_assets(assets)

    assert (assets / "native.jsonl").is_file()
    assert (assets / "snapshot.json").is_file()
