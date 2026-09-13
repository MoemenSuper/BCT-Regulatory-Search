from pathlib import Path

from ingestion.registry import IngestionRegistry


def test_new_ready_version_supersedes_previous_same_filename(tmp_path: Path):
    registry = IngestionRegistry(tmp_path / "ingestion.sqlite3")
    try:
        registry.start("hash-old", "Cir_2026_01_fr.pdf", "/old.pdf")
        registry.ready("hash-old", stored_path="/old.pdf", asset_version="v1", report={"version": 1})
        registry.start("hash-new", "Cir_2026_01_fr.pdf", "/new.pdf")
        registry.ready("hash-new", stored_path="/new.pdf", asset_version="v2", report={"version": 2})

        assert registry.get("hash-new")["status"] == "ready"
        assert registry.get("hash-old")["status"] == "superseded"
        ready = registry.list_ready()
        assert [row["content_sha256"] for row in ready] == ["hash-new"]
    finally:
        registry.close()
