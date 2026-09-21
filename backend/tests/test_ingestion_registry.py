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
        assert [row["document_id"] for row in ready] == ["hash-new"]
        assert ready[0]["filename"] == "Cir_2026_01_fr.pdf"
        assert ready[0]["title"] == "Cir_2026_01_fr.pdf"
    finally:
        registry.close()


def test_list_ready_exposes_administrator_metadata(tmp_path: Path):
    registry = IngestionRegistry(tmp_path / "ingestion.sqlite3")
    try:
        registry.start("hash-meta", "Note_2024_03_fr.pdf", "/note.pdf")
        registry.ready(
            "hash-meta",
            stored_path="/note.pdf",
            asset_version="v1",
            report={
                "pages": 12,
                "administrator_metadata": {
                    "title": "Note prudentielle 2024-03",
                    "publication_date": "2024-03-01",
                    "type": "note",
                    "category": "Prudential",
                    "document_number": "2024-03",
                },
            },
        )
        ready = registry.list_ready()
        assert ready[0]["title"] == "Note prudentielle 2024-03"
        assert ready[0]["publication_date"] == "2024-03-01"
        assert ready[0]["document_type"] == "note"
        assert ready[0]["category"] == "Prudential"
        assert ready[0]["document_number"] == "2024-03"
        assert ready[0]["pages"] == 12
        assert ready[0]["doc_kind"] == "regulatory"
    finally:
        registry.close()


def test_list_ready_exposes_doc_kind(tmp_path: Path):
    registry = IngestionRegistry(tmp_path / "ingestion.sqlite3")
    try:
        registry.start("hash-stats", "Bulletin_2024_fr.pdf", "/b.pdf")
        registry.ready(
            "hash-stats",
            stored_path="/b.pdf",
            asset_version="v1",
            report={
                "administrator_metadata": {
                    "title": "Bulletin",
                    "doc_kind": "statistical",
                },
            },
        )
        registry.start("hash-memo", "Memo_interne.pdf", "/m.pdf")
        registry.ready(
            "hash-memo",
            stored_path="/m.pdf",
            asset_version="v1",
            report={"administrator_metadata": {"doc_kind": "internal"}},
        )
        by_id = {row["document_id"]: row for row in registry.list_ready()}
        assert by_id["hash-stats"]["doc_kind"] == "statistical"
        assert by_id["hash-memo"]["doc_kind"] == "internal"
    finally:
        registry.close()
