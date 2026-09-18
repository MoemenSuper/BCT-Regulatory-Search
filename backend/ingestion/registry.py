from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class IngestionRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_documents (
                content_sha256 TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_path TEXT,
                status TEXT NOT NULL,
                asset_version TEXT,
                report_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                activated_at TEXT
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, content_sha256: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM ingestion_documents WHERE content_sha256 = ?",
            (content_sha256,),
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        if value.get("report_json"):
            value["report"] = json.loads(value["report_json"])
        return value

    def start(self, content_sha256: str, filename: str, stored_path: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            INSERT INTO ingestion_documents (
                content_sha256, original_filename, stored_path, status, created_at
            ) VALUES (?, ?, ?, 'processing', ?)
            ON CONFLICT(content_sha256) DO UPDATE SET
                original_filename=excluded.original_filename,
                stored_path=coalesce(excluded.stored_path, ingestion_documents.stored_path),
                status='processing', error=NULL
            """,
            (content_sha256, filename, stored_path, now),
        )
        self.connection.commit()

    def ready(self, content_sha256: str, *, stored_path: str, asset_version: str, report: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        row = self.connection.execute(
            "SELECT original_filename FROM ingestion_documents WHERE content_sha256=?",
            (content_sha256,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown ingestion document: {content_sha256}")
        filename = str(row[0])
        self.connection.execute(
            """
            UPDATE ingestion_documents
            SET stored_path=?, status='ready', asset_version=?, report_json=?, error=NULL, activated_at=?
            WHERE content_sha256=?
            """,
            (
                stored_path,
                asset_version,
                json.dumps(report, ensure_ascii=False, sort_keys=True),
                now,
                content_sha256,
            ),
        )
        # Only the newest byte-version of one logical source basename is active in
        # the retrieval assets. Keep the older ledger rows for audit, but do not
        # present them as simultaneously ready/searchable.
        self.connection.execute(
            """
            UPDATE ingestion_documents
            SET status='superseded'
            WHERE content_sha256<>? AND status='ready' AND lower(original_filename)=lower(?)
            """,
            (content_sha256, filename),
        )
        self.connection.commit()

    def fail(self, content_sha256: str, error: str) -> None:
        self.connection.execute(
            "UPDATE ingestion_documents SET status='failed', error=? WHERE content_sha256=?",
            (error[:4000], content_sha256),
        )
        self.connection.commit()

    def list_ready(self, limit: int = 100) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT content_sha256, original_filename, stored_path, status, asset_version,
                   report_json, created_at, activated_at
            FROM ingestion_documents
            WHERE status='ready'
            ORDER BY activated_at DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
        documents = []
        for row in rows:
            item = dict(row)
            report = {}
            if item.get("report_json"):
                try:
                    report = json.loads(item["report_json"]) or {}
                except json.JSONDecodeError:
                    report = {}
            title = ""
            admin_meta = {}
            if isinstance(report, dict):
                raw_meta = report.get("administrator_metadata")
                if isinstance(raw_meta, dict):
                    admin_meta = raw_meta
                    title = str(admin_meta.get("title") or "")
                if not title:
                    title = str(report.get("title") or "")
            documents.append(
                {
                    "document_id": item["content_sha256"],
                    "filename": item["original_filename"],
                    "title": title or item["original_filename"],
                    "status": item["status"],
                    "asset_version": item["asset_version"],
                    "activated_at": item["activated_at"],
                    "pages": report.get("pages") if isinstance(report, dict) else None,
                    "publication_date": str(admin_meta.get("publication_date") or ""),
                    "document_type": str(
                        admin_meta.get("document_type") or admin_meta.get("type") or ""
                    ),
                    "category": str(admin_meta.get("category") or ""),
                    "document_number": str(admin_meta.get("document_number") or ""),
                }
            )
        return documents
