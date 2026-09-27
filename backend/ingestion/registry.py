from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


# Statuses whose chunks are live in the active asset version.
SEARCHABLE = ("enriching", "ready", "ready_degraded")
_SEARCHABLE_SQL = "('enriching', 'ready', 'ready_degraded')"


class IngestionRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_pages (
                content_sha256 TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                language TEXT NOT NULL,
                chart_suspect INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                model TEXT,
                error TEXT,
                seconds REAL,
                updated_at TEXT,
                PRIMARY KEY (content_sha256, page_number)
            )
            """
        )
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

    @property
    def connection(self) -> sqlite3.Connection:
        # One connection per thread: request handlers and the enrichment worker share this registry.
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

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

    def ready(
        self,
        content_sha256: str,
        *,
        stored_path: str,
        asset_version: str,
        report: dict,
        status: str = "ready",
    ) -> None:
        if status not in SEARCHABLE:
            raise ValueError(f"Not a searchable status: {status}")
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
            SET stored_path=?, status=?, asset_version=?, report_json=?, error=NULL, activated_at=?
            WHERE content_sha256=?
            """,
            (
                stored_path,
                status,
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
            WHERE content_sha256<>? AND status IN """ + _SEARCHABLE_SQL + """ AND lower(original_filename)=lower(?)
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

    def mark_removed(self, content_sha256: str, *, asset_version: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.connection.execute(
            """
            UPDATE ingestion_documents
            SET status='removed', asset_version=?, error=NULL, activated_at=?
            WHERE content_sha256=? AND status IN """ + _SEARCHABLE_SQL + """
            """,
            (asset_version, now, content_sha256),
        )
        self.connection.execute("DELETE FROM ingestion_pages WHERE content_sha256=?", (content_sha256,))
        self.connection.commit()
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown ready ingestion document: {content_sha256}")

    # ---- background enrichment (per-page visual reading) ----

    def seed_pages(self, content_sha256: str, plans: list[dict]) -> None:
        """Register pages awaiting visual reading; existing rows (resume) are kept."""
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO ingestion_pages
                (content_sha256, page_number, priority, language, chart_suspect)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (content_sha256, int(p["page"]), int(p["priority"]), str(p["language"]), int(bool(p["chart_suspect"])))
                for p in plans
            ],
        )
        self.connection.commit()

    def pending_pages(self, content_sha256: str) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT page_number, priority, language, chart_suspect, attempts FROM ingestion_pages
            WHERE content_sha256=? AND state='pending' ORDER BY priority, page_number
            """,
            (content_sha256,),
        ).fetchall()
        return [dict(row) for row in rows]

    def page_results(self, content_sha256: str) -> dict[int, tuple[str, str | None, str | None]]:
        """``{page: ("done", result_json, model) | ("failed", error, model)}`` for settled pages."""
        rows = self.connection.execute(
            "SELECT page_number, state, result_json, error, model FROM ingestion_pages "
            "WHERE content_sha256=? AND state IN ('done', 'failed')",
            (content_sha256,),
        ).fetchall()
        return {
            int(row["page_number"]): (
                row["state"],
                row["result_json"] if row["state"] == "done" else row["error"],
                row["model"],
            )
            for row in rows
        }

    def page_done(self, content_sha256: str, page: int, *, result_json: str, model: str | None, seconds: float) -> None:
        self.connection.execute(
            """
            UPDATE ingestion_pages SET state='done', result_json=?, model=?, error=NULL, seconds=?,
                attempts=attempts+1, updated_at=?
            WHERE content_sha256=? AND page_number=?
            """,
            (result_json, model, seconds, datetime.now(timezone.utc).isoformat(), content_sha256, page),
        )
        self.connection.commit()

    def page_failed(self, content_sha256: str, page: int, *, error: str, max_attempts: int) -> None:
        self.connection.execute(
            """
            UPDATE ingestion_pages SET attempts=attempts+1, error=?, updated_at=?,
                state=CASE WHEN attempts+1 >= ? THEN 'failed' ELSE 'pending' END
            WHERE content_sha256=? AND page_number=?
            """,
            (error[:2000], datetime.now(timezone.utc).isoformat(), max_attempts, content_sha256, page),
        )
        self.connection.commit()

    def retry_failed_pages(self, content_sha256: str) -> int:
        cursor = self.connection.execute(
            "UPDATE ingestion_pages SET state='pending', attempts=0, error=NULL "
            "WHERE content_sha256=? AND state='failed'",
            (content_sha256,),
        )
        if cursor.rowcount:
            self.connection.execute(
                "UPDATE ingestion_documents SET status='enriching' WHERE content_sha256=? AND status='ready_degraded'",
                (content_sha256,),
            )
        self.connection.commit()
        return cursor.rowcount

    def progress(self, content_sha256: str) -> dict:
        row = self.connection.execute(
            """
            SELECT count(*) AS total,
                   sum(state='done') AS done,
                   sum(state='failed') AS failed,
                   sum(state='pending') AS pending,
                   avg(CASE WHEN state='done' AND seconds IS NOT NULL THEN seconds END) AS avg_seconds
            FROM ingestion_pages WHERE content_sha256=?
            """,
            (content_sha256,),
        ).fetchone()
        total, done, failed, pending = (int(row[key] or 0) for key in ("total", "done", "failed", "pending"))
        average = row["avg_seconds"]
        return {
            "total": total,
            "done": done,
            "failed": failed,
            "pending": pending,
            "seconds_per_page": round(float(average), 1) if average else None,
            "eta_seconds": int(average * pending) if average and pending else None,
        }

    def next_enriching(self, *, exclude: set[str] = frozenset()) -> dict | None:
        rows = self.connection.execute(
            "SELECT content_sha256 FROM ingestion_documents WHERE status='enriching' ORDER BY created_at"
        ).fetchall()
        for row in rows:
            if row[0] not in exclude:
                return self.get(row[0])
        return None

    def fail_interrupted(self) -> int:
        """A quick pass killed mid-way (process exit) never finished; say so."""
        cursor = self.connection.execute(
            "UPDATE ingestion_documents SET status='failed', error='Interrupted: the server stopped during ingestion' "
            "WHERE status='processing'"
        )
        self.connection.commit()
        return cursor.rowcount

    def list_ready(self, limit: int = 100) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT content_sha256, original_filename, stored_path, status, asset_version,
                   report_json, created_at, activated_at
            FROM ingestion_documents
            WHERE status IN """ + _SEARCHABLE_SQL + """
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
            from document_authority import resolve_doc_kind

            explicit_kind = admin_meta.get("doc_kind")
            if not explicit_kind and isinstance(report, dict):
                explicit_kind = report.get("doc_kind")
            doc_kind = resolve_doc_kind(
                explicit=explicit_kind,
                filename=str(item["original_filename"] or ""),
            )
            documents.append(
                {
                    "document_id": item["content_sha256"],
                    "filename": item["original_filename"],
                    "title": title or item["original_filename"],
                    "status": item["status"],
                    "searchable": bool(report.get("searchable", True)) if isinstance(report, dict) else True,
                    "asset_version": item["asset_version"],
                    "activated_at": item["activated_at"],
                    "pages": report.get("pages") if isinstance(report, dict) else None,
                    "doc_kind": doc_kind,
                    "publication_date": str(admin_meta.get("publication_date") or ""),
                    "document_type": str(
                        admin_meta.get("document_type") or admin_meta.get("type") or ""
                    ),
                    "category": str(admin_meta.get("category") or ""),
                    "document_number": str(admin_meta.get("document_number") or ""),
                    "enrichment": self.progress(item["content_sha256"]),
                }
            )
        return documents
