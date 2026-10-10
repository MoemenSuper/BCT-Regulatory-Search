"""One SQLite connection per thread, shared by the app's small stores (auth, settings,
conversations, ingestion registry)."""

import sqlite3
import threading
from pathlib import Path


def thread_connection(local: threading.local, path: Path) -> sqlite3.Connection:
    """This thread's connection to the database at path, opened on first use.

    FastAPI runs sync endpoints on a thread pool, and ingestion has a background worker;
    one sqlite3 connection used from several threads races, so each thread gets its own.
    Rows come back as sqlite3.Row: read them by column name (row["email"]) or index (row[0]).
    """
    conn = getattr(local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # auth sessions are deleted with their user
        local.conn = conn
    return conn
