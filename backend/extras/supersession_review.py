"""Administrator decisions on supersession edges: approve, reject, or add one by hand.

Edges found in the PDFs (supersession_edges.jsonl, rebuilt with every index version) are used as
soon as they are found, and listed as "to review". An administrator's decision is stored here,
in the ingestion database, keyed by the edge itself, so a re-upload or a rebuilt index keeps it.
Search uses: found edges - rejected + added (effective_edges).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from extras.sqlite_local import thread_connection
from extras.supersession_edges import INSTR, SupersessionEdge, edge_key, instruments_from_text

STATUSES = ("approved", "rejected", "added")
_locals: dict[str, threading.local] = {}  # one per database path (tests use several)
_cache: dict | None = None  # decisions, read once per process and dropped on every write


def edge_id(edge: SupersessionEdge) -> str:
    """A short stable id for URLs and the admin page."""
    return hashlib.sha1(json.dumps(edge_key(edge), ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _connection():
    path = os.environ.get("BCT_INGESTION_DB")
    if not path:
        return None
    connection = thread_connection(_locals.setdefault(path, threading.local()), Path(path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS supersession_decisions (
            edge_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            edge_json TEXT NOT NULL,
            decided_by TEXT NOT NULL,
            decided_at TEXT NOT NULL
        )
        """
    )
    return connection


def decisions() -> dict[str, dict]:
    """edge id -> {"status", "edge", "decided_by", "decided_at"}."""
    global _cache
    path = os.environ.get("BCT_INGESTION_DB", "")
    if _cache is None or _cache.get("path") != path:
        connection = _connection()
        rows = connection.execute("SELECT * FROM supersession_decisions").fetchall() if connection else []
        _cache = {"path": path, "data": {
            row["edge_id"]: {
                "status": row["status"],
                "edge": SupersessionEdge(**json.loads(row["edge_json"])),
                "decided_by": row["decided_by"],
                "decided_at": row["decided_at"],
            }
            for row in rows
        }}
    return _cache["data"]


def decide(edge: SupersessionEdge, status: str | None, admin_email: str) -> None:
    """Record a decision; None takes it back (a found edge returns to "to review", an added one goes)."""
    global _cache
    if status is not None and status not in STATUSES:
        raise ValueError(f"Unknown decision: {status}")
    connection = _connection()
    if connection is None:
        raise RuntimeError("BCT_INGESTION_DB is not configured")
    if status is None:
        connection.execute("DELETE FROM supersession_decisions WHERE edge_id = ?", (edge_id(edge),))
    else:
        connection.execute(
            "INSERT OR REPLACE INTO supersession_decisions VALUES (?, ?, ?, ?, ?)",
            (edge_id(edge), status, json.dumps(asdict(edge), ensure_ascii=False), admin_email,
             datetime.now(timezone.utc).isoformat()),
        )
    connection.commit()
    _cache = None


def _added(found: list[SupersessionEdge], decided: dict) -> list[SupersessionEdge]:
    """Edges the admin added that the PDFs do not already give (no duplicate after a re-index)."""
    found_ids = {edge_id(edge) for edge in found}
    return [item["edge"] for key, item in decided.items() if item["status"] == "added" and key not in found_ids]


def effective_edges(found: list[SupersessionEdge]) -> list[SupersessionEdge]:
    """The edges search uses: found ones the admin did not reject, plus the ones they added."""
    decided = decisions()
    kept = [edge for edge in found if decided.get(edge_id(edge), {}).get("status") != "rejected"]
    return kept + _added(found, decided)


def listing(found: list[SupersessionEdge]) -> list[dict]:
    """Every edge for the admin page, found or added, with its status ("review" if undecided)."""
    decided = decisions()
    rows = []
    for edge in found + _added(found, decided):
        decision = decided.get(edge_id(edge), {})
        rows.append({
            "id": edge_id(edge),
            **asdict(edge),
            "status": decision.get("status", "review"),
            "decided_by": decision.get("decided_by"),
            "decided_at": decision.get("decided_at"),
        })
    return rows


def page_text(native_chunks: Path, source_file: str, page: int) -> str:
    """A page's text from the index chunks (empty when the index has no such page)."""
    name = Path(source_file).name.casefold()
    parts = []
    with native_chunks.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row["metadata"]
            if Path(str(meta.get("source"))).name.casefold() == name and int(meta.get("page", -1)) == page:
                parts.append(row["page_content"])
    return "\n".join(parts)


_SENTENCE_ENDS = (".", ";", "\n")


def declaring_sentence(text: str, target_instrument: str) -> str | None:
    """The sentence of a page that names the target instrument: the quote of a hand-added edge."""
    for match in INSTR.finditer(text):
        if target_instrument in instruments_from_text(match.group(0)):
            start = max(text.rfind(mark, 0, match.start()) for mark in _SENTENCE_ENDS) + 1
            ends = [i for i in (text.find(mark, match.end()) for mark in _SENTENCE_ENDS) if i != -1]
            return " ".join(text[start:min(ends) + 1 if ends else len(text)].split())
    return None
