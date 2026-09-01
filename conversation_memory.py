import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4


def new_memory_state():
    return {
        "topics": [],
        "first_topic": None,
        "current_topic": None,
        "turns": [],
    }


def default_conversation_database_path():
    local_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_data) if local_data else Path.home() / ".local" / "share"
    return root / "BCT-Regulatory-Search" / "conversations.sqlite3"


def open_conversation_store():
    configured_path = os.environ.get("BCT_CONVERSATION_DB")
    return ConversationStore(configured_path or default_conversation_database_path())


class ConversationStore:
    def __init__(self, path, *, max_turns=6):
        self.path = Path(path)
        self.max_turns = max_turns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    conversation_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create(self):
        conversation_id = str(uuid4())
        state = new_memory_state()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_sessions (conversation_id, state_json) "
                "VALUES (?, ?)",
                (conversation_id, self._serialize(state)),
            )
        return conversation_id

    def load(self, conversation_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM conversation_sessions "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, conversation_id, state):
        value = dict(state)
        value["turns"] = list(value.get("turns", []))[-self.max_turns :]
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET state_json = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                (self._serialize(value), conversation_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown conversation: {conversation_id}")

    def close(self):
        # Connections are intentionally scoped to each operation so FastAPI
        # worker threads never share a sqlite connection.
        return None

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    @staticmethod
    def _serialize(state):
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
