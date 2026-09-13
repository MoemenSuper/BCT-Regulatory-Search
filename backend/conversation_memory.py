import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4


def new_memory_state():
    return {
        "topics": [],
        "first_topic": None,
        "current_topic": None,
        "turns": [],
    }


_WHITESPACE = re.compile(r"\s+")
_LEADING_FILLERS = (
    r"(?:s'?il vous pla[iî]t|svp|please|bonjour|bonsoir|salut|hello|hi)\b[,:\s]*",
    r"(?:pouvez[- ]vous|pourriez[- ]vous|peux[- ]tu|pourrais[- ]tu|can you|could you|would you)\b[,:\s]*",
    r"(?:me\s+)?(?:dire|expliquer|indiquer|préciser|clarifier|rappeler|tell me|explain|describe)\b[,:\s]*",
    r"(?:je\s+(?:voudrais|veux|souhaite)|j'aimerais|I'd like|I want|I need)\b(?:\s+(?:savoir|connaître|obtenir))?\b[,:\s]*",
    r"(?:est[- ]ce\s+que|est[- ]ce\s+qu')\b[,:\s]*",
    r"(?:quelle\s+est|quelles\s+sont|quel\s+est|quels\s+sont|qu'est[- ]ce\s+que|qu'est[- ]ce\s+qu')\b[,:\s]*",
    r"(?:what\s+(?:is|are)|how\s+(?:do|does|can|to)|why\s+(?:is|are|do|does))\b[,:\s]*",
    r"(?:comment|pourquoi|quand|où|combien)\b[,:\s]*",
)
_LEADING_RE = [re.compile(rf"^{pattern}", re.IGNORECASE) for pattern in _LEADING_FILLERS]
_TRAILING_PUNCT = re.compile(r"[?؟!.…]+$")


def summarize_conversation_title(question: str, *, max_length: int = 72) -> str:
    """Turn a raw question into a short sidebar title."""
    text = _WHITESPACE.sub(" ", str(question or "").strip())
    if not text:
        return "Recherche réglementaire"

    text = _TRAILING_PUNCT.sub("", text).strip()
    changed = True
    while changed:
        changed = False
        for pattern in _LEADING_RE:
            updated = pattern.sub("", text).strip(" ,:;-")
            if updated != text:
                text = updated
                changed = True

    text = _TRAILING_PUNCT.sub("", text).strip(" ,:;-")
    if not text:
        text = _WHITESPACE.sub(" ", str(question).strip())
        text = _TRAILING_PUNCT.sub("", text).strip() or "Recherche réglementaire"

    text = re.sub(
        r"^(?:les?\s+|la\s+|l'|des?\s+|du\s+|d'|un\s+|une\s+|the\s+|an?\s+)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if text:
        text = text[0].upper() + text[1:]

    if len(text) <= max_length:
        return text

    cut = text[: max_length - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return f"{cut.rstrip(' ,;:-')}…"


def default_conversation_database_path():
    local_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_data) if local_data else Path.home() / ".local" / "share"
    return root / "BCT-Regulatory-Search" / "conversations.sqlite3"


def open_conversation_store():
    configured_path = os.environ.get("BCT_CONVERSATION_DB")
    return ConversationStore(configured_path or default_conversation_database_path())


class ConversationStore:
    """Durable routing memory plus a full UI transcript.

    The LLM routing state intentionally keeps only a bounded number of recent turns,
    while `conversation_turns` retains the complete transcript for the history UI.
    """

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
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    title TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    standalone_query TEXT,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    graph_trace_json TEXT NOT NULL,
                    profile TEXT,
                    answer_status TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversation_turns_session_time "
                "ON conversation_turns(conversation_id, created_at, turn_id)"
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
        value = self._bounded_state(state)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET state_json = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                (self._serialize(value), conversation_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown conversation: {conversation_id}")

    def save_with_turn(
        self,
        conversation_id,
        state,
        *,
        question,
        standalone_query=None,
        answer="",
        sources=None,
        graph_trace=None,
        profile=None,
        answer_status=None,
    ):
        value = self._bounded_state(state)
        turn_id = str(uuid4())
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT title FROM conversation_sessions WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown conversation: {conversation_id}")
            title = existing[0] or summarize_conversation_title(question)
            cursor = connection.execute(
                "UPDATE conversation_sessions SET state_json = ?, title = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE conversation_id = ?",
                (self._serialize(value), title, conversation_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown conversation: {conversation_id}")
            connection.execute(
                """
                INSERT INTO conversation_turns (
                    turn_id, conversation_id, question, standalone_query, answer,
                    sources_json, graph_trace_json, profile, answer_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    conversation_id,
                    str(question),
                    str(standalone_query or question),
                    str(answer or ""),
                    json.dumps(list(sources or []), ensure_ascii=False, sort_keys=True),
                    json.dumps(dict(graph_trace or {}), ensure_ascii=False, sort_keys=True),
                    profile,
                    answer_status,
                ),
            )
        return turn_id

    def rename(self, conversation_id, title):
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET title = ? WHERE conversation_id = ?",
                (str(title).strip()[:120], conversation_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown conversation: {conversation_id}")

    def delete(self, conversation_id):
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversation_turns WHERE conversation_id = ?", (conversation_id,)
            )
            cursor = connection.execute(
                "DELETE FROM conversation_sessions WHERE conversation_id = ?", (conversation_id,)
            )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown conversation: {conversation_id}")

    def list_conversations(self, *, limit=100):
        limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                WITH ranked AS (
                    SELECT conversation_id, question, created_at,
                           ROW_NUMBER() OVER (
                               PARTITION BY conversation_id
                               ORDER BY created_at ASC, rowid ASC
                           ) AS rn_asc,
                           ROW_NUMBER() OVER (
                               PARTITION BY conversation_id
                               ORDER BY created_at DESC, rowid DESC
                           ) AS rn_desc
                    FROM conversation_turns
                ),
                agg AS (
                    SELECT conversation_id,
                           MAX(CASE WHEN rn_asc = 1 THEN question END) AS first_question,
                           MAX(CASE WHEN rn_desc = 1 THEN question END) AS last_question,
                           COUNT(*) AS turn_count,
                           MAX(created_at) AS last_turn_at
                    FROM ranked
                    GROUP BY conversation_id
                )
                SELECT
                    s.conversation_id,
                    s.updated_at,
                    s.title,
                    a.first_question,
                    a.last_question,
                    COALESCE(a.turn_count, 0) AS turn_count,
                    a.last_turn_at
                FROM conversation_sessions s
                LEFT JOIN agg a ON a.conversation_id = s.conversation_id
                WHERE a.first_question IS NOT NULL OR a.last_question IS NOT NULL
                ORDER BY COALESCE(a.last_turn_at, s.updated_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result = []
        for row in rows:
            first_question = row["first_question"]
            last_question = row["last_question"]
            title = str(row["title"] or "").strip()
            if not title:
                title = summarize_conversation_title(first_question or last_question or "")
            result.append(
                {
                    "conversation_id": row["conversation_id"],
                    "title": title[:120],
                    "last_question": str(last_question or title),
                    "turn_count": int(row["turn_count"] or 0),
                    "updated_at": str(row["last_turn_at"] or row["updated_at"]),
                }
            )
        return result

    def transcript(self, conversation_id):
        state = self.load(conversation_id)
        if state is None:
            return None
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            title_row = connection.execute(
                "SELECT title FROM conversation_sessions WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT turn_id, question, standalone_query, answer, sources_json,
                       graph_trace_json, profile, answer_status, created_at
                FROM conversation_turns
                WHERE conversation_id=?
                ORDER BY created_at ASC, rowid ASC
                """,
                (conversation_id,),
            ).fetchall()

        turns = [
            {
                "turn_id": row["turn_id"],
                "question": row["question"],
                "standalone_query": row["standalone_query"],
                "answer": row["answer"],
                "sources": json.loads(row["sources_json"] or "[]"),
                "graph_trace": json.loads(row["graph_trace_json"] or "{}"),
                "profile": row["profile"],
                "answer_status": row["answer_status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        stored_title = (title_row["title"] if title_row else None) or ""
        if not stored_title and turns:
            stored_title = summarize_conversation_title(turns[0].get("question", ""))

        return {
            "conversation_id": conversation_id,
            "title": stored_title or "Recherche réglementaire",
            "memory_state": state,
            "turns": turns,
        }

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def _bounded_state(self, state):
        value = dict(state)
        value["turns"] = list(value.get("turns", []))[-self.max_turns :]
        return value

    @staticmethod
    def _serialize(state):
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
