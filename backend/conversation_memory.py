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
                    title TEXT,
                    user_id TEXT
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS answer_refusals (
                    refusal_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    user_id TEXT,
                    user_email TEXT,
                    question TEXT NOT NULL,
                    answer_status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    profile TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_answer_refusals_time "
                "ON answer_refusals(created_at DESC, refusal_id)"
            )
            self._ensure_session_columns(connection)

    def create(self, user_id: str):
        owner = self._require_user_id(user_id)
        conversation_id = str(uuid4())
        state = new_memory_state()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_sessions (conversation_id, state_json, user_id) "
                "VALUES (?, ?, ?)",
                (conversation_id, self._serialize(state), owner),
            )
        return conversation_id

    def load(self, conversation_id, *, user_id: str):
        owner = self._require_user_id(user_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM conversation_sessions "
                "WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, owner),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, conversation_id, state, *, user_id: str):
        owner = self._require_user_id(user_id)
        value = self._bounded_state(state)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET state_json = ?, "
                "updated_at = CURRENT_TIMESTAMP "
                "WHERE conversation_id = ? AND user_id = ?",
                (self._serialize(value), conversation_id, owner),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown conversation: {conversation_id}")

    def save_with_turn(
        self,
        conversation_id,
        state,
        *,
        user_id: str,
        question,
        standalone_query=None,
        answer="",
        sources=None,
        graph_trace=None,
        profile=None,
        answer_status=None,
    ):
        owner = self._require_user_id(user_id)
        value = self._bounded_state(state)
        turn_id = str(uuid4())
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT title FROM conversation_sessions "
                "WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, owner),
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown conversation: {conversation_id}")
            title = existing[0] or summarize_conversation_title(question)
            cursor = connection.execute(
                "UPDATE conversation_sessions SET state_json = ?, title = ?, "
                "updated_at = CURRENT_TIMESTAMP "
                "WHERE conversation_id = ? AND user_id = ?",
                (self._serialize(value), title, conversation_id, owner),
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

    def record_answer_refusal(
        self,
        *,
        question,
        answer_status,
        reason,
        diagnostics=None,
        conversation_id=None,
        user_id=None,
        user_email=None,
        profile=None,
    ):
        """Persist why a turn fell back or abstained (not shown to chat users)."""
        refusal_id = str(uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO answer_refusals (
                    refusal_id, conversation_id, user_id, user_email, question,
                    answer_status, reason, diagnostics_json, profile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    refusal_id,
                    conversation_id,
                    user_id,
                    user_email,
                    str(question or ""),
                    str(answer_status or ""),
                    str(reason or "")[:4000],
                    json.dumps(list(diagnostics or []), ensure_ascii=False),
                    profile,
                ),
            )
        return refusal_id

    def list_answer_refusals(self, *, limit=100, reasons=None, buckets=None):
        from answer_contract import refusal_reason_bucket, refusal_reason_title

        limit = max(1, min(int(limit), 100_000))
        selected_reasons = [str(item) for item in (reasons or []) if str(item)]
        selected_buckets = [str(item) for item in (buckets or []) if str(item)]
        query = """
            SELECT refusal_id, conversation_id, user_id, user_email, question,
                   answer_status, reason, diagnostics_json, profile, created_at
            FROM answer_refusals
        """
        params: list = []
        if selected_reasons:
            placeholders = ", ".join("?" for _ in selected_reasons)
            query += f" WHERE reason IN ({placeholders})"
            params.extend(selected_reasons)
        query += " ORDER BY created_at DESC, rowid DESC"
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            # ponytail: bucket filter in Python; admin log stays small enough
            rows = connection.execute(query, params).fetchall()
        items = []
        for row in rows:
            reason = row["reason"]
            bucket = refusal_reason_bucket(reason)
            if selected_buckets and bucket not in selected_buckets:
                continue
            items.append(
                {
                    "refusal_id": row["refusal_id"],
                    "conversation_id": row["conversation_id"],
                    "user_id": row["user_id"],
                    "user_email": row["user_email"],
                    "question": row["question"],
                    "answer_status": row["answer_status"],
                    "reason": reason,
                    "reason_bucket": bucket,
                    "reason_title": refusal_reason_title(bucket=bucket),
                    "diagnostics": json.loads(row["diagnostics_json"] or "[]"),
                    "profile": row["profile"],
                    "created_at": row["created_at"],
                }
            )
            if len(items) >= limit:
                break
        return items

    def count_answer_refusals(self, *, reasons=None, buckets=None):
        selected_reasons = [str(item) for item in (reasons or []) if str(item)]
        selected_buckets = [str(item) for item in (buckets or []) if str(item)]
        if not selected_buckets:
            with self._connect() as connection:
                if not selected_reasons:
                    row = connection.execute("SELECT COUNT(*) FROM answer_refusals").fetchone()
                else:
                    placeholders = ", ".join("?" for _ in selected_reasons)
                    row = connection.execute(
                        f"SELECT COUNT(*) FROM answer_refusals WHERE reason IN ({placeholders})",
                        selected_reasons,
                    ).fetchone()
            return int(row[0] if row else 0)
        return len(self.list_answer_refusals(limit=100_000, reasons=selected_reasons, buckets=selected_buckets))

    def distinct_answer_refusal_reasons(self, *, limit=200):
        from answer_contract import refusal_reason_bucket, refusal_reason_title

        limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT reason, COUNT(*) AS n
                FROM answer_refusals
                WHERE reason IS NOT NULL AND TRIM(reason) != ''
                GROUP BY reason
                """
            ).fetchall()
        totals: dict[str, int] = {}
        for reason, count in rows:
            bucket = refusal_reason_bucket(reason)
            totals[bucket] = totals.get(bucket, 0) + int(count)
        ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [
            {
                "bucket": bucket,
                "reason": bucket,
                "title": refusal_reason_title(bucket=bucket),
                "count": count,
            }
            for bucket, count in ordered
        ]

    def rename(self, conversation_id, title, *, user_id: str):
        owner = self._require_user_id(user_id)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversation_sessions SET title = ? "
                "WHERE conversation_id = ? AND user_id = ?",
                (str(title).strip()[:120], conversation_id, owner),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown conversation: {conversation_id}")

    def delete(self, conversation_id, *, user_id: str):
        owner = self._require_user_id(user_id)
        with self._connect() as connection:
            owned = connection.execute(
                "SELECT 1 FROM conversation_sessions "
                "WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, owner),
            ).fetchone()
            if owned is None:
                raise KeyError(f"Unknown conversation: {conversation_id}")
            connection.execute(
                "DELETE FROM conversation_turns WHERE conversation_id = ?", (conversation_id,)
            )
            cursor = connection.execute(
                "DELETE FROM conversation_sessions "
                "WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, owner),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"Unknown conversation: {conversation_id}")

    def list_conversations(self, *, user_id: str, limit=100):
        owner = self._require_user_id(user_id)
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
                WHERE s.user_id = ?
                  AND (a.first_question IS NOT NULL OR a.last_question IS NOT NULL)
                ORDER BY COALESCE(a.last_turn_at, s.updated_at) DESC
                LIMIT ?
                """,
                (owner, limit),
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

    def get_turn(self, conversation_id, turn_id, *, user_id: str):
        owner = self._require_user_id(user_id)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            owned = connection.execute(
                "SELECT 1 FROM conversation_sessions "
                "WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, owner),
            ).fetchone()
            if owned is None:
                return None
            row = connection.execute(
                """
                SELECT turn_id, conversation_id, question, answer, answer_status, profile
                FROM conversation_turns
                WHERE conversation_id = ? AND turn_id = ?
                """,
                (conversation_id, turn_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "turn_id": row["turn_id"],
            "conversation_id": row["conversation_id"],
            "question": row["question"],
            "answer": row["answer"],
            "answer_status": row["answer_status"],
            "profile": row["profile"],
        }

    def transcript(self, conversation_id, *, user_id: str):
        owner = self._require_user_id(user_id)
        state = self.load(conversation_id, user_id=owner)
        if state is None:
            return None
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            title_row = connection.execute(
                "SELECT title FROM conversation_sessions "
                "WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, owner),
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

    @staticmethod
    def _require_user_id(user_id: str) -> str:
        owner = str(user_id or "").strip()
        if not owner:
            raise ValueError("user_id is required")
        return owner

    @staticmethod
    def _ensure_session_columns(connection):
        # Older DBs were created before title/user_id existed; CREATE TABLE IF NOT EXISTS
        # does not add new columns to an already-present table.
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(conversation_sessions)").fetchall()
        }
        if "title" not in columns:
            connection.execute(
                "ALTER TABLE conversation_sessions ADD COLUMN title TEXT"
            )
        if "user_id" not in columns:
            connection.execute(
                "ALTER TABLE conversation_sessions ADD COLUMN user_id TEXT"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation_sessions_user "
            "ON conversation_sessions(user_id)"
        )

    def _bounded_state(self, state):
        value = dict(state)
        value["turns"] = list(value.get("turns", []))[-self.max_turns :]
        return value

    @staticmethod
    def _serialize(state):
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
