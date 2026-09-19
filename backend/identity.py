"""User accounts, password hashing, and server-side sessions."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import Cookie, HTTPException, Request, Response

Role = Literal["user", "admin"]
Status = Literal["pending", "approved", "rejected"]

SESSION_COOKIE = "bct_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
_HASH_PREFIX = "scrypt"
# Avatar: empty = letter mark; otherwise a JPEG/PNG data URL.
MAX_AVATAR_DATA_URL_CHARS = 180_000
_LEGACY_AVATAR_ICONS = frozenset({"initial", "scale", "book", "landmark", "compass", "shield"})


def normalize_avatar(value: str | None) -> str:
    """Accept '', letter fallback, or data:image/(jpeg|png);base64,..."""
    if value is None:
        return ""
    cleaned = value.strip()
    if not cleaned or cleaned.casefold() in _LEGACY_AVATAR_ICONS:
        return ""
    lower = cleaned.casefold()
    if not (
        lower.startswith("data:image/jpeg;base64,")
        or lower.startswith("data:image/png;base64,")
    ):
        raise ValueError("Avatar must be a JPEG or PNG image.")
    if len(cleaned) > MAX_AVATAR_DATA_URL_CHARS:
        raise ValueError("Avatar image is too large (max about 128 KB).")
    try:
        import base64

        payload = cleaned.split(",", 1)[1]
        raw = base64.b64decode(payload, validate=True)
    except Exception as error:
        raise ValueError("Avatar image data is invalid.") from error
    if len(raw) < 24:
        raise ValueError("Avatar image data is invalid.")
    if lower.startswith("data:image/jpeg"):
        if raw[:3] != b"\xff\xd8\xff":
            raise ValueError("Avatar must be a JPEG or PNG image.")
    elif raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Avatar must be a JPEG or PNG image.")
    return cleaned


def default_auth_database_path() -> Path:
    local_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_data) if local_data else Path.home() / ".local" / "share"
    return root / "BCT-Regulatory-Search" / "auth.sqlite3"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    # ponytail: scrypt n=2**12 is enough for this local admin tool; raise if facing online attacks
    n = 2**12
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=8, p=1, dklen=32)
    return f"{_HASH_PREFIX}${n}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        parts = encoded.split("$")
        if len(parts) == 3:
            prefix, salt_hex, digest_hex = parts
            n = 2**14
        elif len(parts) == 4:
            prefix, n_raw, salt_hex, digest_hex = parts
            n = int(n_raw)
        else:
            return False
    except ValueError:
        return False
    if prefix != _HASH_PREFIX:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=8, p=1, dklen=32)
    return hmac.compare_digest(actual, expected)


@dataclass(frozen=True)
class UserRecord:
    id: str
    email: str
    role: Role
    status: Status
    created_at: float
    updated_at: float
    display_name: str = ""
    avatar_icon: str = ""
    token_limit: int = 0
    tokens_used: int = 0
    tokens_llm: int = 0
    tokens_embed: int = 0
    tokens_rerank: int = 0

    def public_dict(self) -> dict:
        limit = max(0, int(self.token_limit))
        llm = max(0, int(self.tokens_llm))
        embed = max(0, int(self.tokens_embed))
        rerank = max(0, int(self.tokens_rerank))
        used = max(0, int(self.tokens_used), llm + embed + rerank)
        remaining = None if limit <= 0 else max(0, limit - used)
        usd_per_million = float(os.environ.get("BCT_TOKEN_USD_PER_MILLION", "0.5"))
        try:
            avatar = normalize_avatar(self.avatar_icon)
        except ValueError:
            avatar = ""
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "avatar_icon": avatar,
            "role": self.role,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "token_limit": limit,
            "tokens_used": used,
            "tokens_llm": llm,
            "tokens_embed": embed,
            "tokens_rerank": rerank,
            "tokens_remaining": remaining,
            "estimated_spend_usd": round(used / 1_000_000 * usd_per_million, 4),
        }


class AuthStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'admin')),
                status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'rejected')),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
            """
        )
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "token_limit" not in columns:
            default_limit = int(os.environ.get("BCT_DEFAULT_TOKEN_LIMIT", "500000"))
            self._conn.execute(
                f"ALTER TABLE users ADD COLUMN token_limit INTEGER NOT NULL DEFAULT {default_limit}"
            )
        if "tokens_used" not in columns:
            self._conn.execute(
                "ALTER TABLE users ADD COLUMN tokens_used INTEGER NOT NULL DEFAULT 0"
            )
            columns.add("tokens_used")
        for column in ("tokens_llm", "tokens_embed", "tokens_rerank"):
            if column not in columns:
                self._conn.execute(
                    f"ALTER TABLE users ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                )
                columns.add(column)
        if "display_name" not in columns:
            self._conn.execute(
                "ALTER TABLE users ADD COLUMN display_name TEXT NOT NULL DEFAULT ''"
            )
            columns.add("display_name")
        if "avatar_icon" not in columns:
            self._conn.execute(
                "ALTER TABLE users ADD COLUMN avatar_icon TEXT NOT NULL DEFAULT ''"
            )
            columns.add("avatar_icon")
        # One-time: treat legacy flat usage as Groq until the next reset.
        self._conn.execute(
            """
            UPDATE users
            SET tokens_llm = tokens_used
            WHERE tokens_used > 0
              AND tokens_llm = 0
              AND tokens_embed = 0
              AND tokens_rerank = 0
            """
        )
        self._conn.commit()

    def bootstrap_admin(self) -> UserRecord | None:
        email = (os.environ.get("BCT_BOOTSTRAP_ADMIN_EMAIL") or "").strip().casefold()
        password = (os.environ.get("BCT_BOOTSTRAP_ADMIN_PASSWORD") or "").strip()
        if not email or not password:
            return None
        existing = self.get_by_email(email)
        if existing is not None:
            return existing
        return self.create_user(email=email, password=password, role="admin", status="approved")

    def create_user(
        self,
        *,
        email: str,
        password: str,
        role: Role = "user",
        status: Status = "pending",
    ) -> UserRecord:
        cleaned = email.strip().casefold()
        if "@" not in cleaned or len(cleaned) > 254:
            raise ValueError("Invalid email address.")
        if len(password) < 8 or len(password) > 128:
            raise ValueError("Password must be between 8 and 128 characters.")
        now = time.time()
        user_id = secrets.token_urlsafe(16)
        default_limit = int(os.environ.get("BCT_DEFAULT_TOKEN_LIMIT", "500000"))
        try:
            self._conn.execute(
                """
                INSERT INTO users (
                    id, email, password_hash, role, status,
                    created_at, updated_at, token_limit, tokens_used,
                    tokens_llm, tokens_embed, tokens_rerank
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0)
                """,
                (
                    user_id,
                    cleaned,
                    hash_password(password),
                    role,
                    status,
                    now,
                    now,
                    0 if role == "admin" else default_limit,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as error:
            raise ValueError("An account with this email already exists.") from error
        user = self.get_by_id(user_id)
        assert user is not None
        return user

    def authenticate(self, email: str, password: str) -> UserRecord | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().casefold(),),
        ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return self._row_to_user(row)

    def get_by_id(self, user_id: str) -> UserRecord | None:
        row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return None if row is None else self._row_to_user(row)

    def get_by_email(self, email: str) -> UserRecord | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().casefold(),),
        ).fetchone()
        return None if row is None else self._row_to_user(row)

    def list_users(self) -> list[UserRecord]:
        rows = self._conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_user(row) for row in rows]

    def set_status(self, user_id: str, status: Status) -> UserRecord:
        now = time.time()
        cur = self._conn.execute(
            "UPDATE users SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, user_id),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(user_id)
        if status != "approved":
            self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            self._conn.commit()
        user = self.get_by_id(user_id)
        assert user is not None
        return user

    def promote_to_admin(self, user_id: str) -> UserRecord:
        """Irreversible privilege grant from the product UI (no demote path)."""
        user = self.get_by_id(user_id)
        if user is None:
            raise KeyError(user_id)
        if user.role == "admin":
            return user
        if user.status != "approved":
            raise ValueError("Only approved users can be promoted to administrator.")
        now = time.time()
        self._conn.execute(
            "UPDATE users SET role = 'admin', updated_at = ? WHERE id = ?",
            (now, user_id),
        )
        self._conn.commit()
        promoted = self.get_by_id(user_id)
        assert promoted is not None
        return promoted

    def set_token_limit(self, user_id: str, token_limit: int) -> UserRecord:
        if token_limit < 0:
            raise ValueError("Token limit must be zero or positive.")
        now = time.time()
        cur = self._conn.execute(
            "UPDATE users SET token_limit = ?, updated_at = ? WHERE id = ?",
            (int(token_limit), now, user_id),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(user_id)
        user = self.get_by_id(user_id)
        assert user is not None
        return user

    def reset_token_usage(self, user_id: str) -> UserRecord:
        now = time.time()
        cur = self._conn.execute(
            """
            UPDATE users
            SET tokens_used = 0,
                tokens_llm = 0,
                tokens_embed = 0,
                tokens_rerank = 0,
                updated_at = ?
            WHERE id = ?
            """,
            (now, user_id),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(user_id)
        user = self.get_by_id(user_id)
        assert user is not None
        return user

    def consume_tokens(self, user_id: str, tokens: int) -> UserRecord:
        # Legacy helper: treat unspecified usage as LLM (Groq) tokens.
        return self.consume_cloud_usage(user_id, llm=tokens)

    def consume_cloud_usage(
        self,
        user_id: str,
        *,
        llm: int = 0,
        embed: int = 0,
        rerank: int = 0,
    ) -> UserRecord:
        llm_n = max(0, int(llm))
        embed_n = max(0, int(embed))
        rerank_n = max(0, int(rerank))
        total = llm_n + embed_n + rerank_n
        if total == 0:
            user = self.get_by_id(user_id)
            if user is None:
                raise KeyError(user_id)
            return user
        now = time.time()
        cur = self._conn.execute(
            """
            UPDATE users
            SET tokens_llm = tokens_llm + ?,
                tokens_embed = tokens_embed + ?,
                tokens_rerank = tokens_rerank + ?,
                tokens_used = tokens_used + ?,
                updated_at = ?
            WHERE id = ?
            """,
            (llm_n, embed_n, rerank_n, total, now, user_id),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(user_id)
        user = self.get_by_id(user_id)
        assert user is not None
        return user

    def tokens_remaining(self, user: UserRecord) -> int | None:
        if user.token_limit <= 0:
            return None
        return max(0, user.token_limit - user.tokens_used)

    def ensure_token_budget(self, user: UserRecord) -> None:
        remaining = self.tokens_remaining(user)
        if remaining is not None and remaining <= 0:
            raise PermissionError("Token quota exhausted.")

    def delete_user(self, user_id: str) -> None:
        target = self.get_by_id(user_id)
        if target is None:
            raise KeyError(user_id)
        if target.role == "admin":
            raise PermissionError("Administrator accounts cannot be deleted from the UI.")
        cur = self._conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(user_id)

    def update_profile(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        avatar_icon: str | None = None,
    ) -> UserRecord:
        user = self.get_by_id(user_id)
        if user is None:
            raise KeyError(user_id)
        name = user.display_name if display_name is None else " ".join(display_name.split())
        if len(name) > 80:
            raise ValueError("Display name must be at most 80 characters.")
        icon = user.avatar_icon if avatar_icon is None else normalize_avatar(avatar_icon)
        now = time.time()
        self._conn.execute(
            """
            UPDATE users
            SET display_name = ?, avatar_icon = ?, updated_at = ?
            WHERE id = ?
            """,
            (name, icon, now, user_id),
        )
        self._conn.commit()
        updated = self.get_by_id(user_id)
        assert updated is not None
        return updated

    def change_password(self, user_id: str, *, current_password: str, new_password: str) -> None:
        row = self._conn.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            raise KeyError(user_id)
        if not verify_password(current_password, row["password_hash"]):
            raise PermissionError("Current password is incorrect.")
        if len(new_password) < 8 or len(new_password) > 128:
            raise ValueError("Password must be between 8 and 128 characters.")
        if current_password == new_password:
            raise ValueError("New password must be different from the current password.")
        now = time.time()
        self._conn.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
            (hash_password(new_password), now, user_id),
        )
        self._conn.commit()

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now + SESSION_TTL_SECONDS, now),
        )
        self._conn.commit()
        return token

    def user_for_session(self, token: str | None) -> UserRecord | None:
        if not token:
            return None
        now = time.time()
        row = self._conn.execute(
            """
            SELECT u.* FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token = ? AND s.expires_at > ?
            """,
            (token, now),
        ).fetchone()
        if row is None:
            self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            self._conn.commit()
            return None
        return self._row_to_user(row)

    def delete_session(self, token: str | None) -> None:
        if not token:
            return
        self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        self._conn.commit()

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> UserRecord:
        keys = set(row.keys())
        raw_avatar = str(row["avatar_icon"]) if "avatar_icon" in keys else ""
        try:
            avatar = normalize_avatar(raw_avatar)
        except ValueError:
            avatar = ""
        return UserRecord(
            id=row["id"],
            email=row["email"],
            role=row["role"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            display_name=str(row["display_name"]) if "display_name" in keys else "",
            avatar_icon=avatar,
            token_limit=int(row["token_limit"]) if "token_limit" in keys else 0,
            tokens_used=int(row["tokens_used"]) if "tokens_used" in keys else 0,
            tokens_llm=int(row["tokens_llm"]) if "tokens_llm" in keys else 0,
            tokens_embed=int(row["tokens_embed"]) if "tokens_embed" in keys else 0,
            tokens_rerank=int(row["tokens_rerank"]) if "tokens_rerank" in keys else 0,
        )


def open_auth_store() -> AuthStore:
    configured = os.environ.get("BCT_AUTH_DB")
    return AuthStore(configured or default_auth_database_path())


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=os.environ.get("BCT_COOKIE_SECURE", "0") == "1",
        max_age=SESSION_TTL_SECONDS,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


def auth_store_from_request(request: Request) -> AuthStore:
    store = getattr(request.app.state, "auth_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Authentication is unavailable.")
    return store


def current_user(
    request: Request,
    bct_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> UserRecord | None:
    return auth_store_from_request(request).user_for_session(bct_session)


def require_user(
    request: Request,
    bct_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> UserRecord:
    user = current_user(request, bct_session)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def require_approved_user(
    request: Request,
    bct_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> UserRecord:
    user = require_user(request, bct_session)
    if user.status != "approved":
        raise HTTPException(status_code=403, detail="Account is not approved.")
    return user


def require_admin(
    request: Request,
    bct_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> UserRecord:
    user = require_approved_user(request, bct_session)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required.")
    return user
