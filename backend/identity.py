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

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
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
        self._conn.commit()

    def bootstrap_admin(self) -> UserRecord | None:
        email = (os.environ.get("BCT_BOOTSTRAP_ADMIN_EMAIL") or "").strip().casefold()
        password = os.environ.get("BCT_BOOTSTRAP_ADMIN_PASSWORD") or ""
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
        try:
            self._conn.execute(
                """
                INSERT INTO users (id, email, password_hash, role, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, cleaned, hash_password(password), role, status, now, now),
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
        return UserRecord(
            id=row["id"],
            email=row["email"],
            role=row["role"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
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
