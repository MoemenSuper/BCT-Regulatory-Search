"""Central application settings: active runtime profile and provider secrets."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

from extras.sqlite_local import thread_connection
from extras.runtime_profiles import RuntimeProfile, parse_profile, profile_options

# Keys administrators may set, in the order the Configuration page shows them. Values are
# applied into os.environ for the process. The answer model: BCT_ANSWER_PROVIDER picks one
# (empty = the first provider with a key), each provider has its key and an optional model.
MANAGED_SECRET_KEYS = (
    "BCT_ANSWER_PROVIDER",
    "BCT_SPEED_MODE",
    "OPENAI_API_KEY",
    "BCT_OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "BCT_ANTHROPIC_MODEL",
    "GEMINI_API_KEY",
    "BCT_GEMINI_ANSWER_MODEL",
    "GROQ_API_KEY",
    "BCT_GROQ_MODEL",
    "VOYAGE_API_KEY",
    "BCT_LOCAL_LLM_URL",
    "BCT_LOCAL_LLM_MODEL",
)

# Settings that are not secret (a provider or model name, a local address) are shown in full.
PLAIN_SETTING_KEYS = (
    "BCT_ANSWER_PROVIDER", "BCT_SPEED_MODE", "BCT_OPENAI_MODEL", "BCT_ANTHROPIC_MODEL", "BCT_GEMINI_ANSWER_MODEL",
    "BCT_GROQ_MODEL", "BCT_LOCAL_LLM_URL", "BCT_LOCAL_LLM_MODEL",
)
# Settings with a fixed list of values (a dropdown in Admin > Configuration).
CHOICES = {
    "BCT_ANSWER_PROVIDER": ("openai", "anthropic", "gemini", "groq"),
    "BCT_SPEED_MODE": ("auto", "fast"),
}

ACTIVE_PROFILE_KEY = "active_profile"


def default_settings_database_path() -> Path:
    local_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_data) if local_data else Path.home() / ".local" / "share"
    return root / "BCT-Regulatory-Search" / "app_settings.sqlite3"


def mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}…{value[-2:]}"


class AppSettingsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()
        # What the server started with (.env or the real environment), before saved
        # values override it. Removing a saved value falls back to this.
        self._started_with = {key: os.environ.get(key) for key in MANAGED_SECRET_KEYS}
        self.apply_to_environment()

    @property
    def _conn(self) -> sqlite3.Connection:
        return thread_connection(self._local, self.path)

    def close(self) -> None:
        self._conn.close()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else row[0]

    def set(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, time.time()),
        )
        self._conn.commit()

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        self._conn.commit()

    def active_profile(self) -> RuntimeProfile:
        stored = self.get(ACTIVE_PROFILE_KEY)
        if stored:
            return parse_profile(stored)
        return parse_profile(os.environ.get("BCT_DEFAULT_PROFILE", RuntimeProfile.LOCAL_HYBRID.value))

    def set_active_profile(self, profile: str | RuntimeProfile) -> RuntimeProfile:
        resolved = parse_profile(profile)
        self.set(ACTIVE_PROFILE_KEY, resolved.value)
        os.environ["BCT_DEFAULT_PROFILE"] = resolved.value
        return resolved

    def apply_to_environment(self) -> None:
        profile = self.get(ACTIVE_PROFILE_KEY)
        if profile:
            os.environ["BCT_DEFAULT_PROFILE"] = profile
        for key in MANAGED_SECRET_KEYS:
            value = self.get(key)
            if value is not None:
                os.environ[key] = value

    def public_configuration(self) -> dict:
        secrets = []
        for key in MANAGED_SECRET_KEYS:
            stored = self.get(key)
            env_value = os.environ.get(key)
            effective = stored if stored is not None else env_value
            secret = key not in PLAIN_SETTING_KEYS
            secrets.append(
                {
                    "key": key,
                    "secret": secret,
                    "configured": bool(effective),
                    "masked": mask_secret(effective) if secret else None,
                    "value": None if secret else effective,
                    "source": "store" if stored is not None else ("environment" if env_value else "unset"),
                    "choices": list(CHOICES.get(key, ())) or None,
                }
            )
        from rag.llm import answer_provider

        return {
            "active_profile": self.active_profile().value,
            "profiles": profile_options(),
            "secrets": secrets,
            # What the app answers with now (the choice, or the first provider with a key).
            "answer_provider": answer_provider(),
        }

    def update_secrets(self, updates: dict[str, str | None]) -> dict:
        for key, value in updates.items():
            if key not in MANAGED_SECRET_KEYS:
                raise ValueError(f"Unsupported configuration key: {key}")
            if value is None or value.strip() == "":
                self.delete(key)
                # Fall back to the value the server started with, or to nothing.
                original = self._started_with.get(key)
                if original:
                    os.environ[key] = original
                else:
                    os.environ.pop(key, None)
                continue
            cleaned = value.strip()
            if key in CHOICES:
                cleaned = cleaned.casefold()
                if cleaned not in CHOICES[key]:
                    raise ValueError(f"{key} must be one of: {', '.join(CHOICES[key])}")
            self.set(key, cleaned)
            os.environ[key] = cleaned
        self.apply_to_environment()
        return self.public_configuration()


def open_app_settings() -> AppSettingsStore:
    configured = os.environ.get("BCT_SETTINGS_DB")
    return AppSettingsStore(configured or default_settings_database_path())
