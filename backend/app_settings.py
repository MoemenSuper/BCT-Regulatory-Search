"""Central application settings: active runtime profile and provider secrets."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

from runtime_profiles import RuntimeProfile, parse_profile, profile_options

# Keys administrators may set. Values are applied into os.environ for the process.
MANAGED_SECRET_KEYS = (
    "GROQ_API_KEY",
    "VOYAGE_API_KEY",
    "GEMINI_API_KEY",
    "BCT_GROQ_MODEL",
    "BCT_LOCAL_LLM_URL",
    "BCT_LOCAL_LLM_MODEL",
    "BCT_NEO4J_PASSWORD",
)

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
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
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
        self.apply_to_environment()

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
        return parse_profile(os.environ.get("BCT_DEFAULT_PROFILE", RuntimeProfile.CLOUD.value))

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
            secrets.append(
                {
                    "key": key,
                    "configured": bool(effective),
                    "masked": mask_secret(effective),
                    "source": "store" if stored is not None else ("environment" if env_value else "unset"),
                }
            )
        return {
            "active_profile": self.active_profile().value,
            "profiles": profile_options(),
            "secrets": secrets,
        }

    def update_secrets(self, updates: dict[str, str | None]) -> dict:
        for key, value in updates.items():
            if key not in MANAGED_SECRET_KEYS:
                raise ValueError(f"Unsupported configuration key: {key}")
            if value is None or value.strip() == "":
                self.delete(key)
                # Leave existing process env alone if clearing store override.
                continue
            cleaned = value.strip()
            self.set(key, cleaned)
            os.environ[key] = cleaned
        self.apply_to_environment()
        return self.public_configuration()


def open_app_settings() -> AppSettingsStore:
    configured = os.environ.get("BCT_SETTINGS_DB")
    return AppSettingsStore(configured or default_settings_database_path())
