"""Run the BCT API on loopback for the React frontend's /api proxy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn

from ingestion.index import configure_runtime_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--data-dir", type=Path, default=Path(".demo-data"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--documents", type=Path, help="Root directory containing the original source PDFs (searched recursively)")
    parser.add_argument("--enable-ingestion", action="store_true", help="Enable the administrator PDF ingestion endpoints")
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)
    asset_root = args.assets.resolve(strict=True)
    active = configure_runtime_assets(asset_root, validate=True)
    data = args.data_dir.resolve()
    data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BCT_DEFAULT_PROFILE", "cloud")
    os.environ["BCT_ENABLE_INGESTION"] = "1" if args.enable_ingestion else os.environ.get("BCT_ENABLE_INGESTION", "0")
    os.environ.setdefault("BCT_CONVERSATION_DB", str(data / "conversations.sqlite3"))
    os.environ.setdefault("BCT_INGESTION_DB", str(data / "ingestion.sqlite3"))
    os.environ.setdefault("BCT_AUTH_DB", str(data / "auth.sqlite3"))
    os.environ.setdefault("BCT_SETTINGS_DB", str(data / "app_settings.sqlite3"))
    os.environ.setdefault("BCT_INGESTION_DATA_DIR", str(data / "ingestion"))
    os.environ.setdefault("BCT_VOYAGE_RUNTIME_CACHE", str(data / "voyage-cache"))
    os.environ.setdefault("BCT_GOOGLE_RUNTIME_CACHE", str(data / "google-cache"))
    os.environ.setdefault("BCT_CLOUD_RETRIEVAL_PROVIDER", "voyage")
    if args.documents is not None:
        os.environ["BCT_DOCUMENTS_DIR"] = str(args.documents.resolve(strict=True))
    print(f"BCT runtime assets: {active}")
    uvicorn.run("app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
