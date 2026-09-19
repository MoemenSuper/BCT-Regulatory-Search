"""Serve the API and built React UI (Docker / one-port pilot)."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ingestion.index import configure_runtime_assets, resolve_active_assets


def ensure_empty_assets(asset_root: Path) -> None:
    """Allow first boot before any PDF is ingested."""
    asset_root.mkdir(parents=True, exist_ok=True)
    active = resolve_active_assets(asset_root)
    active.mkdir(parents=True, exist_ok=True)
    native = active / "native.jsonl"
    ocr = active / "arabic_ocr_secondary.jsonl"
    snapshot = active / "snapshot.json"
    if not native.exists():
        native.write_text("", encoding="utf-8")
    if not ocr.exists():
        ocr.write_text("", encoding="utf-8")
    if not snapshot.exists():
        snapshot.write_text(
            json.dumps({"created_by": "docker_serve_empty_bootstrap", "documents": 0}, sort_keys=True),
            encoding="utf-8",
        )


def build_gateway(static_dir: Path) -> FastAPI:
    from app import app as api_app

    # Mounted sub-apps do not run their own lifespan unless the parent wires it.
    # Without this, bootstrap_admin never runs and login always fails.
    @asynccontextmanager
    async def lifespan(_gateway: FastAPI):
        async with api_app.router.lifespan_context(api_app):
            yield

    gateway = FastAPI(title="BCT Regulatory Search", lifespan=lifespan)
    gateway.mount("/api", api_app)
    gateway.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")
    return gateway


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)

    asset_root = Path(os.environ.get("BCT_ASSETS_DIR", "/data/assets")).resolve()
    data = Path(os.environ.get("BCT_DATA_DIR", "/data/state")).resolve()
    data.mkdir(parents=True, exist_ok=True)
    documents = Path(os.environ.get("BCT_DOCUMENTS_DIR", "/data/documents")).resolve()
    documents.mkdir(parents=True, exist_ok=True)
    os.environ["BCT_DOCUMENTS_DIR"] = str(documents)

    ensure_empty_assets(asset_root)
    active = configure_runtime_assets(asset_root, validate=True)

    static_dir = Path(os.environ.get("BCT_STATIC_DIR", "/app/static")).resolve()
    if not static_dir.is_dir():
        raise SystemExit(f"UI build missing at {static_dir}")

    os.environ.setdefault("BCT_DEFAULT_PROFILE", "cloud")
    os.environ.setdefault("BCT_ENABLE_GRAPH", "1")
    os.environ.setdefault("BCT_INGEST_GRAPH", "1")
    os.environ.setdefault("BCT_INGEST_LOCAL_INDEX", "0")
    os.environ.setdefault("BCT_ENABLE_INGESTION", "1")
    os.environ.setdefault("BCT_NEO4J_URI", "bolt://neo4j:7687")
    os.environ.setdefault("BCT_NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("BCT_CONVERSATION_DB", str(data / "conversations.sqlite3"))
    os.environ.setdefault("BCT_INGESTION_DB", str(data / "ingestion.sqlite3"))
    os.environ.setdefault("BCT_AUTH_DB", str(data / "auth.sqlite3"))
    os.environ.setdefault("BCT_SETTINGS_DB", str(data / "app_settings.sqlite3"))
    os.environ.setdefault("BCT_INGESTION_DATA_DIR", str(data / "ingestion"))
    os.environ.setdefault("BCT_VOYAGE_RUNTIME_CACHE", str(data / "voyage-cache"))
    os.environ.setdefault("BCT_GOOGLE_RUNTIME_CACHE", str(data / "google-cache"))
    os.environ.setdefault("BCT_GEMINI_CACHE", str(data / "gemini-cache"))
    os.environ.setdefault("BCT_CLOUD_RETRIEVAL_PROVIDER", "voyage")

    host = os.environ.get("BCT_BIND_HOST", "0.0.0.0")
    port = int(os.environ.get("BCT_BIND_PORT", "8000"))
    print(f"BCT runtime assets: {active}")
    print(f"BCT documents: {documents}")
    print(f"BCT UI: {static_dir}")
    uvicorn.run(build_gateway(static_dir), host=host, port=port)


if __name__ == "__main__":
    main()
