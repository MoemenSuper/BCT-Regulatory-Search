"""Serve the API and built React UI (Docker / one-port pilot)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ingestion.index import configure_runtime_assets, resolve_active_assets
from server_logging import configure_logging


logger = logging.getLogger(__name__)


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


# Which baked index the assets volume was filled from, written beside ACTIVE.json.
SEEDED_FROM = "SEEDED_FROM.txt"


def _baked_version(baked: Path) -> str | None:
    try:
        return json.loads((baked / "ACTIVE.json").read_text(encoding="utf-8"))["active_version"]
    except (OSError, ValueError, KeyError):
        return None


def ensure_runtime_assets(asset_root: Path) -> bool:
    """Fill the assets volume from the index baked into the image; empty stubs when there is none.

    Docker Compose mounts a named volume over /data/assets, and a volume outlives the image:
    after a rebuild it still holds the index of the first image that filled it (or the empty
    stubs of an image that had none: every question then finds nothing). So the volume records
    which baked index it came from, and an image shipping another one replaces it. Returns True
    when an existing index was replaced: the caller queues the uploaded PDFs again, so they are
    indexed on top of the shipped one.
    """
    asset_root.mkdir(parents=True, exist_ok=True)
    baked = Path(os.environ.get("BCT_BAKED_ASSETS_DIR", "/opt/bct/baked-assets"))
    shipped = _baked_version(baked)
    marker = asset_root / SEEDED_FROM
    replaced = False
    if (asset_root / "ACTIVE.json").exists() or (asset_root / "native.jsonl").exists():
        seeded = marker.read_text(encoding="utf-8").strip() if marker.is_file() else "unknown"
        if shipped is None or seeded == shipped:
            return False
        # Set the old index aside, never delete it.
        old = asset_root / f"replaced-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        old.mkdir()
        for item in list(asset_root.iterdir()):
            if item != old and not item.name.startswith("replaced-"):
                shutil.move(str(item), old / item.name)
        logger.warning(f"Replacing the index from {seeded} with the one this image ships; old one kept in {old}")
        replaced = True

    if shipped is not None:
        logger.info(f"Seeding runtime assets from baked corpus: {baked} ({shipped})")
        for item in baked.iterdir():
            dest = asset_root / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        marker.write_text(shipped, encoding="utf-8")
        return replaced

    ensure_empty_assets(asset_root)
    return False


def _requeue_uploads() -> None:
    """After the index was replaced: index the PDFs uploaded into the old one again (background worker)."""
    from ingestion.registry import IngestionRegistry

    registry = IngestionRegistry(os.environ["BCT_INGESTION_DB"])
    try:
        count = registry.requeue_indexed()
    finally:
        registry.close()
    if count:
        logger.warning(f"{count} uploaded PDF(s) queued again: they are indexed on top of the shipped index.")


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
    print(f"BCT log file: {configure_logging(data)}", flush=True)
    documents = Path(os.environ.get("BCT_DOCUMENTS_DIR", "/data/documents")).resolve()
    documents.mkdir(parents=True, exist_ok=True)
    os.environ["BCT_DOCUMENTS_DIR"] = str(documents)

    os.environ.setdefault("BCT_INGESTION_DB", str(data / "ingestion.sqlite3"))
    if ensure_runtime_assets(asset_root):
        _requeue_uploads()
    active = configure_runtime_assets(asset_root, validate=True)

    static_dir = Path(os.environ.get("BCT_STATIC_DIR", "/app/static")).resolve()
    if not static_dir.is_dir():
        raise SystemExit(f"UI build missing at {static_dir}")

    os.environ.setdefault("BCT_DEFAULT_PROFILE", "local_hybrid")
    os.environ.setdefault("BCT_INGEST_LOCAL_INDEX", "1")
    os.environ.setdefault("BCT_ENABLE_INGESTION", "1")
    os.environ.setdefault("BCT_WARM_START", "1")
    os.environ.setdefault("BCT_CONVERSATION_DB", str(data / "conversations.sqlite3"))
    os.environ.setdefault("BCT_AUTH_DB", str(data / "auth.sqlite3"))
    os.environ.setdefault("BCT_SETTINGS_DB", str(data / "app_settings.sqlite3"))
    os.environ.setdefault("BCT_INGESTION_DATA_DIR", str(data / "ingestion"))
    os.environ.setdefault("BCT_VOYAGE_RUNTIME_CACHE", str(data / "voyage-cache"))
    os.environ.setdefault("BCT_GEMINI_CACHE", str(data / "gemini-cache"))

    host = os.environ.get("BCT_BIND_HOST", "0.0.0.0")
    port = int(os.environ.get("BCT_BIND_PORT", "8000"))
    logger.info(f"BCT runtime assets: {active}")
    logger.info(f"BCT documents: {documents}")
    logger.info(f"BCT UI: {static_dir}")
    uvicorn.run(build_gateway(static_dir), host=host, port=port, log_config=None)  # uvicorn logs go to bct.log too


if __name__ == "__main__":
    main()
