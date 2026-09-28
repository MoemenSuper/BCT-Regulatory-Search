"""Bake the live corpus into ../baked-runtime-assets, the corpus the Docker image ships.

Copies only what serving needs from a running asset root: the active version (chunks,
supersession edges, snapshot) and its local search collections. Older versions, caches and
ingestion state stay behind. An existing bake is renamed, never deleted.

  python bake_assets.py --source <live asset root> [--dest ../baked-runtime-assets]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ingestion.index import _add_rows, _read_collection, _resolve_local_chroma_db, resolve_active_assets


def bake(source: Path, dest: Path) -> Path:
    import chromadb

    active = resolve_active_assets(source)
    snapshot = json.loads((active / "snapshot.json").read_text(encoding="utf-8"))
    if dest.exists():
        dest.rename(dest.with_name(f"{dest.name}-previous-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"))
    shutil.copytree(active, dest / "versions" / active.name)

    source_db = _resolve_local_chroma_db(snapshot.get("local_chroma_db"), asset_root=source, active=active)
    reader = chromadb.PersistentClient(path=str(source_db))
    writer = chromadb.PersistentClient(path=str(dest / "local_chroma"))
    for key in ("local_collection", "local_visual_collection"):
        if snapshot.get(key):
            rows = _read_collection(reader.get_collection(snapshot[key]))
            _add_rows(writer.create_collection(snapshot[key]), rows)
            print(f"{snapshot[key]}: {len(rows)} chunks")

    snapshot["local_chroma_db"] = "local_chroma"  # relative: resolved against the asset root it is seeded into
    snapshot_path = dest / "versions" / active.name / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    (dest / "ACTIVE.json").write_text(json.dumps({
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "active_version": f"versions/{active.name}",
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Live asset root (contains ACTIVE.json)")
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "baked-runtime-assets")
    args = parser.parse_args()
    print("baked:", bake(args.source.resolve(strict=True), args.dest.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
