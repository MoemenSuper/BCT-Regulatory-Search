"""Rebuild supersession_edges.jsonl from a PDF corpus (no full ingest)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from ingestion.index import configure_runtime_assets, resolve_active_assets
from jsonl_supersession import (
    clear_supersession_cache,
    load_edges,
    maybe_wrap_backend,
    rebuild_supersession_edges_from_documents,
    resolve_edges_path,
)


def _default_documents() -> Path:
    env = (os.environ.get("BCT_DOCUMENTS_DIR") or "").strip()
    if env:
        return Path(env)
    # backend/rebuild_supersession.py → repo documents/
    return Path(__file__).resolve().parents[1] / "documents"


def _prepare_asset_root(asset_root: Path) -> Path:
    """Ensure ACTIVE.json + empty native stubs so configure_runtime_assets works."""
    asset_root.mkdir(parents=True, exist_ok=True)
    versions = asset_root / "versions"
    version = versions / "supersession-rebuild"
    version.mkdir(parents=True, exist_ok=True)
    for name in ("native.jsonl", "arabic_ocr_secondary.jsonl"):
        path = version / name
        if not path.exists():
            path.write_text("", encoding="utf-8")
    snapshot = version / "snapshot.json"
    if not snapshot.exists():
        snapshot.write_text(
            json.dumps(
                {"created_by": "rebuild_supersession", "documents": 0},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    pointer = asset_root / "ACTIVE.json"
    pointer.write_text(
        json.dumps(
            {
                "active_version": "versions/supersession-rebuild",
                "activated_at": "rebuild_supersession",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return resolve_active_assets(asset_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents",
        type=Path,
        default=None,
        help="PDF corpus root (default: BCT_DOCUMENTS_DIR or ../documents)",
    )
    parser.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="Runtime asset root (receives ACTIVE.json + supersession_edges.jsonl)",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)

    documents = (args.documents or _default_documents()).resolve()
    asset_root = args.assets.resolve()
    active = _prepare_asset_root(asset_root)
    output = active / "supersession_edges.jsonl"

    report = rebuild_supersession_edges_from_documents(documents, output)
    clear_supersession_cache()
    configure_runtime_assets(asset_root, validate=True)

    resolved = resolve_edges_path(active)
    edges = load_edges(resolved) if resolved else []
    # Prove maybe_wrap_backend sees the regenerated file (dummy inner backend).
    class _Dummy:
        def retrieve(self, query):
            return []

    wrapped = maybe_wrap_backend(_Dummy(), active)
    from jsonl_supersession import SupersessionPinBackend

    verification = {
        "rebuild": report,
        "resolved_edges_path": str(resolved) if resolved else None,
        "resolved_matches_output": resolved is not None
        and Path(resolved).resolve() == output.resolve(),
        "edge_count_loaded": len(edges),
        "runtime_wrap": type(wrapped).__name__,
        "pin_enabled": isinstance(wrapped, SupersessionPinBackend),
        "false_2026_04_amends_2025_13": any(
            e.source_instrument == "cir:2026:4"
            and e.target_instrument == "cir:2025:13"
            for e in edges
        ),
        "sample_true_edges": [
            {
                "action": e.action,
                "source": e.source_instrument,
                "target": e.target_instrument,
                "file": e.source_file,
                "page": e.source_page,
            }
            for e in edges
            if (e.source_instrument, e.target_instrument)
            in {
                ("cir:2024:14", "cir:2007:18"),
                ("cir:2019:7", "cir:2018:7"),
                ("cir:2025:12", "cir:1972:56"),
                ("cir:2023:5", "cir:2017:6"),
            }
        ],
    }
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    if not verification["resolved_matches_output"] or not verification["pin_enabled"]:
        return 1
    if verification["false_2026_04_amends_2025_13"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
