"""Ingest one BCT PDF into the active cloud/local runtime assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataclasses import replace

from dotenv import load_dotenv

from ingestion.pipeline import IngestionConfig, IngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--assets", type=Path, required=True, help="Runtime asset root (legacy root or root containing ACTIVE.json)")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--document-number")
    parser.add_argument("--publication-date")
    parser.add_argument("--title")
    parser.add_argument("--type")
    parser.add_argument("--category")
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)

    config = IngestionConfig.from_environment(asset_root=args.assets)
    if args.skip_local:
        config = replace(config, build_local=False)
    metadata = {
        key: value
        for key, value in {
            "document_number": args.document_number,
            "publication_date": args.publication_date,
            "title": args.title,
            "type": args.type,
            "category": args.category,
        }.items()
        if value is not None
    }
    pipeline = IngestionPipeline(config)
    try:
        report = pipeline.ingest(args.pdf, metadata=metadata or None)
    finally:
        pipeline.close()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
