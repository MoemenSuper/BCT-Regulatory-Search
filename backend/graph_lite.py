"""CLI for the bounded BCT relationship graph.

Examples:
  python graph_lite.py ingest-page --source Cir_2025_17_fr.pdf --page 3 --text-file page3.txt --catalog-dir documents
  python graph_lite.py ingest-jsonl --input extracted_pages.jsonl --catalog-dir documents
  python graph_lite.py bootstrap-assets --native-chunks runtime-assets/native.jsonl --catalog-dir documents
  python graph_lite.py list-relationships
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from regulatory_graph_lite.builder import create_builder_from_environment, page_may_contain_relationship
from regulatory_graph_lite.models import VerificationStatus
from regulatory_graph_lite.store import Neo4jGraphLiteStore, open_neo4j_driver_from_environment
from source_metadata import normalize_page


def _store():
    driver = open_neo4j_driver_from_environment()
    store = Neo4jGraphLiteStore(driver, database=os.environ.get("BCT_NEO4J_DATABASE", "neo4j"))
    return driver, store


def _builder(args, driver, store):
    chunk_files = [args.native_chunks] if getattr(args, "native_chunks", None) else []
    return create_builder_from_environment(
        driver=driver,
        store=store,
        documents_dir=getattr(args, "catalog_dir", None),
        chunk_files=chunk_files,
    )


def _print_report(source, page, report):
    print(json.dumps({
        "source": source,
        "page": page,
        "accepted": [candidate.as_dict() for candidate in report.accepted],
        "rejected_builder_edges": report.rejected_count,
    }, ensure_ascii=False, indent=2))


def _ingest(pages: Iterable[tuple[str, int, str]], builder, *, limit: int = 0, verbose: bool = True):
    processed = skipped = proposed = 0
    for source, page, text in pages:
        if not page_may_contain_relationship(text):
            skipped += 1
            continue
        report = builder.ingest_page(source_file=source, page=page, text=text)
        processed += 1
        proposed += len(report.accepted)
        if verbose:
            print(f"[{processed}] {Path(source).name} p.{page}: {len(report.accepted)} candidate(s)", flush=True)
        if limit and processed >= limit:
            break
    return processed, skipped, proposed


def ingest_page(args) -> int:
    driver, store = _store()
    try:
        store.ensure_schema()
        builder = _builder(args, driver, store)
        text = Path(args.text_file).read_text(encoding="utf-8")
        report = builder.ingest_page(source_file=args.source, page=args.page, text=text)
        _print_report(args.source, args.page, report)
        return 0
    finally:
        driver.close()


def ingest_jsonl(args) -> int:
    def pages():
        with Path(args.input).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                yield str(value["source"]), int(value["page"]), str(value["text"])

    driver, store = _store()
    try:
        store.ensure_schema()
        builder = _builder(args, driver, store)
        processed, skipped, proposed = _ingest(pages(), builder, limit=args.limit)
        print(json.dumps({"processed_pages": processed, "prefilter_skipped": skipped, "stored_candidates": proposed}, indent=2))
        return 0
    finally:
        driver.close()


def _group_runtime_pages(path: Path):
    pages: dict[tuple[str, int], list[tuple[int, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            metadata = value.get("metadata", {})
            source = str(metadata.get("source", ""))
            page = normalize_page(metadata)
            if not source or type(page) is not int or page < 1:
                continue
            order = metadata.get("flat_part", metadata.get("chunk_index", 0))
            try:
                order = int(order)
            except (TypeError, ValueError):
                order = 0
            pages.setdefault((source, page), []).append((order, value["page_content"]))
    for (source, page), chunks in sorted(pages.items()):
        # Preserve all unique chunk text in recorded order. Overlap is harmless for
        # proposal extraction; evidence validation still recovers an exact substring.
        seen = set()
        texts = []
        for _order, text in sorted(chunks, key=lambda item: item[0]):
            if text not in seen:
                seen.add(text)
                texts.append(text)
        yield source, page, "\n\n".join(texts)


def bootstrap_assets(args) -> int:
    source_filter = {Path(value).name.casefold() for value in (args.source or [])}

    def pages():
        for source, page, text in _group_runtime_pages(Path(args.native_chunks)):
            if source_filter and Path(source).name.casefold() not in source_filter:
                continue
            yield source, page, text

    driver, store = _store()
    try:
        store.ensure_schema()
        builder = _builder(args, driver, store)
        processed, skipped, proposed = _ingest(pages(), builder, limit=args.limit)
        print(json.dumps({"processed_pages": processed, "prefilter_skipped": skipped, "stored_candidates": proposed}, indent=2))
        return 0
    finally:
        driver.close()


def list_relationships(args) -> int:
    driver, store = _store()
    try:
        rows = store.list_candidates(status=VerificationStatus.VERIFIED, limit=args.limit)
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        driver.close()

def _add_catalog(parser, *, required=False):
    parser.add_argument(
        "--catalog-dir",
        required=required,
        help="Directory containing trusted BCT PDFs",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="BCT Graph Lite automatic relationship extraction")
    parser.add_argument("--env-file", default=".env")
    sub = parser.add_subparsers(dest="command", required=True)

    item = sub.add_parser("ingest-page")
    item.add_argument("--source", required=True)
    item.add_argument("--page", required=True, type=int)
    item.add_argument("--text-file", required=True)
    _add_catalog(item, required=True)
    item.set_defaults(func=ingest_page)

    batch = sub.add_parser("ingest-jsonl")
    batch.add_argument("--input", required=True, help='JSONL rows: {"source":..., "page":..., "text":...}')
    batch.add_argument("--limit", type=int, default=0)
    _add_catalog(batch, required=True)
    batch.set_defaults(func=ingest_jsonl)

    boot = sub.add_parser("bootstrap-assets")
    boot.add_argument("--native-chunks", required=True)
    boot.add_argument("--source", action="append", help="Optional filename filter; repeatable")
    boot.add_argument("--limit", type=int, default=0, help="Maximum prefiltered pages to send to the LLM")
    _add_catalog(boot, required=False)
    boot.set_defaults(func=bootstrap_assets)

    listing = sub.add_parser("list-relationships")
    listing.add_argument("--limit", type=int, default=100)
    listing.set_defaults(func=list_relationships)

    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
