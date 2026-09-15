"""Re-extract PDFs whose active pages have unreliable native digits, into a candidate asset root.

The live corpus is never touched: pass a candidate root (seeded from the live root with
--seed-from). Each affected PDF goes through the normal ingestion pipeline; pages whose
native text contradicts the trusted filename (reversed or garbled digits) are replaced by
Gemini's transcription of the page image. Compare, then switch the API to the candidate
root when satisfied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from answer_evidence import evidence_warning
from ingestion.index import resolve_active_assets
from ingestion.pipeline import IngestionConfig, IngestionPipeline
from runtime_retrieval import _read_chunks


def warned_pages(asset_root: Path) -> dict[str, set[int]]:
    """Pages in the active native representation whose digits cannot be trusted."""
    pages: dict[str, set[int]] = defaultdict(set)
    texts: dict[tuple[str, int], list[str]] = defaultdict(list)
    for chunk in _read_chunks(resolve_active_assets(asset_root) / "native.jsonl"):
        texts[(Path(str(chunk.metadata.get("source"))).name, int(chunk.metadata.get("page") or 0))].append(chunk.page_content)
    for (source, page), parts in texts.items():
        if evidence_warning({"source": source, "text": "\n".join(parts)}):
            pages[source].add(page)
    return pages


def seed_candidate(live_root: Path, candidate_root: Path) -> None:
    active = resolve_active_assets(live_root)
    if candidate_root.exists() and any(candidate_root.iterdir()):
        raise SystemExit(f"candidate root is not empty: {candidate_root}")
    target = candidate_root / "versions" / active.name if active != live_root else candidate_root
    shutil.copytree(active, target, ignore=shutil.ignore_patterns("local_chroma", "versions", "voyage-cache", "ingestion-data"))
    if active != live_root:
        shutil.copy2(live_root / "ACTIVE.json", candidate_root / "ACTIVE.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Candidate asset root (never the live one)")
    parser.add_argument("--seed-from", type=Path, help="Live asset root to copy into --assets first")
    parser.add_argument("--documents", type=Path, help="Directory searched recursively for the source PDFs")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=0, help="Re-ingest at most N PDFs (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Only list affected PDFs and pages")
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)

    if args.seed_from:
        seed_candidate(args.seed_from.resolve(strict=True), args.assets.resolve())
    root = args.assets.resolve(strict=True)
    os.environ.setdefault("BCT_INGESTION_DATA_DIR", str(root / "ingestion-data"))
    os.environ.setdefault("BCT_VOYAGE_RUNTIME_CACHE", str(root / "voyage-cache"))

    before = warned_pages(root)
    print(f"{sum(map(len, before.values()))} unreliable pages in {len(before)} PDFs")
    if args.dry_run or not args.documents:
        for source in sorted(before):
            print(f"  {source}: pages {sorted(before[source])}")
        return 0

    pdfs = {path.name.casefold(): path for path in args.documents.resolve(strict=True).rglob("*.pdf")}
    config = replace(IngestionConfig.from_environment(asset_root=root), build_local=False, build_graph=False)
    pipeline = IngestionPipeline(config)
    failures: dict[str, str] = {}
    try:
        for index, source in enumerate(sorted(before)):
            if args.limit and index >= args.limit:
                break
            pdf = pdfs.get(source.casefold())
            if pdf is None:
                failures[source] = "pdf_not_found"
                continue
            # Same PDF bytes must be re-extracted; clear a prior ready row.
            from ingestion.extract import sha256_file
            digest = sha256_file(pdf)
            known = pipeline.registry.get(digest)
            if known and known.get("status") == "ready":
                pipeline.registry.fail(digest, "force_visual_reingest")
            try:
                report = pipeline.ingest(pdf)
            except Exception as error:  # keep going; the failed PDF keeps its old pages
                failures[source] = f"{type(error).__name__}: {error}"
                print(f"[FAIL] {source}: {failures[source]}")
                continue
            print(f"[ok] {source}: {report['pages']} pages, {report['gemini_visual_pages']} via Gemini, "
                  f"version {report['asset_version']}{' (duplicate)' if report.get('duplicate') else ''}")
    finally:
        pipeline.close()

    after = warned_pages(root)
    print(json.dumps({
        "unreliable_pages_before": sum(map(len, before.values())),
        "unreliable_pages_after": sum(map(len, after.values())),
        "still_unreliable": {source: sorted(pages) for source, pages in sorted(after.items())},
        "failures": failures,
        "active": str(resolve_active_assets(root)),
    }, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
