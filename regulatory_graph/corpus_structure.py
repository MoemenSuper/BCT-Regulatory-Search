from dataclasses import dataclass
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from regulatory_graph.models import (
    GraphChunk,
    GraphPage,
    Instrument,
    InstrumentKind,
    Language,
    RegulatoryGraphBundle,
    SourceEdition,
    SourceStatus,
    VerificationStatus,
)


@dataclass(frozen=True)
class InstrumentIdentityResolution:
    instrument_uid: str
    kind: InstrumentKind
    year: int | None
    number: str | None
    language: Language
    status: VerificationStatus
    evidence: str


class CorpusCacheError(ValueError):
    """Raised before graph writes when frozen corpus artifacts do not reconcile."""


@dataclass(frozen=True)
class CachedEdition:
    relative_path: str
    filename: str
    pdf_sha256: str
    artifact_path: Path
    artifact_sha256: str
    document: dict[str, Any]
    identity: InstrumentIdentityResolution


@dataclass(frozen=True)
class CacheInventory:
    editions: tuple[CachedEdition, ...]
    chunks: tuple[dict[str, Any], ...]
    pdf_count: int
    page_count: int
    chunk_count: int
    chunk_page_count: int
    language_counts: dict[str, int]
    manifest_sha256: str
    chunks_sha256: str
    artifact_hash_aggregate: str


_NORMAL_FILENAME = re.compile(
    r"^(?P<kind>Cir|Note)_(?P<year>\d{4})_(?P<number>\d+)_(?P<language>fr|ar)\.pdf$",
    re.IGNORECASE,
)
_LEGACY_FILENAMES = (
    (
        re.compile(
            r"^(?:CB|CI)_(?P<year>\d{4})_(?P<number>\d+)_(?P<language>fr|ar)\.pdf$",
            re.IGNORECASE,
        ),
        InstrumentKind.CIRCULAR,
    ),
    (
        re.compile(
            r"^Cir(?P<year>\d{4})(?P<number>\d{2})_(?P<language>fr|ar)\.pdf$",
            re.IGNORECASE,
        ),
        InstrumentKind.CIRCULAR,
    ),
    (
        re.compile(
            r"^NB[-_](?P<year>\d{4})_(?P<number>\d+)(?:_[^.]*)?_(?P<language>fr|ar)\.pdf$",
            re.IGNORECASE,
        ),
        InstrumentKind.NOTE,
    ),
)


def inventory_corpus_cache(
    documents_dir: str | Path,
    manifest_path: str | Path,
    chunks_path: str | Path,
) -> CacheInventory:
    documents_root = Path(documents_dir).resolve()
    manifest_file = Path(manifest_path).resolve()
    chunks_file = Path(chunks_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("errors"):
        raise CorpusCacheError("ingestion manifest contains extraction errors")
    records = manifest.get("records", {})
    pdfs = sorted(documents_root.rglob("*.pdf"))
    relative_paths = {path.relative_to(documents_root).as_posix() for path in pdfs}
    if relative_paths != set(records):
        missing = sorted(relative_paths - set(records))
        stale = sorted(set(records) - relative_paths)
        raise CorpusCacheError(
            f"manifest/corpus path mismatch: missing={missing}, stale={stale}"
        )
    if manifest.get("document_count", len(records)) != len(records):
        raise CorpusCacheError("manifest document count does not match its records")

    editions = []
    pages_by_source: dict[str, set[int]] = {}
    artifact_hashes = []
    languages: Counter[str] = Counter()
    for pdf_path in pdfs:
        relative_path = pdf_path.relative_to(documents_root).as_posix()
        record = records[relative_path]
        pdf_hash = _path_sha256(pdf_path)
        if pdf_hash.casefold() != str(record["sha256"]).casefold():
            raise CorpusCacheError(f"PDF hash mismatch: {relative_path}")
        artifact_path = Path(record["artifact"]).resolve()
        if not artifact_path.is_file():
            raise CorpusCacheError(f"structured artifact missing: {relative_path}")
        artifact_hash = _path_sha256(artifact_path)
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("filename") != record["source"]:
            raise CorpusCacheError(f"artifact filename mismatch: {relative_path}")
        pages = artifact.get("pages", [])
        if len(pages) != int(record["pages"]):
            raise CorpusCacheError(f"artifact page count mismatch: {relative_path}")
        page_numbers = {int(page["page_number"]) for page in pages}
        if len(page_numbers) != len(pages):
            raise CorpusCacheError(f"duplicate artifact page number: {relative_path}")
        source = str(record["source"])
        if source in pages_by_source:
            raise CorpusCacheError(f"duplicate source filename: {source}")
        pages_by_source[source] = page_numbers
        first_page_text = str(pages[0].get("raw_text", "")) if pages else ""
        identity = resolve_instrument_identity(source, first_page_text)
        artifact_hashes.append(artifact_hash)
        languages[str(artifact.get("language", "unknown"))] += 1
        editions.append(
            CachedEdition(
                relative_path=relative_path,
                filename=source,
                pdf_sha256=pdf_hash,
                artifact_path=artifact_path,
                artifact_sha256=artifact_hash,
                document=artifact,
                identity=identity,
            )
        )

    chunks = []
    chunk_pages = set()
    with chunks_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            metadata = chunk.get("metadata", {})
            source = str(metadata.get("source", ""))
            if source not in pages_by_source:
                raise CorpusCacheError(f"chunk source missing at line {line_number}: {source}")
            pages = _chunk_pages(metadata)
            if not pages or not pages.issubset(pages_by_source[source]):
                raise CorpusCacheError(
                    f"chunk page missing at line {line_number}: {source} {sorted(pages)}"
                )
            if not str(chunk.get("page_content", "")).strip():
                raise CorpusCacheError(f"empty chunk text at line {line_number}")
            chunks.append(chunk)
            chunk_pages.update((source, page) for page in pages)

    aggregate = sha256(
        "".join(sorted(value.casefold() for value in artifact_hashes)).encode("ascii")
    ).hexdigest().upper()
    return CacheInventory(
        editions=tuple(editions),
        chunks=tuple(chunks),
        pdf_count=len(editions),
        page_count=sum(len(item.document["pages"]) for item in editions),
        chunk_count=len(chunks),
        chunk_page_count=len(chunk_pages),
        language_counts=dict(sorted(languages.items())),
        manifest_sha256=_path_sha256(manifest_file),
        chunks_sha256=_path_sha256(chunks_file),
        artifact_hash_aggregate=aggregate,
    )


def build_structural_bundle(inventory: CacheInventory) -> RegulatoryGraphBundle:
    instruments_by_uid: dict[str, Instrument] = {}
    editions = []
    pages = []
    edition_uid_by_source = {}
    cached_by_source = {item.filename: item for item in inventory.editions}
    for cached in inventory.editions:
        identity = cached.identity
        instruments_by_uid.setdefault(
            identity.instrument_uid,
            Instrument(
                uid=identity.instrument_uid,
                authority="BCT",
                kind=identity.kind,
                year=identity.year,
                number=identity.number,
                corpus_present=True,
                source_status=SourceStatus.LOCAL,
            ),
        )
        edition_uid = f"edition:{_slug(Path(cached.filename).stem)}"
        page_uid_prefix = edition_uid.removeprefix("edition:")
        edition_uid_by_source[cached.filename] = edition_uid
        document_pages = cached.document["pages"]
        editions.append(
            SourceEdition(
                uid=edition_uid,
                instrument_uid=identity.instrument_uid,
                language=identity.language,
                filename=cached.filename,
                sha256=cached.pdf_sha256,
                extraction_status="cached_hash_verified",
                page_count=len(document_pages),
                is_scan=bool(document_pages)
                and all(page.get("extraction_method") != "native" for page in document_pages),
                relative_path=cached.relative_path,
                extraction_artifact_hash=cached.artifact_sha256,
            )
        )
        for page in document_pages:
            page_number = int(page["page_number"])
            raw_text = str(page.get("raw_text", ""))
            pages.append(
                GraphPage(
                    uid=f"page:{page_uid_prefix}:{page_number}",
                    source_edition_uid=edition_uid,
                    page_number=page_number,
                    page_label=str(page_number),
                    source_sha256=cached.pdf_sha256,
                    extraction_artifact_hash=cached.artifact_sha256,
                    text_hash=sha256(raw_text.encode("utf-8")).hexdigest(),
                    extraction_method=str(page.get("extraction_method", "unknown")),
                    quality_score=page.get("quality_score"),
                    quality_flags=tuple(str(flag) for flag in page.get("quality_flags", [])),
                )
            )

    chunk_indexes: Counter[tuple[str, int]] = Counter()
    chunks = []
    for raw_chunk in inventory.chunks:
        metadata = raw_chunk["metadata"]
        source = str(metadata["source"])
        page_numbers = tuple(sorted(_chunk_pages(metadata)))
        start_page = int(metadata.get("page", page_numbers[0]))
        position = (source, start_page)
        chunk_index = chunk_indexes[position]
        chunk_indexes[position] += 1
        text = str(raw_chunk["page_content"])
        content_hash = sha256(text.encode("utf-8")).hexdigest()
        cached = cached_by_source[source]
        edition_uid = edition_uid_by_source[source]
        page_uid_prefix = edition_uid.removeprefix("edition:")
        chunks.append(
            GraphChunk(
                uid=(
                    f"chunk:{edition_uid}:{start_page}:{chunk_index}:"
                    f"{content_hash[:16]}"
                ),
                page_uid=f"page:{page_uid_prefix}:{start_page}",
                chunk_index=chunk_index,
                text=text,
                content_hash=content_hash,
                source_sha256=cached.pdf_sha256,
                extraction_artifact_hash=cached.artifact_sha256,
                extraction_method=str(metadata.get("extraction_methods", "unknown")),
                page_numbers=page_numbers,
            )
        )

    return RegulatoryGraphBundle(
        instruments=tuple(sorted(instruments_by_uid.values(), key=lambda item: item.uid)),
        source_editions=tuple(editions),
        pages=tuple(pages),
        chunks=tuple(chunks),
        provisions=(),
        provision_versions=(),
        evidence_spans=(),
        change_events=(),
    )


def resolve_instrument_identity(
    filename: str,
    first_page_text: str,
) -> InstrumentIdentityResolution:
    normal = _NORMAL_FILENAME.fullmatch(filename)
    if normal:
        kind = _kind_from_filename(normal.group("kind"))
        return _resolved_identity(normal, kind, evidence="filename_identity")

    for pattern, kind in _LEGACY_FILENAMES:
        legacy = pattern.fullmatch(filename)
        if legacy is None:
            continue
        if _first_page_confirms(
            first_page_text,
            kind=kind,
            year=legacy.group("year"),
            number=legacy.group("number"),
        ):
            return _resolved_identity(legacy, kind, evidence="first_page_identity")
        break

    language_match = re.search(r"_(fr|ar)\.pdf$", filename, re.IGNORECASE)
    language: Language = (
        language_match.group(1).lower() if language_match else "unknown"
    )
    digest = sha256(filename.casefold().encode("utf-8")).hexdigest()[:16]
    return InstrumentIdentityResolution(
        instrument_uid=f"BCT:UNRESOLVED:{digest}",
        kind=InstrumentKind.OTHER,
        year=None,
        number=None,
        language=language,
        status=VerificationStatus.NEEDS_REVIEW,
        evidence="identity_not_verified",
    )


def _resolved_identity(
    match: re.Match[str],
    kind: InstrumentKind,
    *,
    evidence: str,
) -> InstrumentIdentityResolution:
    year = int(match.group("year"))
    number = _normalized_number(match.group("number"))
    language: Language = match.group("language").lower()
    return InstrumentIdentityResolution(
        instrument_uid=f"BCT:{kind.value}:{year}:{number}",
        kind=kind,
        year=year,
        number=number,
        language=language,
        status=VerificationStatus.VERIFIED,
        evidence=evidence,
    )


def _kind_from_filename(value: str) -> InstrumentKind:
    return InstrumentKind.CIRCULAR if value.casefold() == "cir" else InstrumentKind.NOTE


def _normalized_number(value: str) -> str:
    return value.zfill(2) if len(value) < 2 else value


def _first_page_confirms(
    text: str,
    *,
    kind: InstrumentKind,
    year: str,
    number: str,
) -> bool:
    normalized = re.sub(r"\s+", " ", text).casefold()
    kind_confirmed = (
        "circulaire" in normalized or "منشور" in normalized
        if kind == InstrumentKind.CIRCULAR
        else "note" in normalized
        or "مذكرة" in normalized
        or "مذ كرة" in normalized
    )
    french_identity = re.search(
        rf"n\s*[°º]?\s*{re.escape(year)}\s*[-–]\s*0*{int(number)}\b",
        normalized,
    )
    arabic_identity = re.search(
        rf"عدد\s*0*{int(number)}\s*لسنة\s*{re.escape(year)}\b",
        normalized,
    )
    return bool(kind_confirmed and (french_identity or arabic_identity))


def _path_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest().upper()


def _chunk_pages(metadata: dict[str, Any]) -> set[int]:
    raw_pages = metadata.get("pages")
    if isinstance(raw_pages, list):
        return {int(page) for page in raw_pages}
    if isinstance(raw_pages, str) and raw_pages.strip():
        return {int(page) for page in raw_pages.split(",")}
    start = int(metadata.get("page", -1))
    end = int(metadata.get("page_end", start))
    return set(range(start, end + 1)) if start >= 1 and end >= start else set()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
