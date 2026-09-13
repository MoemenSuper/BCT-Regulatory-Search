from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from retrieval_selection import parse_query_identity, parse_source_identity

from .models import InstrumentRef


_CANONICAL_ID = re.compile(r"^(?P<kind>cir|note):(?P<year>\d{4}):(?P<number>\d+)$", re.IGNORECASE)


_KIND_GROUPS = {
    "cir": "cir",
    "ci": "cir",
    "cb": "cir",
    "circulaire": "cir",
    "circular": "cir",
    "المنشور": "cir",
    "منشور": "cir",
    "note": "note",
    "nb": "note",
    "المذكرة": "note",
    "مذكرة": "note",
}


def normalize_kind(value: object) -> str | None:
    if value is None:
        return None
    return _KIND_GROUPS.get(str(value).strip().casefold())


def canonical_instrument_id(kind: str, year: int, number: int) -> str:
    normalized = normalize_kind(kind)
    if normalized is None:
        raise ValueError(f"Unsupported BCT instrument kind: {kind!r}")
    year = int(year)
    # BCT texts often write older circulars as n°91-24; expand to Gregorian years.
    if 0 <= year < 100:
        year = 1900 + year if year >= 50 else 2000 + year
    return f"{normalized}:{year}:{int(number)}"


def _ref(kind: object, year: object, number: object, filename: str | None = None) -> InstrumentRef | None:
    normalized = normalize_kind(kind)
    if normalized is None:
        return None
    try:
        return InstrumentRef(
            id=canonical_instrument_id(normalized, year, number),
            kind=normalized,
            year=int(year),
            number=int(number),
            filename=filename,
        )
    except (TypeError, ValueError):
        return None


def instrument_from_filename(filename: str) -> InstrumentRef | None:
    identity = parse_source_identity(Path(filename).name)
    if identity is None:
        return None
    return _ref(identity.get("kind"), identity["year"], identity["number"], Path(filename).name)


def instrument_from_text(text: str) -> InstrumentRef | None:
    canonical = _CANONICAL_ID.match(text.strip())
    if canonical:
        values = canonical.groupdict()
        return _ref(values["kind"], values["year"], values["number"])

    identity = parse_query_identity(text)
    if identity is None or identity.get("number") is None or identity.get("kind") is None:
        return None
    return _ref(identity["kind"], identity["year"], identity["number"])


def instrument_from_properties(properties: dict[str, object]) -> InstrumentRef | None:
    name = str(properties.get("name") or properties.get("filename") or "").strip()
    if name:
        parsed = instrument_from_filename(name) or instrument_from_text(name)
        if parsed is not None:
            return parsed

    return _ref(properties.get("kind"), properties.get("year"), properties.get("number"), name or None)


def build_catalog(
    *,
    documents_dir: str | Path | None = None,
    chunk_files: Iterable[str | Path] = (),
) -> dict[str, InstrumentRef]:
    """Build a trusted identity catalog from filenames already present in the corpus."""
    catalog: dict[str, InstrumentRef] = {}
    if documents_dir is not None:
        root = Path(documents_dir)
        if root.exists():
            for path in root.rglob("*.pdf"):
                ref = instrument_from_filename(path.name)
                if ref is not None:
                    catalog[ref.id] = ref

    for chunk_file in chunk_files:
        path = Path(chunk_file)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                source = str(value.get("metadata", {}).get("source", ""))
                ref = instrument_from_filename(source)
                if ref is not None:
                    catalog.setdefault(ref.id, ref)

    return catalog
