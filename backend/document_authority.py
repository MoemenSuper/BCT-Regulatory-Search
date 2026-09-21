"""Document kind / authority tags for mixed regulatory–stats–internal corpora.

Tag at ingest; enforce at claim time. Retrieval stays one corpus.
"""
from __future__ import annotations

from typing import Literal

DocKind = Literal["regulatory", "statistical", "internal"]
Authority = Literal["primary", "secondary"]

DOC_KINDS: tuple[str, ...] = ("regulatory", "statistical", "internal")
_KIND_ALIASES = {
    "regulatory": "regulatory",
    "regulation": "regulatory",
    "circulaire": "regulatory",
    "circular": "regulatory",
    "note": "regulatory",
    "reglementaire": "regulatory",
    "réglementaire": "regulatory",
    "statistical": "statistical",
    "statistic": "statistical",
    "statistics": "statistical",
    "stats": "statistical",
    "bulletin": "statistical",
    "internal": "internal",
    "interne": "internal",
    "procedure": "internal",
    "procédure": "internal",
    "memo": "internal",
}


def normalize_doc_kind(value: object | None) -> DocKind:
    raw = str(value or "").strip().casefold()
    if not raw:
        return "regulatory"
    if raw in _KIND_ALIASES:
        return _KIND_ALIASES[raw]  # type: ignore[return-value]
    for key, kind in _KIND_ALIASES.items():
        if key in raw:
            return kind  # type: ignore[return-value]
    return "regulatory"


def authority_for_kind(kind: object | None) -> Authority:
    return "primary" if normalize_doc_kind(kind) == "regulatory" else "secondary"


def infer_doc_kind_from_filename(filename: str) -> DocKind:
    stem = str(filename or "").casefold()
    if any(token in stem for token in ("stat", "bulletin", "indicateur", "stats", "tableau")):
        return "statistical"
    if any(token in stem for token in ("interne", "internal", "procedure", "procédure", "memo", "manuel")):
        return "internal"
    return "regulatory"


def resolve_doc_kind(*, explicit: object | None = None, filename: str = "") -> DocKind:
    if explicit is not None and str(explicit).strip():
        return normalize_doc_kind(explicit)
    return infer_doc_kind_from_filename(filename)
