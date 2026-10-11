"""What kind of document a PDF is, and whether it can prove a rule.

Every PDF is tagged once, at upload: regulatory (circulars, notes), statistical (bulletins,
tables) or internal (procedures, memos). Search looks at all of them together; only the answer
checks care: a rule must be proven by a regulatory text ("primary"), while a statistical bulletin
can support a figure but not a legal obligation ("secondary").
"""
from __future__ import annotations

from typing import Literal

DocKind = Literal["regulatory", "statistical", "internal"]
Authority = Literal["primary", "secondary"]

DOC_KINDS: tuple[str, ...] = ("regulatory", "statistical", "internal")
# The words an administrator or a filename might use for each kind, in French or English.
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
    """Any spelling ("Circulaire", "stats", "procédure") -> one of the three kinds.
    Unknown or empty -> "regulatory", the strictest kind for the answer checks."""
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
    """Only regulatory texts are "primary" evidence for a rule."""
    return "primary" if normalize_doc_kind(kind) == "regulatory" else "secondary"


def infer_doc_kind_from_filename(filename: str) -> DocKind:
    """A guess from the filename, used when nobody chose a kind at upload."""
    stem = str(filename or "").casefold()
    if any(token in stem for token in ("stat", "bulletin", "indicateur", "stats", "tableau")):
        return "statistical"
    if any(token in stem for token in ("interne", "internal", "procedure", "procédure", "memo", "manuel")):
        return "internal"
    return "regulatory"


def resolve_doc_kind(*, explicit: object | None = None, filename: str = "") -> DocKind:
    """The administrator's choice wins; otherwise guess from the filename."""
    if explicit is not None and str(explicit).strip():
        return normalize_doc_kind(explicit)
    return infer_doc_kind_from_filename(filename)
