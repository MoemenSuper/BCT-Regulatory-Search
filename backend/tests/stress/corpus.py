"""Load trusted page text from committed fixtures (extracted from documents/)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "corpus_pages.json"


@lru_cache(maxsize=1)
def _bundle() -> dict[str, dict[str, str]]:
    return json.loads(_FIXTURES.read_text(encoding="utf-8"))


def page_text(source: str, page: int) -> str:
    """Return 1-based page text for a fixture PDF basename."""
    pages = _bundle().get(source)
    if not pages:
        raise KeyError(f"missing fixture source: {source}")
    text = pages.get(str(page))
    if text is None:
        raise KeyError(f"missing fixture page: {source} p.{page}")
    return text


def has_fixture(source: str, page: int) -> bool:
    pages = _bundle().get(source) or {}
    return str(page) in pages
