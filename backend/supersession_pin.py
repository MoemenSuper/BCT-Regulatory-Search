"""JSONL SUPERSEDES pin: rank/pin declaring pages onto retrieval hits."""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from query_currentness import is_temporal_rule_query
from retrieval_selection import (
    prefer_historical_hits,
    prefer_named_instrument_hits,
    query_instrument_refs,
)
from source_metadata import normalize_page
from supersession_edges import (
    ARTICLE,
    FORCEISH,
    SupersessionEdge,
    instrument_from_filename,
    instruments_from_text,
    load_edges,
    resolve_edges_path,
)

logger = logging.getLogger(__name__)

def select_edges(edges: list[SupersessionEdge], query: str, *, limit: int = 2) -> list[SupersessionEdge]:
    named = instruments_from_text(query)
    if not named:
        return []
    arts = {m.group("a") for m in ARTICLE.finditer(query)}
    by_target: dict[str, list[SupersessionEdge]] = defaultdict(list)
    by_source: dict[str, list[SupersessionEdge]] = defaultdict(list)
    for edge in edges:
        by_target[edge.target_instrument].append(edge)
        by_source[edge.source_instrument].append(edge)
    scored: list[tuple[tuple, SupersessionEdge]] = []
    for instrument in named:
        for edge in by_target.get(instrument, []):
            if arts and edge.target_article and edge.target_article not in arts:
                continue
            art_score = 0 if (not arts or edge.target_article in arts or edge.target_article is None) else 1
            both = 0 if edge.source_instrument in named else 1
            scored.append(((both, art_score, -int(edge.source_instrument.split(":")[1])), edge))
        if len(named) >= 2:
            for edge in by_source.get(instrument, []):
                if edge.target_instrument not in named:
                    continue
                scored.append(((0, 0, -int(edge.source_instrument.split(":")[1])), edge))
        elif FORCEISH.search(query or ""):
            # Single named instrument that is itself an amending source
            for edge in by_source.get(instrument, []):
                scored.append(((1, 0, -int(edge.source_instrument.split(":")[1])), edge))
    scored.sort(key=lambda row: row[0])
    out: list[SupersessionEdge] = []
    seen: set[tuple] = set()
    for _key, edge in scored:
        key = (edge.source_instrument, edge.target_instrument, edge.target_article, edge.source_page)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
        if len(out) >= limit:
            break
    return out


def _source_name(document: Document) -> str:
    return Path(str(document.metadata.get("source") or "")).name


def _page_label(document: Document) -> int | None:
    meta = document.metadata or {}
    raw = meta.get("page_label", meta.get("page"))
    try:
        return normalize_page(raw, meta)
    except Exception:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None


def select_edges_from_hits(
    edges: list[SupersessionEdge],
    ranked: list[tuple[Document, float]],
    *,
    limit: int = 3,
    look_at: int = 8,
) -> list[SupersessionEdge]:
    """If top hits include superseded instruments, return edges that supersede them.

    This is the topical-currentness path: user asks about hours/rates without naming
    a circular; classic retrieve returns the old page; we pin the amending successor.
    """
    if not edges or not ranked:
        return []
    by_target: dict[str, list[SupersessionEdge]] = defaultdict(list)
    for edge in edges:
        if edge.action not in {"ABROGATE", "REPLACE", "AMEND"}:
            continue
        by_target[edge.target_instrument].append(edge)

    hit_instruments: list[str] = []
    for doc, _score in ranked[:look_at]:
        instrument = instrument_from_filename(_source_name(doc))
        if instrument and instrument not in hit_instruments:
            hit_instruments.append(instrument)

    scored: list[tuple[tuple, SupersessionEdge]] = []
    for instrument in hit_instruments:
        for edge in by_target.get(instrument, []):
            # Prefer hard supersession, then later amending year.
            action_rank = 0 if edge.action in {"ABROGATE", "REPLACE"} else 1
            year = int(edge.source_instrument.split(":")[1])
            scored.append(((action_rank, -year, edge.source_page), edge))
    scored.sort(key=lambda row: row[0])
    out: list[SupersessionEdge] = []
    seen: set[tuple] = set()
    for _key, edge in scored:
        key = (edge.source_instrument, edge.target_instrument, edge.target_article, edge.source_page)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
        if len(out) >= limit:
            break
    return out


def _edge_document(edge: SupersessionEdge, page_lookup) -> Document | None:
    chosen, page_label, text = page_lookup(edge)
    if not chosen or page_label is None:
        return None
    return Document(
        page_content=text,
        metadata={
            "source": chosen,
            "page": int(page_label),
            "pages": [int(page_label)],
            "page_label": int(page_label),
            "retrieval_source": "jsonl_supersession",
            "temporal_relation": (
                "ABROGATES"
                if edge.action == "ABROGATE"
                else "AMENDS"
                if edge.action == "AMEND"
                else "REPLACES"
            ),
            "temporal_source_id": edge.source_instrument,
            "temporal_target_id": edge.target_instrument,
        },
    )


def _demote_fully_superseded(
    ranked: list[tuple[Document, float]],
    edges: list[SupersessionEdge],
    pinned_sources: set[str],
) -> list[tuple[Document, float]]:
    """Push abrogated/replaced instruments below live hits, only when their successor is pinned."""
    if not ranked or not pinned_sources:
        return ranked
    superseded: set[str] = set()
    for edge in edges:
        if edge.action not in {"ABROGATE", "REPLACE"}:
            continue
        if edge.source_instrument in pinned_sources:
            superseded.add(edge.target_instrument)
    if not superseded:
        return ranked
    keep: list[tuple[Document, float]] = []
    demoted: list[tuple[Document, float]] = []
    for doc, score in ranked:
        instrument = instrument_from_filename(_source_name(doc))
        if instrument and instrument in superseded:
            demoted.append((doc, score))
        else:
            keep.append((doc, score))
    return keep + demoted if demoted else ranked


def _prefer_successor_instrument_hits(
    ranked: list[tuple[Document, float]],
    pinned_sources: set[str],
    *,
    max_boost: int = 1,
) -> list[tuple[Document, float]]:
    """Keep a few already-retrieved successor pages near the front.

    Declaring pages alone often only state the abrogation; topical answers need
    the successor's substantive rule pages when classic retrieve already found them.
    Cap the boost so successor flood does not erase the triggering older hit.
    """
    if not ranked or not pinned_sources:
        return ranked
    front: list[tuple[Document, float]] = []
    rest: list[tuple[Document, float]] = []
    boosted = 0
    for doc, score in ranked:
        instrument = instrument_from_filename(_source_name(doc))
        if instrument and instrument in pinned_sources:
            if float(score) >= 8000.0:
                front.append((doc, score))
            elif boosted < max_boost:
                front.append((doc, max(float(score), 8500.0)))
                boosted += 1
            else:
                rest.append((doc, score))
        else:
            rest.append((doc, score))
    return front + rest if front else ranked


def pin_supersession_edges(
    ranked: list[tuple[Document, float]],
    query: str,
    edges: list[SupersessionEdge],
    *,
    page_lookup,
) -> list[tuple[Document, float]]:
    """Pin declaring pages for (1) instruments named in the query and (2) superseded
    instruments that already appear in classic top hits (topical currentness)."""
    if not edges or not ranked:
        return ranked
    picked: list[SupersessionEdge] = []
    seen_edge: set[tuple] = set()
    candidates = select_edges(edges, query, limit=2) + select_edges_from_hits(
        edges, ranked, limit=3
    )
    for edge in candidates:
        key = (edge.source_instrument, edge.target_instrument, edge.target_article, edge.source_page)
        if key in seen_edge:
            continue
        seen_edge.add(key)
        picked.append(edge)
        if len(picked) >= 2:
            break
    if not picked:
        return ranked

    seen = {(_source_name(doc).casefold(), _page_label(doc)) for doc, _ in ranked}
    extras: list[tuple[Document, float]] = []
    working = list(ranked)
    pinned_sources: set[str] = set()
    for edge in picked:
        doc = _edge_document(edge, page_lookup)
        if doc is None:
            continue
        pinned_sources.add(edge.source_instrument)
        key = (_source_name(doc).casefold(), _page_label(doc))
        if key in seen:
            working = [(doc, 9000.0)] + [
                (d, s)
                for d, s in working
                if not (
                    _source_name(d).casefold() == key[0] and _page_label(d) == key[1]
                )
            ]
        else:
            seen.add(key)
            extras.append((doc, 9000.0))
    combined = extras + working if extras else working
    combined = _prefer_successor_instrument_hits(combined, pinned_sources)
    combined = _demote_fully_superseded(combined, edges, pinned_sources)
    # "Is X still in force?" wants the successor first; "Selon X" wants X itself first.
    if not is_temporal_rule_query(query):
        combined = prefer_named_instrument_hits(combined, query_instrument_refs(query))
    return prefer_historical_hits(combined, query)


class SupersessionPinBackend:
    """Wrap a retrieval backend; pin JSONL SUPERSEDES pages when instruments match."""

    def __init__(self, inner, edges: list[SupersessionEdge], page_lookup):
        self.inner = inner
        self.edges = edges
        self.page_lookup = page_lookup
        self.expand_pages = getattr(inner, "expand_pages", None)

    def retrieve(self, query: str):
        ranked = list(self.inner.retrieve(query))
        return pin_supersession_edges(
            ranked, query, self.edges, page_lookup=self.page_lookup
        )


def build_page_lookup_from_native(native_path: Path):
    """Map edge → (filename, page_label, text) using native.jsonl when present."""
    by_file: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    if not native_path.is_file():
        return lambda edge: (edge.source_file, edge.source_page, edge.quote)

    with native_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            meta = row.get("metadata") or {}
            source = Path(str(meta.get("source") or "")).name
            page = meta.get("page")
            if not source or page is None:
                continue
            label = int(page) if "pages" in meta else int(page) + 1
            by_file[source.casefold()].append(
                (label, str(row.get("page_content") or ""), source)
            )

    def aliases(name: str) -> list[str]:
        stem = Path(name).name
        return list(
            {
                stem,
                stem.replace("CB_", "Cir_"),
                stem.replace("Cir_", "CB_"),
                stem.replace("_FR", "_fr"),
                stem.replace("_fr", "_FR"),
            }
        )

    def fold(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").casefold())

    def lookup(edge: SupersessionEdge) -> tuple[str, int, str]:
        needle = fold(edge.quote)[:120]
        candidates = []
        for alias in aliases(edge.source_file):
            for page_label, text, chosen in by_file.get(alias.casefold(), []):
                if needle and needle in fold(text):
                    candidates.append((0, chosen, page_label, text))
                elif abs(page_label - edge.source_page) <= 2:
                    candidates.append(
                        (1 + abs(page_label - edge.source_page), chosen, page_label, text)
                    )
        if candidates:
            candidates.sort(key=lambda row: row[0])
            _rank, chosen, page_label, text = candidates[0]
            body = f"{edge.quote.strip()}\n\n{text}" if edge.quote.strip() else text
            return chosen, page_label, body[:6000]
        return edge.source_file, edge.source_page, (edge.quote or "")[:6000]

    return lookup


@lru_cache(maxsize=4)
def _cached_bundle(edges_path: str, native_path: str, edges_mtime: float, native_mtime: float):
    edges = load_edges(Path(edges_path))
    lookup = build_page_lookup_from_native(Path(native_path)) if native_path else (
        lambda edge: (edge.source_file, edge.source_page, edge.quote)
    )
    return edges, lookup


def clear_supersession_cache() -> None:
    _cached_bundle.cache_clear()


def maybe_wrap_backend(backend, runtime_assets: Path | None):
    """Return backend wrapped with supersession pin when an edges file is present."""
    edges_path = resolve_edges_path(runtime_assets)
    if edges_path is None:
        return backend
    native = Path(runtime_assets) / "native.jsonl" if runtime_assets else Path()
    try:
        edges_mtime = edges_path.stat().st_mtime
        native_mtime = native.stat().st_mtime if native.is_file() else 0.0
        edges, lookup = _cached_bundle(
            str(edges_path),
            str(native) if native.is_file() else "",
            edges_mtime,
            native_mtime,
        )
    except OSError as error:
        logger.warning("Supersession edges unavailable: %s", error)
        return backend
    if not edges:
        return backend
    logger.info("Supersession pin enabled (%s edges from %s)", len(edges), edges_path)
    return SupersessionPinBackend(backend, edges, lookup)
