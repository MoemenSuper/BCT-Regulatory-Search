"""JSONL SUPERSEDES pin: declaring pages for named instruments, relationship labels for topic hits,
and the replaced texts for "avant la circulaire X" questions."""
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
    is_historical_cutoff_query,
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


def _relation_metadata(edge: SupersessionEdge) -> dict:
    """The metadata the answer layer reads to say "X replaces / abrogates / amends Y"."""
    return {
        "temporal_relation": (
            "ABROGATES"
            if edge.action == "ABROGATE"
            else "AMENDS"
            if edge.action == "AMEND"
            else "REPLACES"
        ),
        "temporal_source_id": edge.source_instrument,
        "temporal_target_id": edge.target_instrument,
    }


def label_successor_hits(
    ranked: list[tuple[Document, float]],
    edges: list[SupersessionEdge],
    *,
    look_at: int = 8,
) -> list[tuple[Document, float]]:
    """Topic questions ("heures du marché des changes") can retrieve an old circular and the one
    that replaced or amended it. Label the newer circular's hits with that relationship: the
    answer layer then says so and follows the newer text where the two disagree.

    Nothing is added or moved: a declaring page ("la circulaire X est abrogée") does not answer
    a topic question, and many edges only replace an annex or an article, so the older text
    can still be the one that answers.
    """
    hits = {instrument_from_filename(_source_name(doc)) for doc, _score in ranked[:look_at]}
    edge_for: dict[str, SupersessionEdge] = {}
    for edge in edges:
        if edge.action not in {"ABROGATE", "REPLACE", "AMEND"}:
            continue
        if edge.source_instrument in hits and edge.target_instrument in hits:
            edge_for.setdefault(edge.source_instrument, edge)
    if not edge_for:
        return ranked
    labelled = []
    for doc, score in ranked:
        edge = edge_for.get(instrument_from_filename(_source_name(doc)))
        if edge is not None and not doc.metadata.get("temporal_relation"):
            doc = Document(page_content=doc.page_content, metadata={**doc.metadata, **_relation_metadata(edge)})
        labelled.append((doc, score))
    return labelled


def replacement_chain(
    edges: list[SupersessionEdge],
    ranked: list[tuple[Document, float]],
    *,
    look_at: int = 3,
    max_edges: int = 4,
) -> list[SupersessionEdge]:
    """The replacements of the circulars that answer the question, and theirs in turn.

    Only the first `look_at` hits count: those are the texts that answer. For each one, every
    later instrument that replaced, abrogated or amended it, then whatever replaced those
    (2016-01 -> 2021-02 and 2021-03), so the answer can give the latest rule and name each step.
    """
    by_target: dict[str, list[SupersessionEdge]] = defaultdict(list)
    for edge in edges:
        if edge.action in {"ABROGATE", "REPLACE", "AMEND"}:
            by_target[edge.target_instrument].append(edge)
    to_visit = [instrument_from_filename(_source_name(doc)) for doc, _score in ranked[:look_at]]
    chain: list[SupersessionEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    while to_visit and len(chain) < max_edges:
        instrument = to_visit.pop(0)
        for edge in by_target.get(instrument, []):
            pair = (edge.source_instrument, edge.target_instrument)
            if pair in seen_pairs or len(chain) >= max_edges:
                continue
            seen_pairs.add(pair)
            chain.append(edge)
            to_visit.append(edge.source_instrument)
    return chain


def _instrument_ref(instrument: str, reason: str) -> dict:
    kind, year, number = instrument.split(":")
    return {"kind": kind, "year": int(year), "number": int(number), "route_reason": reason}


def predecessor_refs(edges: list[SupersessionEdge], query: str) -> list[dict]:
    """For "avant la circulaire X": the instruments X replaced, abrogated or amended."""
    named = instruments_from_text(query)
    refs, seen = [], set()
    for edge in edges:
        if edge.source_instrument in named and edge.target_instrument not in seen:
            seen.add(edge.target_instrument)
            refs.append(_instrument_ref(edge.target_instrument, "historical_predecessor"))
    return refs


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
            **_relation_metadata(edge),
        },
    )


def _declaring_pages(chain, ranked, page_lookup) -> list[tuple[Document, float]]:
    """The pages that say "la présente circulaire abroge et remplace X", after the ranked hits.

    They prove "X a été remplacée par Y", which the answer states; they are not the answer, so
    they come last and never take a place from a page that answers (conversation.py adds them
    to the evidence)."""
    present = {(_source_name(doc).casefold(), _page_label(doc)) for doc, _score in ranked}
    pages = []
    for edge in chain:
        doc = _edge_document(edge, page_lookup)
        if doc is None:
            continue
        key = (_source_name(doc).casefold(), _page_label(doc))
        if key not in present:
            present.add(key)
            pages.append((doc, 0.0))
    return pages


def _demote_wholly_replaced(
    ranked: list[tuple[Document, float]],
    edges: list[SupersessionEdge],
) -> list[tuple[Document, float]]:
    """A text another circular abrogated or replaced as a whole is no longer the governing rule:
    keep it in the list, below the texts still in force."""
    replaced = {
        edge.target_instrument
        for edge in edges
        # An edge that names an article, an annex or a list leaves the rest of the text in force.
        if edge.action in {"ABROGATE", "REPLACE"} and not edge.target_article
    }
    keep: list[tuple[Document, float]] = []
    demoted: list[tuple[Document, float]] = []
    for doc, score in ranked:
        if instrument_from_filename(_source_name(doc)) in replaced:
            demoted.append((doc, score))
        else:
            keep.append((doc, score))
    return keep + demoted


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
    """Order hits by what is still in force, and pin replacement pages for named instruments.

    1. Retrieved texts that replace or amend another retrieved text are labelled with it.
    2. Wholly replaced texts go below the texts still in force, unless the question asks for
       the time before a circular ("avant la circulaire X"): then the old texts are the answer.
    3. A question that names an instrument ("la circulaire 2016-01 est-elle en vigueur ?") gets
       the page that declares its replacement pinned in front.
    """
    if not edges or not ranked:
        return ranked
    ranked = label_successor_hits(ranked, edges)
    if not is_historical_cutoff_query(query):
        ranked = _demote_wholly_replaced(ranked, edges)
    picked = select_edges(edges, query, limit=2)
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
    # "Is X still in force?" wants the successor first; "Selon X" wants X itself first.
    if not is_temporal_rule_query(query):
        combined = prefer_named_instrument_hits(combined, query_instrument_refs(query))
    return prefer_historical_hits(combined, query)


class SupersessionPinBackend:
    """Wrap a retrieval backend; pin JSONL SUPERSEDES pages when instruments match."""

    def __init__(self, inner, edges: list[SupersessionEdge], page_lookup):
        self.inner = inner
        self.found_edges = edges
        self.page_lookup = page_lookup
        self.expand_pages = getattr(inner, "expand_pages", None)

    @property
    def edges(self) -> list[SupersessionEdge]:
        """The edges found in the PDFs, with the administrator's decisions applied (read per question)."""
        from supersession_review import effective_edges

        return effective_edges(self.found_edges)

    def retrieve(self, query: str, other_queries=()):
        edges = self.edges
        historical = is_historical_cutoff_query(query)
        # "Avant la circulaire X": search inside the texts X replaced and put them first.
        predecessors = predecessor_refs(edges, query) if historical else []
        ranked = list(self.inner.retrieve(query, other_queries=other_queries, instruments=predecessors))
        # A text that answers was later replaced: search inside its successors too, so the rule
        # that applies now competes for the answer.
        chain = [] if historical else replacement_chain(edges, ranked)
        if chain:
            successors = [_instrument_ref(instrument, "successor")
                          for instrument in dict.fromkeys(edge.source_instrument for edge in chain)]
            ranked = list(self.inner.retrieve(query, other_queries=other_queries, instruments=successors))
        ranked = prefer_named_instrument_hits(ranked, predecessors)
        ranked = pin_supersession_edges(ranked, query, edges, page_lookup=self.page_lookup)
        return ranked + _declaring_pages(chain, ranked, self.page_lookup)


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
    # Wrapped even with no edges found: an administrator may add some (supersession_review).
    logger.info("Supersession pin enabled (%s edges from %s)", len(edges), edges_path)
    return SupersessionPinBackend(backend, edges, lookup)
