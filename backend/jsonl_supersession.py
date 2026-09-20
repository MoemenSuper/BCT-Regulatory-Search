"""JSONL SUPERSEDES pin — force and topical currentness without a graph DB.

Pins successor declaring pages when:
  1. the query names an edged instrument (force / “still in force?” path), or
  2. classic top hits include an instrument that is itself a SUPERSEDES target
     (topical currentness — e.g. work hours lands on an old circular).

On PDF ingest, amendment language is extracted from the new pages and merged
into supersession_edges.jsonl inside the staged asset version (before activate).

Edges path (first hit wins):
  1. BCT_SUPERSESSION_EDGES
  2. <active-version>/supersession_edges.jsonl
  3. <asset-root>/supersession_edges.jsonl (legacy seed)
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from source_metadata import normalize_page

logger = logging.getLogger(__name__)

INSTR = re.compile(
    r"(?P<kind>circulaire|note|cir|منشور)\s*(?:n[°o.]?\s*|عدد\s*)?(?P<y>\d{2,4})\s*[-–/]\s*(?P<n>\d{1,3})",
    re.I,
)
# User-like: "circ 2018 07", "circulaire 2018 07"
MESSY_INSTR = re.compile(
    r"(?P<kind>circulaire|note|circ|cir|منشور)\s*(?:n[°o.]?\s*|عدد\s*)?(?P<y>\d{2,4})\s+(?P<n>\d{1,3})\b",
    re.I,
)
BARE = re.compile(r"\b(?P<y>\d{2,4})\s*[-–/]\s*(?P<n>\d{1,3})\b")
# User-like bare: "2018 07" only when force-ish context is nearby (handled in instruments_from_text)
BARE_SPACE = re.compile(r"\b(?P<y>\d{2,4})\s+(?P<n>\d{1,3})\b")
ARTICLE = re.compile(r"\b(?:articles?|art\.?)\s*(?P<a>\d+)\b", re.I)
SRC_FROM_FILE = re.compile(r"(?P<kind>Cir|Note|CB)_(?P<y>\d{4})_(?P<n>\d+)", re.I)
FORCEISH = re.compile(
    r"\b(?:valable|vigueur|applicable|abroge|remplac|modifi|encore|toujours|tjrs|"
    r"substit|annul|remplacé|qui\s+a|toujours|en\s+force)\b|"
    r"(?:تلغي|تعوض|يلغى|يعوض|ساري|نافذ)",
    re.I,
)
ACTION_RE = re.compile(
    r"(abroge\s+et\s+remplace|annule\s+et\s+substitue|"
    r"abroge|abrogées?|abrogés?|remplace|remplacent|remplacées?|"
    r"modifie|modifiées?|modifiés?|"
    r"تلغي|تعوض|يلغى|يعوض|تلغى|تستبدل)",
    re.I,
)
TARGET_CIR = re.compile(
    r"(?:circulaire|note|منشور)\s*(?:n[°o.]?\s*|عدد\s*)?(?P<y>\d{2,4})\s*[-–/]\s*(?P<n>\d{1,3})",
    re.I,
)
# Skip Vu/preamble citation noise without an operative verb nearby
VU_ONLY = re.compile(r"^\s*(?:vu|vu\s+la|نظرا|بناء\s+على)\b", re.I)


@dataclass(frozen=True)
class SupersessionEdge:
    source_instrument: str
    source_file: str
    source_page: int
    action: str
    target_instrument: str
    target_article: str | None
    quote: str


def _norm_year(raw: int | str) -> int:
    year = int(raw)
    if year < 100:
        return 1900 + year if year >= 50 else 2000 + year
    return year


def _instrument(kind: str, year: int, number: int) -> str:
    k = kind.casefold()
    if k in {"cir", "cb", "circulaire", "circular", "circ", "منشور"}:
        k = "cir"
    elif "note" in k:
        k = "note"
    else:
        k = "cir"
    return f"{k}:{_norm_year(year)}:{int(number)}"


def instrument_from_filename(name: str) -> str | None:
    match = SRC_FROM_FILE.search(Path(name).name)
    if not match:
        return None
    return _instrument(match.group("kind"), int(match.group("y")), int(match.group("n")))


def instruments_from_text(text: str) -> set[str]:
    """Parse instrument ids from precise and imperfect user questions."""
    found: set[str] = set()
    raw = text or ""
    for match in INSTR.finditer(raw):
        kind = "note" if "note" in match.group(0).casefold() else "cir"
        found.add(_instrument(kind, int(match.group("y")), int(match.group("n"))))
    for match in MESSY_INSTR.finditer(raw):
        kind = "note" if "note" in match.group("kind").casefold() else "cir"
        found.add(_instrument(kind, int(match.group("y")), int(match.group("n"))))
    for match in BARE.finditer(raw):
        found.add(_instrument("cir", int(match.group("y")), int(match.group("n"))))
    # "2018 07" only in force-ish questions — avoids grabbing random amounts
    if FORCEISH.search(raw):
        for match in BARE_SPACE.finditer(raw):
            year = int(match.group("y"))
            number = int(match.group("n"))
            if 1990 <= year <= 2035 and 1 <= number <= 99:
                found.add(_instrument("cir", year, number))
    return found


def resolve_edges_path(runtime_assets: Path | None = None) -> Path | None:
    env = (os.environ.get("BCT_SUPERSESSION_EDGES") or "").strip()
    if env:
        path = Path(env)
        return path if path.is_file() else None
    if runtime_assets is not None:
        root = Path(runtime_assets)
        candidate = root / "supersession_edges.jsonl"
        if candidate.is_file():
            return candidate
        # Legacy seed at asset root when version dir has no edges yet
        if root.parent.name == "versions":
            legacy = root.parent.parent / "supersession_edges.jsonl"
            if legacy.is_file():
                return legacy
    return None


def edge_key(edge: SupersessionEdge) -> tuple:
    return (
        edge.source_instrument,
        edge.target_instrument,
        edge.target_article,
        edge.action,
        Path(edge.source_file).name.casefold(),
        int(edge.source_page),
    )


def load_edges(path: Path) -> list[SupersessionEdge]:
    edges: list[SupersessionEdge] = []
    if not path.is_file():
        return edges
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                edges.append(
                    SupersessionEdge(
                        source_instrument=str(row["source_instrument"]),
                        source_file=str(row["source_file"]),
                        source_page=int(row["source_page"]),
                        action=str(row["action"]),
                        target_instrument=str(row["target_instrument"]),
                        target_article=(
                            str(row["target_article"])
                            if row.get("target_article") not in (None, "")
                            else None
                        ),
                        quote=str(row.get("quote") or ""),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return edges


def write_edges(path: Path, edges: Iterable[SupersessionEdge]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(asdict(edge), ensure_ascii=False, sort_keys=True)
        for edge in edges
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def merge_edge_lists(*groups: Iterable[SupersessionEdge]) -> list[SupersessionEdge]:
    by_key: dict[tuple, SupersessionEdge] = {}
    for group in groups:
        for edge in group:
            by_key[edge_key(edge)] = edge
    return list(by_key.values())


def _classify_action(snippet: str) -> str:
    low = snippet.casefold()
    if re.search(r"remplac|substit|تعوض|يعوض|تستبدل", snippet, re.I):
        return "REPLACE"
    if re.search(r"abroge|abrog|تلغي|يلغى|تلغى|إلغاء|الغاء", snippet, re.I):
        return "ABROGATE"
    if re.search(r"modifi|amend|يعدل|تعدل|ينقح|تنقح|تعديل", snippet, re.I):
        return "AMEND"
    return "ABROGATE"


def extract_edges_from_page_text(
    *,
    filename: str,
    page_number: int,
    text: str,
) -> list[SupersessionEdge]:
    """Deterministic SUPERSEDES edges from one page of amendment language."""
    source = instrument_from_filename(filename)
    if source is None:
        return []
    body = (text or "").strip()
    if len(body) < 40 or not ACTION_RE.search(body):
        return []
    edges: list[SupersessionEdge] = []
    for match in ACTION_RE.finditer(body):
        start = max(0, match.start() - 100)
        end = min(len(body), match.end() + 280)
        snippet = " ".join(body[start:end].split())
        if len(snippet) < 40:
            continue
        if VU_ONLY.search(snippet) and not re.search(
            r"abroge|remplac|substit|تلغي|تعوض", snippet, re.I
        ):
            continue
        targets: list[str] = []
        for tgt in TARGET_CIR.finditer(snippet):
            kind = "note" if "note" in tgt.group(0).casefold() else "cir"
            instrument = _instrument(kind, int(tgt.group("y")), int(tgt.group("n")))
            if instrument != source and instrument not in targets:
                targets.append(instrument)
        if not targets:
            continue
        action = _classify_action(snippet)
        # Prefer article attached to the target mention, not "Article N" of the amending PDF.
        art = None
        for tgt in TARGET_CIR.finditer(snippet):
            kind = "note" if "note" in tgt.group(0).casefold() else "cir"
            instrument = _instrument(kind, int(tgt.group("y")), int(tgt.group("n")))
            if instrument == source:
                continue
            window = snippet[max(0, tgt.start() - 40) : tgt.end() + 40]
            near = ARTICLE.search(window)
            if near:
                art = near.group("a")
                break
        if art is None:
            # Fall back: article immediately before "de la circulaire"
            before_cir = re.search(
                r"(?:articles?|art\.?)\s*(?P<a>\d+)\s+de\s+la\s+circulaire",
                snippet,
                re.I,
            )
            if before_cir:
                art = before_cir.group("a")
        for target in targets[:3]:
            edges.append(
                SupersessionEdge(
                    source_instrument=source,
                    source_file=Path(filename).name,
                    source_page=int(page_number),
                    action=action,
                    target_instrument=target,
                    target_article=art,
                    quote=snippet[:700],
                )
            )
    return merge_edge_lists(edges)


def extract_edges_from_pages(filename: str, pages: Iterable) -> list[SupersessionEdge]:
    """Extract edges from StructuredDocument pages (or objects with page_number/raw_text)."""
    found: list[SupersessionEdge] = []
    for page in pages:
        page_number = int(getattr(page, "page_number", 0) or 0)
        text = str(getattr(page, "raw_text", "") or "")
        meta = getattr(page, "metadata", None) or {}
        if meta.get("visual_complete") is True:
            visual = str(meta.get("visual_text") or "").strip()
            if visual:
                text = visual
        if page_number < 1 or not text:
            continue
        found.extend(
            extract_edges_from_page_text(
                filename=filename, page_number=page_number, text=text
            )
        )
    return merge_edge_lists(found)


def load_prior_edges(active_before: Path, asset_root: Path | None = None) -> list[SupersessionEdge]:
    """Prior edges from previous active version, else legacy asset-root seed."""
    version_path = Path(active_before) / "supersession_edges.jsonl"
    if version_path.is_file():
        return load_edges(version_path)
    root = Path(asset_root) if asset_root is not None else (
        Path(active_before).parent.parent
        if Path(active_before).parent.name == "versions"
        else Path(active_before)
    )
    legacy = root / "supersession_edges.jsonl"
    if legacy.is_file():
        return load_edges(legacy)
    return []


def merge_supersession_edges_for_ingest(
    *,
    active_before: Path,
    staged_version: Path,
    filename: str,
    pages: Iterable,
    asset_root: Path | None = None,
) -> dict[str, int]:
    """Drop old edges for this PDF, add freshly extracted ones, write staged file."""
    prior = load_prior_edges(active_before, asset_root=asset_root)
    basename = Path(filename).name.casefold()
    kept = [
        edge
        for edge in prior
        if Path(edge.source_file).name.casefold() != basename
    ]
    extracted = extract_edges_from_pages(filename, pages)
    merged = merge_edge_lists(kept, extracted)
    write_edges(Path(staged_version) / "supersession_edges.jsonl", merged)
    clear_supersession_cache()
    return {
        "prior": len(prior),
        "kept": len(kept),
        "from_pdf": len(extracted),
        "total": len(merged),
    }


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
    """Push fully abrogated/replaced instruments below live/successor hits.

    Only demotes when we actually pinned a successor for that target — avoids
    burying old pages when we have no amending evidence in the pack.
    """
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
    return keep + demoted


def _prefer_successor_instrument_hits(
    ranked: list[tuple[Document, float]],
    pinned_sources: set[str],
) -> list[tuple[Document, float]]:
    """Keep already-retrieved pages from the successor circular near the front.

    Declaring pages alone often only state the abrogation; topical answers need
    the successor's substantive rule pages when classic retrieve already found them.
    """
    if not ranked or not pinned_sources:
        return ranked
    front: list[tuple[Document, float]] = []
    rest: list[tuple[Document, float]] = []
    for doc, score in ranked:
        instrument = instrument_from_filename(_source_name(doc))
        if instrument and instrument in pinned_sources:
            front.append((doc, score if score >= 8000.0 else max(float(score), 8500.0)))
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
    for edge in select_edges(edges, query, limit=2) + select_edges_from_hits(
        edges, ranked, limit=3
    ):
        key = (edge.source_instrument, edge.target_instrument, edge.target_article, edge.source_page)
        if key in seen_edge:
            continue
        seen_edge.add(key)
        picked.append(edge)
        if len(picked) >= 4:
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
    return _demote_fully_superseded(combined, edges, pinned_sources)


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
