"""JSONL SUPERSEDES edges: extract, load, merge for ingest.

Pins successor declaring pages when:
  1. the query names an edged instrument (force / “still in force?” path), or
  2. classic top hits include an instrument that is itself a SUPERSEDES target
     (topical currentness — e.g. work hours lands on an old circular).

On PDF ingest, operative amendment language is extracted from the new pages and
merged into supersession_edges.jsonl inside the staged asset version (before
activate). Vu / "telle que modifiée par" citations are not SUPERSEDES edges.

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

from retrieval_selection import (
    parse_source_identity,
    prefer_historical_hits,
    prefer_named_instrument_hits,
    prefer_regime_hits,
    query_instrument_refs,
    query_regulatory_regime,
    _doc_matches_regime,
)
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
    r"(abroge\s+et\s+remplace|annule\s+et\s+remplace|annule\s+et\s+substitue|"
    r"abroge|abrogées?|abrogés?|remplace|remplacent|remplacée?s?|"
    r"modifie|modifiées?|modifiés?|"
    r"تلغي|تعوض|يلغى|يعوض|تلغى|تستبدل)",
    re.I,
)
TARGET_CIR = re.compile(
    r"(?:circulaire|note|منشور)"
    r"(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ'’-]{1,30}){0,8}"
    r"\s*(?:n[°o.]?\s*|عدد\s*)?(?P<y>\d{2,4})\s*[-–/]\s*(?P<n>\d{1,3})",
    re.I,
)
# Historical citation: "X telle que modifiée par Y" — Y amended X; the citing PDF is not the actor.
CITATION_AS_MODIFIED_BY = re.compile(
    r"tel(?:le|s|les)?\s+que\s+"
    r"(?:modifi[ée]e?s?|compl[ée]t[ée]e?s?)"
    r"(?:\s+et\s+(?:modifi[ée]e?s?|compl[ée]t[ée]e?s?))?\s+par",
    re.I,
)
# Operative body starts after the enacting formula, not mid-Vu "l'article 42".
DECIDE_START = re.compile(r"\bD[ée]cide\s*:", re.I)
# Allow PDF line breaks between "Article" and "premier" / "1".
ARTICLE_LINE_START = re.compile(
    r"(?mi)^\s*Articles?(?:\s|\n)+(?:premier|premi[eè]re|unique|\d+)\b",
)
PRESENT_CIRCULAR_ACTOR = re.compile(
    r"\bla\s+pr[ée]sente\s+circulaire\s+"
    r"(?:annule\s+et\s+remplace|abroge(?:\s+et\s+remplace)?|remplace|modifie)",
    re.I,
)


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


def edge_quote_is_operative(edge: SupersessionEdge) -> bool:
    """Reject stored Vu/citation-history rows that are not real amend/replace acts."""
    quote = edge.quote or ""
    if not quote.strip():
        return False
    if CITATION_AS_MODIFIED_BY.search(quote) and not PRESENT_CIRCULAR_ACTOR.search(quote):
        return False
    if edge.action in {"ABROGATE", "REPLACE"}:
        if not re.search(
            r"abroge|abrog|remplac|annule\s+et\s+substitue|تلغي|تعوض|يلغى|يعوض",
            quote,
            re.I,
        ):
            return False
    return True


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
                edge = SupersessionEdge(
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
            except (KeyError, TypeError, ValueError):
                continue
            if edge_quote_is_operative(edge):
                edges.append(edge)
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
    if re.search(r"remplac|substit|تعوض|يعوض|تستبدل", snippet, re.I):
        return "REPLACE"
    if re.search(r"abroge|abrog|تلغي|يلغى|تلغى|إلغاء|الغاء", snippet, re.I):
        return "ABROGATE"
    if re.search(r"modifi|amend|يعدل|تعدل|ينقح|تنقح|تعديل", snippet, re.I):
        return "AMEND"
    return "ABROGATE"


def _operative_body(text: str) -> tuple[str, int]:
    """Return (operative text, offset). Prefer post-Décide; else line-start Article."""
    raw = text or ""
    decide = DECIDE_START.search(raw)
    if decide:
        return raw[decide.start() :], decide.start()
    article = ARTICLE_LINE_START.search(raw)
    if article:
        return raw[article.start() :], article.start()
    return raw, 0


def _action_is_citation_history(text: str, action_start: int, action_end: int) -> bool:
    """True for 'tel(le) que modifié(e) par …' — citation of prior amendments, not our act."""
    window = text[max(0, action_start - 100) : min(len(text), action_end + 20)]
    return bool(CITATION_AS_MODIFIED_BY.search(window))


def _exact_quote_span(text: str, start: int, end: int, *, limit: int = 700) -> str | None:
    """Contiguous evidence span from the page; must be recoverable from page text."""
    raw = (text or "")[start:end]
    quote = " ".join(raw.split())
    if len(quote) < 40:
        return None
    quote = quote[:limit]
    collapsed_page = " ".join((text or "").split())
    if quote not in collapsed_page:
        return None
    return quote


def _targets_acted_upon(snippet: str, source: str, *, action_verb: str) -> list[str]:
    """Instruments the current circular acts on — not prior amenders in citation grammar."""
    scrubbed = CITATION_AS_MODIFIED_BY.sub(" ", snippet)
    present = PRESENT_CIRCULAR_ACTOR.search(scrubbed)
    if present:
        search_regions = [scrubbed[present.end() :]]
    else:
        rel = None
        for cand in ACTION_RE.finditer(scrubbed):
            if cand.group(0).casefold() == action_verb.casefold():
                rel = cand
        if rel is None:
            rel = ACTION_RE.search(scrubbed)
        search_regions = [scrubbed[: rel.start()], scrubbed] if rel else [scrubbed]
    targets: list[str] = []
    for region in search_regions:
        for tgt in TARGET_CIR.finditer(region):
            kind = "note" if "note" in tgt.group(0).casefold() else "cir"
            instrument = _instrument(kind, int(tgt.group("y")), int(tgt.group("n")))
            if instrument != source and instrument not in targets:
                targets.append(instrument)
        if targets:
            return targets
    return targets


def _target_article(snippet: str, source: str) -> str | None:
    for tgt in TARGET_CIR.finditer(snippet):
        kind = "note" if "note" in tgt.group(0).casefold() else "cir"
        instrument = _instrument(kind, int(tgt.group("y")), int(tgt.group("n")))
        if instrument == source:
            continue
        window = snippet[max(0, tgt.start() - 40) : tgt.end() + 40]
        near = ARTICLE.search(window)
        if near:
            return near.group("a")
    before_cir = re.search(
        r"(?:articles?|art\.?)\s*(?P<a>\d+)\s+de\s+la\s+circulaire",
        snippet,
        re.I,
    )
    return before_cir.group("a") if before_cir else None


def extract_edges_from_page_text(
    *,
    filename: str,
    page_number: int,
    text: str,
) -> list[SupersessionEdge]:
    """Direction-aware SUPERSEDES edges from operative amendment language only.

    Vu / preamble citations such as "circulaire 94-14 … telle que modifiée par …
    notamment la circulaire 2025-13" are not edges from the citing PDF. Source is
    always the ingesting instrument; targets are instruments it amends/replaces/
    abrogates. Every edge carries an exact evidence span from the page text.
    """
    source = instrument_from_filename(filename)
    if source is None:
        return []
    full = (text or "").strip()
    if len(full) < 40 or not ACTION_RE.search(full):
        return []
    body, _body_offset = _operative_body(full)
    if len(body) < 40 or not ACTION_RE.search(body):
        return []

    edges: list[SupersessionEdge] = []
    for match in ACTION_RE.finditer(body):
        if _action_is_citation_history(body, match.start(), match.end()):
            continue
        start = max(0, match.start() - 220)
        end = min(len(body), match.end() + 280)
        snippet = " ".join(body[start:end].split())
        if len(snippet) < 40:
            continue
        targets = _targets_acted_upon(snippet, source, action_verb=match.group(0))
        if not targets:
            continue
        # Classify from the matched verb (+ short tail), not citation noise in lookbehind.
        verb_tail = body[match.start() : min(len(body), match.end() + 40)]
        action = _classify_action(verb_tail)
        art = _target_article(snippet, source)
        quote = _exact_quote_span(body, start, end)
        if not quote:
            continue
        for target in targets[:3]:
            edges.append(
                SupersessionEdge(
                    source_instrument=source,
                    source_file=Path(filename).name,
                    source_page=int(page_number),
                    action=action,
                    target_instrument=target,
                    target_article=art,
                    quote=quote,
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


def _pdf_page_texts(pdf_path: Path) -> list[tuple[int, str]]:
    """Best-effort page text for offline rebuild (PyMuPDF, else pypdf)."""
    try:
        import pymupdf
    except ImportError:
        pymupdf = None
    if pymupdf is not None:
        doc = pymupdf.open(pdf_path)
        try:
            return [
                (i + 1, (doc.load_page(i).get_text("text") or ""))
                for i in range(doc.page_count)
            ]
        finally:
            doc.close()
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        out.append((i + 1, page.extract_text() or ""))
    return out


def rebuild_supersession_edges_from_documents(
    documents_dir: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """Scan a PDF corpus and write a fresh supersession_edges.jsonl.

    Fail-closed: pages without explicit operative amendment language contribute
    no edges. Clears the in-process pin cache so the next resolve/load sees the
    new file.
    """
    root = Path(documents_dir)
    output = Path(output_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {root}")

    class _Page:
        def __init__(self, page_number: int, raw_text: str):
            self.page_number = page_number
            self.raw_text = raw_text
            self.metadata = {}

    edges: list[SupersessionEdge] = []
    pdf_count = 0
    for pdf in sorted(root.rglob("*.pdf")):
        if instrument_from_filename(pdf.name) is None:
            continue
        pdf_count += 1
        try:
            pages = [_Page(n, text) for n, text in _pdf_page_texts(pdf)]
        except Exception as error:
            logger.warning("Supersession rebuild skipped %s: %s", pdf.name, error)
            continue
        edges.extend(extract_edges_from_pages(pdf.name, pages))
    merged = merge_edge_lists(edges)
    write_edges(output, merged)
    from supersession_pin import clear_supersession_cache

    clear_supersession_cache()
    return {
        "pdfs": pdf_count,
        "edges": len(merged),
        "output": str(output.resolve()),
    }


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
    from supersession_pin import clear_supersession_cache

    clear_supersession_cache()
    return {
        "prior": len(prior),
        "kept": len(kept),
        "from_pdf": len(extracted),
        "total": len(merged),
    }


