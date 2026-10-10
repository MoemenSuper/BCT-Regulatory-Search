from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import heapq
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata


from ingestion.registry import _SEARCHABLE_SQL

# Uploads an admin can open: indexed ones, and ones still queued, being read or failed (their
# kept copy exists; the admin page links every upload). Removed and superseded ones cannot.
_OPENABLE_SQL = _SEARCHABLE_SQL[:-1] + ", 'queued', 'processing', 'failed')"
from rag.retrieval_selection import _ARABIC_RANGE
from rag.retrieval_selection import is_arabic_query
from rag.source_metadata import safe_pdf_filename


_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TOKEN_RE = re.compile(rf"[\w{_ARABIC_RANGE}]+", re.UNICODE)


@dataclass(frozen=True)
class SourceDocument:
    filename: str
    path: Path
    metadata: dict


def _safe_basename(filename: str) -> str:
    return safe_pdf_filename(filename, require_exact_basename=True, max_length=240)


def _default_ingestion_db() -> Path | None:
    configured = os.environ.get("BCT_INGESTION_DB")
    if configured:
        return Path(configured).expanduser().resolve()
    asset_root = os.environ.get("BCT_RUNTIME_ASSET_ROOT")
    if not asset_root:
        return None
    return Path(asset_root).expanduser().resolve().parent / "ingestion-data" / "ingestion.sqlite3"


def _candidate_roots() -> list[Path]:
    raw: list[str] = []
    multi = os.environ.get("BCT_SOURCE_DOCUMENT_ROOTS")
    if multi:
        raw.extend(part for part in multi.split(os.pathsep) if part.strip())
    for name in ("BCT_DOCUMENTS_DIR", "BCT_LEGACY_DOCUMENTS_DIR"):
        value = os.environ.get(name)
        if value:
            raw.append(value)
    backend = Path(__file__).resolve().parent.parent  # this file lives in backend/extras/
    raw.extend([str(backend / "documents"), str(backend.parent / "documents")])

    roots: list[Path] = []
    seen: set[Path] = set()
    for value in raw:
        path = Path(value).expanduser().resolve()
        if path in seen or not path.exists() or not path.is_dir():
            continue
        seen.add(path)
        roots.append(path)
    return roots


class SourceDocumentResolver:
    """Resolve chat source basenames to trusted local PDFs.

    Resolution order is intentionally conservative:
    1. the ingestion ledger's latest ready immutable copy;
    2. a unique basename match under configured document roots.

    The API never accepts arbitrary paths from the browser.
    """

    def __init__(self) -> None:
        self.ingestion_db = _default_ingestion_db()
        self.roots = _candidate_roots()
        self._scan_index: dict[str, list[Path]] | None = None

    def refresh(self) -> None:
        self.roots = _candidate_roots()
        self._scan_index = None

    def _resolve_from_ingestion(self, filename: str) -> Path | None:
        database = self.ingestion_db
        if database is None or not database.exists():
            return None
        try:
            with sqlite3.connect(database, timeout=5) as connection:
                row = connection.execute(
                    """
                    SELECT stored_path
                    FROM ingestion_documents
                    WHERE status IN """ + _OPENABLE_SQL + """ AND lower(original_filename)=lower(?)
                    ORDER BY activated_at DESC, created_at DESC
                    LIMIT 1
                    """,
                    (filename,),
                ).fetchone()
        except sqlite3.Error:
            return None
        if not row or not row[0]:
            return None
        path = Path(row[0]).expanduser().resolve()
        return path if path.is_file() else None

    def _build_scan_index(self) -> dict[str, list[Path]]:
        index: dict[str, list[Path]] = {}
        for root in self.roots:
            try:
                paths = root.rglob("*.pdf")
                for path in paths:
                    if not path.is_file():
                        continue
                    try:
                        resolved = path.resolve()
                        resolved.relative_to(root)
                    except (OSError, ValueError):
                        continue
                    index.setdefault(path.name.casefold(), []).append(resolved)
            except OSError:
                continue
        return index

    def resolve(self, filename: str) -> SourceDocument:
        safe_name = _safe_basename(filename)
        metadata: dict = {}

        path = self._resolve_from_ingestion(safe_name)
        if path is None:
            if self._scan_index is None:
                self._scan_index = self._build_scan_index()
            matches = self._scan_index.get(safe_name.casefold(), [])
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                # Prefer the newest immutable ingestion copy when multiple versions
                # of the same basename exist and no registry was available.
                path = max(matches, key=lambda item: item.stat().st_mtime_ns)

        if path is None or not path.is_file():
            raise FileNotFoundError(safe_name)
        return SourceDocument(filename=safe_name, path=path, metadata=metadata)


def source_info(document: SourceDocument) -> dict:
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError("Source viewing requires PyMuPDF") from error

    with pymupdf.open(document.path) as pdf:
        pages = int(pdf.page_count)
    return {
        "filename": document.filename,
        "pages": pages,
        "title": document.metadata.get("title") or "",
    }


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("ـ", "")
    value = _ARABIC_DIACRITICS.sub("", value)
    value = value.casefold()
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"[\u2010-\u2015]", "-", value)
    return " ".join(value.split())


def _tokens(value: str) -> list[str]:
    return [_normalize_text(token) for token in _TOKEN_RE.findall(value) if _normalize_text(token)]


def _word_highlight_rects(page, quote: str, deadline: float | None = None):
    """Fallback fuzzy locator using real PDF word geometry.

    This never changes evidence text; it is only a visual locator when PDF.js/
    MuPDF's phrase search cannot match whitespace, punctuation or Arabic text order.
    """
    quote_tokens = _tokens(quote)
    if not quote_tokens:
        return []
    raw_words = page.get_text("words", sort=True)
    if is_arabic_query(quote):
        # sort=True orders each line left-to-right (visual order); Arabic reads
        # right-to-left, so the logical word sequence is reversed within a line.
        raw_words.sort(key=lambda item: (round(item[3], 1), -item[0]))
    words = []
    for item in raw_words:
        if len(item) < 8:
            continue
        token = _normalize_text(str(item[4]))
        token_parts = _tokens(token)
        if not token_parts:
            continue
        # PDF words occasionally contain punctuation-separated tokens. Reuse the
        # same geometry for each token so subsequence matching remains simple.
        for part in token_parts:
            words.append((part, item))
    if not words:
        return []

    page_tokens = [item[0] for item in words]
    # The quote comes from the request and the fuzzy pass costs about the square of its
    # length per page window: locate a long quote piece by piece (pieces of at least half the
    # chunk size, so a short tail cannot land on an unrelated repeat) and stop fuzzy
    # matching once the time budget (shared by every quote segment) is spent. Exact
    # matches stay cheap and always run.
    pieces = -(-len(quote_tokens) // _FUZZY_CHUNK_TOKENS)
    size = -(-len(quote_tokens) // pieces)
    if deadline is None:
        deadline = time.monotonic() + _FUZZY_BUDGET_SECONDS
    matched = []
    for start in range(0, len(quote_tokens), size):
        span = _match_span(quote_tokens[start : start + size], page_tokens, deadline)
        if span:
            matched.extend(item[1] for item in words[span[0] : span[1]])
    if not matched:
        return []

    # Merge word boxes line-by-line to make the overlay look like a normal PDF
    # search highlight rather than dozens of small boxes.
    # Group by baseline, not (block_no, line_no): some PDFs emit one block per
    # word, which would leave one rect per word and hit the 20-rect render cap.
    lines: dict[float, list] = {}
    for item in matched:
        lines.setdefault(round(float(item[3]), 1), []).append(item)

    import pymupdf

    rects = []
    for line_words in lines.values():
        x0 = min(float(item[0]) for item in line_words)
        y0 = min(float(item[1]) for item in line_words)
        x1 = max(float(item[2]) for item in line_words)
        y1 = max(float(item[3]) for item in line_words)
        rects.append(pymupdf.Rect(x0 - 1.0, y0 - 0.5, x1 + 1.0, y1 + 0.5))
    return rects


_FUZZY_CHUNK_TOKENS = 30
_FUZZY_BUDGET_SECONDS = 2.0


def _match_span(quote_tokens, page_tokens, deadline):
    """(start, end) of the page tokens matching the quote tokens, or None."""
    qn = len(quote_tokens)

    # Exact normalized token subsequence first.
    start_index = None
    end_index = None
    for start in range(0, max(0, len(page_tokens) - qn + 1)):
        if page_tokens[start : start + qn] == quote_tokens:
            start_index, end_index = start, start + qn
            break

    # Conservative fuzzy fallback. It is better to show no yellow overlay than
    # highlight the wrong regulatory sentence.
    if start_index is None:
        best = (0.0, 0, 0)
        min_len = max(1, qn - min(4, qn // 4))
        max_len = min(len(page_tokens), qn + min(4, qn // 4))
        quote_joined = " ".join(quote_tokens)
        # Rank every window on cheap token-level similarity, then confirm only the
        # best few with the character-level ratio that decides the threshold.
        # Character ratios on every window took 5-50 s per page for long quotes.
        scored = []
        for window in range(min_len, max_len + 1):
            if time.monotonic() > deadline:
                return None
            scored.extend(
                (
                    SequenceMatcher(None, quote_tokens, page_tokens[start : start + window], autojunk=False).ratio(),
                    start,
                    start + window,
                )
                for start in range(0, len(page_tokens) - window + 1)
            )
        for _score, start, end in heapq.nlargest(20, scored):
            candidate = " ".join(page_tokens[start:end])
            ratio = SequenceMatcher(None, quote_joined, candidate, autojunk=False).ratio()
            if ratio > best[0]:
                best = (ratio, start, end)
        threshold = 0.78 if qn <= 5 else 0.72
        if best[0] < threshold:
            return None
        start_index, end_index = best[1], best[2]
    return start_index, end_index


def _quote_segments(quote: str) -> list[str]:
    # answer_contract joins multiple exact quotes from the same physical page with
    # an ellipsis separator. Locate each original quote independently so the PDF
    # viewer highlights every supported passage instead of trying to search for a
    # synthetic combined string that never existed in the source PDF.
    raw = (quote or "").strip()
    if not raw:
        return []
    pieces = re.split(r"\s*(?:\n\s*)?…(?:\s*\n)?\s*", raw)
    return [" ".join(piece.split()).strip() for piece in pieces if piece.strip()]


def locate_quote(page, quote: str):
    rects = []
    deadline = time.monotonic() + _FUZZY_BUDGET_SECONDS
    for cleaned in _quote_segments(quote):
        # PyMuPDF recommends quads=True for text-marker annotations because it
        # preserves orientation information for rotated / non-horizontal text.
        try:
            direct = list(page.search_for(cleaned, quads=True))
        except Exception:
            direct = []
        if direct:
            rects.extend(direct)
            continue
        rects.extend(_word_highlight_rects(page, cleaned, deadline))
    return rects


def render_page_png(
    document: SourceDocument,
    *,
    page_number: int,
    quote: str = "",
    scale: float = 1.8,
) -> tuple[bytes, dict[str, str]]:
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError("Source viewing requires PyMuPDF") from error

    scale = max(0.75, min(float(scale), 3.0))
    with pymupdf.open(document.path) as pdf:
        if page_number < 1 or page_number > pdf.page_count:
            raise IndexError(page_number)
        page = pdf.load_page(page_number - 1)
        rects = locate_quote(page, quote[:2000]) if quote else []
        highlight_method = "native" if rects else "none"
        annotations = []
        for rect in rects[:20]:
            try:
                annotation = page.add_highlight_annot(rect)
                annotation.set_colors(stroke=(1.0, 0.78, 0.0))
                annotation.set_opacity(0.42)
                annotation.update()
                annotations.append(annotation)
            except Exception:
                continue
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False, annots=True)
        payload = pixmap.tobytes("png")
        headers = {
            "X-BCT-Page": str(page_number),
            "X-BCT-Total-Pages": str(pdf.page_count),
            "X-BCT-Highlight-Matches": str(len(annotations)),
            "X-BCT-Highlight-Method": highlight_method,
            "Cache-Control": "no-store",
        }
        return payload, headers
