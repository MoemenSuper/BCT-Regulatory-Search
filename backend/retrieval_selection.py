"""Document identity and page diversity shared by retrieval backends."""
import re
import unicodedata
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

from source_metadata import normalize_page

_ARABIC_RANGE = r"\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
ARABIC = re.compile(f"[{_ARABIC_RANGE}]")

_SOURCE_ID = re.compile(
    r"(?i)^(?P<kind>cir|note|cb|nb|ci)[_ -]?(?P<year>\d{4})"
    r"[_ -]?(?P<number>\d+).*?[_ -]?(?P<language>fr|ar)\.pdf$"
)


_FILENAME_QUERY_ID = re.compile(
    r"(?i)\b(?P<kind>cir|note|cb|nb|ci)[_ -](?P<year>\d{4})"
    r"[_ -](?P<number>\d+)(?:[_ -](?:fr|ar))?(?:\.pdf)?\b"
)


_FRENCH_QUERY_ID = re.compile(
    r"(?i)\b(?P<kind>circulaire|circular|note)\s+(?:BCT\s+)?"
    r"(?:(?:n(?:[°ºo])?\.?|numero|number)\s*)?"
    r"(?:(?P<year>\d{4})\s*[-/]\s*(?P<number>\d+)"
    r"|(?P<number_first>\d+)\s*(?:de|/)\s*(?P<year_last>\d{4}))"
)


_ARABIC_QUERY_ID = re.compile(
    r"(?P<kind>المنشور|منشور|المذكرة|مذكرة)\s*"
    r"(?:عدد|رقم)\s*(?P<number>\d+)\s*"
    r"(?:لسنة|سنة|لعام|عام)\s*(?P<year>\d{4})"
)


_FRENCH_YEAR = re.compile(
    r"(?i)\b(?:en|type|publiee?\s+en|emise?\s+en|adoptee?\s+en)"
    r"\s+(?P<year>(?:19|20)\d{2})\b"
)


_ARABIC_YEAR = re.compile(
    r"(?:لسنة|سنة|لعام|عام)\s*(?P<year>(?:19|20)\d{2})(?!\d)"
)


_FUTURE = re.compile(
    r"(?i)\b(?:futur|future|prochain|prochaine|sera\s+publie)\b|"
    r"سيصدر|ستصدر|القادم|القادمة"
)


def _ascii_fold(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )


def parse_source_identity(source: str) -> dict[str, Any] | None:
    filename = str(source).replace("\\", "/").rsplit("/", 1)[-1]
    match = _SOURCE_ID.match(filename)
    if not match:
        return None
    values = match.groupdict()
    return {
        "kind": values["kind"].casefold(),
        "year": int(values["year"]),
        "number": int(values["number"]),
        "language": values["language"].casefold(),
    }


# Shared instrument-kind aliases for filename parsing.
_KIND_ALIASES = {
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


def _query_identity(
    *, kind: str | None, year: str | int, number: str | int | None, reason: str
) -> dict[str, Any]:
    normalized_kind = kind.casefold() if kind is not None else None
    if normalized_kind is not None:
        normalized_kind = _KIND_ALIASES.get(normalized_kind, normalized_kind)
    return {
        "kind": normalized_kind,
        "year": int(year),
        "number": int(number) if number is not None else None,
        "route_reason": reason,
    }


def parse_query_identity(query: str) -> dict[str, Any] | None:
    """Return only identity signals observable in the runtime query text."""
    query = query.translate(str.maketrans({character: "-" for character in "‐‑‒–—−"}))
    filename_match = _FILENAME_QUERY_ID.search(query)
    if filename_match:
        values = filename_match.groupdict()
        return _query_identity(
            kind=values["kind"],
            year=values["year"],
            number=values["number"],
            reason="explicit_instrument_identity",
        )

    folded = _ascii_fold(query)
    french_match = _FRENCH_QUERY_ID.search(folded)
    if french_match:
        values = french_match.groupdict()
        return _query_identity(
            kind=values["kind"],
            year=values["year"] or values["year_last"],
            number=values["number"] or values["number_first"],
            reason="explicit_instrument_identity",
        )

    arabic_match = _ARABIC_QUERY_ID.search(query)
    if arabic_match:
        values = arabic_match.groupdict()
        return _query_identity(
            kind=values["kind"],
            year=values["year"],
            number=values["number"],
            reason="explicit_instrument_identity",
        )

    if _FUTURE.search(folded) or _FUTURE.search(query):
        return None
    years = {
        int(match.group("year"))
        for pattern, text in ((_FRENCH_YEAR, folded), (_ARABIC_YEAR, query))
        for match in pattern.finditer(text)
    }
    if len(years) != 1:
        return None
    return _query_identity(
        kind=None,
        year=years.pop(),
        number=None,
        reason="explicit_source_year",
    )


def _identity_prefix(identity: dict[str, Any]) -> str:
    return (
        f"[document kind={identity['kind']}; year={identity['year']}; "
        f"number={identity['number']}; language={identity['language']}]"
    )


def build_identity_reranker_documents(
    documents: list[Document], query_identity: dict[str, Any] | None
) -> list[Document]:
    if query_identity is None:
        return list(documents)
    reranker_documents = []
    for original in documents:
        identity = parse_source_identity(str(original.metadata.get("source", "")))
        if identity is None:
            reranker_documents.append(original)
            continue
        reranker_documents.append(
            Document(
                page_content=f"{_identity_prefix(identity)}\n{original.page_content}",
                metadata=dict(original.metadata),
            )
        )
    return reranker_documents


def _page_key(document: Document) -> tuple[str, int]:
    metadata = document.metadata
    page = normalize_page(metadata)
    try:
        page_number = int(page) if page is not None else -1
    except (TypeError, ValueError):
        page_number = -1
    return str(metadata.get("source", "")).casefold(), page_number


def diversify_ranked_pages(
    ranked: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Keep the highest-scored chunk for each source page, preserving score order."""
    seen: set[tuple[str, int]] = set()
    output = []
    for document, score in ranked:
        key = _page_key(document)
        if key in seen:
            continue
        seen.add(key)
        output.append((document, score))
    return output


def is_arabic_query(query: str) -> bool:
    return bool(ARABIC.search(query))


def _chunk_order(document: Document) -> int:
    try:
        return int(document.metadata.get("chunk_index", document.metadata.get("flat_part", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def _representation_page_key(document: Document):
    return _page_key(document), document.metadata.get("representation")


def page_chunks(documents: list[Document]) -> dict:
    """Group indexed chunks by (source, physical page, representation) in page order."""
    pages: dict = {}
    for document in documents:
        pages.setdefault(_representation_page_key(document), []).append(document)
    for chunks in pages.values():
        chunks.sort(key=_chunk_order)
    return pages


def _join(left: str, right: str) -> str:
    """Join consecutive page chunks, removing the chunker's overlap when it is visible."""
    for size in range(min(len(right), 300), 19, -1):
        if left.endswith(right[:size]):
            return left + right[size:]
    return left + "\n" + right


def _chunks_for_ranked_hit(document: Document, pages: dict) -> list[Document] | None:
    exact = pages.get(_representation_page_key(document))
    if exact:
        return exact
    # Re-ingest may change representation (structured_baseline → native). Prefer
    # the active page for the same source/page regardless of representation label.
    page = _page_key(document)
    candidates = [chunks for key, chunks in pages.items() if key[0] == page]
    if not candidates:
        return None
    for chunks in candidates:
        if chunks and chunks[0].metadata.get("representation") == "native":
            return chunks
    return candidates[0]


def expand_ranked_pages(
    ranked: list[tuple[Document, float]],
    pages: dict,
    *,
    max_chars: int = 8000,
) -> list[tuple[Document, float]]:
    """Give the answer layer the retrieved page, not only its best-scoring chunk.

    Retrieval hit the right page; the answer may sit in a neighbouring chunk. The page
    is rebuilt from the same indexed chunks (same trusted source/page metadata), growing
    outward from the retrieved chunk until max_chars so long pages stay bounded.
    """
    expanded = []
    for document, score in ranked:
        chunks = _chunks_for_ranked_hit(document, pages)
        if not chunks:
            expanded.append((document, score))
            continue
        if len(chunks) == 1:
            chunk = chunks[0]
            if chunk.page_content == document.page_content:
                expanded.append((document, score))
            else:
                metadata = {**document.metadata, "expanded_from_chunk": document.metadata.get("chunk_id")}
                expanded.append((Document(page_content=chunk.page_content, metadata=metadata), score))
            continue
        texts = [chunk.page_content for chunk in chunks]
        try:
            center = texts.index(document.page_content)
        except ValueError:
            # Re-ingested page text no longer equals the ranked chunk; still
            # feed the answer layer the active page (ranking metadata intact).
            center = 0
        left, right, size = center, center, len(texts[center])
        while True:
            grew = False
            for candidate, side in ((left - 1, "left"), (right + 1, "right")):
                if 0 <= candidate < len(texts) and size + len(texts[candidate]) <= max_chars:
                    size += len(texts[candidate])
                    left, right = (candidate, right) if side == "left" else (left, candidate)
                    grew = True
            if not grew:
                break
        text = texts[left]
        for piece in texts[left + 1:right + 1]:
            text = _join(text, piece)
        if text == document.page_content:
            expanded.append((document, score))
            continue
        metadata = {**document.metadata, "expanded_from_chunk": document.metadata.get("chunk_id")}
        expanded.append((Document(page_content=text, metadata=metadata), score))
    return expanded


def _page_document_from_index(
    pages: dict,
    source: str,
    page_number: int,
    *,
    max_chars: int = 8000,
    template: Document | None = None,
) -> Document | None:
    """Rebuild one physical page from the index (same source only)."""
    key = (str(source).casefold(), int(page_number))
    candidates = [chunks for (page_key, _repr), chunks in pages.items() if page_key == key]
    if not candidates:
        return None
    chunks = candidates[0]
    for group in candidates:
        if group and group[0].metadata.get("representation") == "native":
            chunks = group
            break
    if not chunks:
        return None
    text = chunks[0].page_content
    for piece in chunks[1:]:
        if len(text) + len(piece.page_content) > max_chars:
            break
        text = _join(text, piece.page_content)
    meta = {**(template.metadata if template is not None else chunks[0].metadata)}
    meta.update({
        "source": Path(str(meta.get("source") or source)).name,
        "page": page_number,
        "pages": [page_number],
        "page_label": page_number,
        "adjacent_page_expand": True,
    })
    return Document(page_content=text, metadata=meta)


def expand_adjacent_instrument_pages(
    ranked: list[tuple[Document, float]],
    pages: dict,
    *,
    max_extra_per_hit: int = 1,
    max_total_extras: int = 3,
    max_chars: int = 8000,
) -> list[tuple[Document, float]]:
    """Attach same-PDF neighbour pages (±1) when missing from the pack.

    Structural continuation only (same instrument, adjacent physical page). Prefer
    the forward page (article lists often continue). Does not fetch other PDFs.
    """
    if not ranked or not pages:
        return list(ranked or [])
    present = {_page_key(document) for document, _score in ranked}
    extras: list[tuple[Document, float]] = []
    for document, score in ranked:
        if len(extras) >= max_total_extras:
            break
        source, page_number = _page_key(document)
        if page_number < 1 or not source:
            continue
        added = 0
        # Forward first: exceptions often follow the general rule on the next page.
        for neighbor in (page_number + 1, page_number - 1):
            if added >= max_extra_per_hit or len(extras) >= max_total_extras:
                break
            if neighbor < 1 or (source, neighbor) in present:
                continue
            neighbor_doc = _page_document_from_index(
                pages, source, neighbor, max_chars=max_chars, template=document,
            )
            if neighbor_doc is None:
                continue
            extras.append((neighbor_doc, float(score) * 0.99))
            present.add((source, neighbor))
            added += 1
    return list(ranked) + extras


def expand_answer_pages(
    ranked: list[tuple[Document, float]],
    pages: dict,
    *,
    max_chars: int = 8000,
) -> list[tuple[Document, float]]:
    """Same-page chunk rebuild, then bounded same-instrument adjacent pages."""
    expanded = expand_ranked_pages(ranked, pages, max_chars=max_chars)
    return expand_adjacent_instrument_pages(expanded, pages, max_chars=max_chars)
