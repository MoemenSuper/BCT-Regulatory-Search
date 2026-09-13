"""Document identity and page diversity shared by retrieval backends."""
import re
import unicodedata
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


# Shared with regulatory_graph_lite.identity (keep aliases in sync).
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
