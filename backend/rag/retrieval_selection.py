"""Helpers both search engines (local and cloud) use around the reranker:

1. Recognise circulars by name: in a filename ("Cir_2018_10_fr.pdf") and in a question
   ("la circulaire 2018-10", "المنشور عدد 10 لسنة 2018").
2. Reorder a ranked list: one result per page, named circulars first, "avant la circulaire X"
   questions, named sections ("tableau IV-2").
3. Build what the answer model reads: the whole page around each found chunk, plus a
   neighbour page when a passage continues there.
"""
import re
import unicodedata
from pathlib import Path
from typing import Any
from langchain_core.documents import Document

from rag.source_metadata import normalize_page

# The three Unicode blocks that hold Arabic letters; ARABIC matches any one of them.
_ARABIC_RANGE = r"\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
ARABIC = re.compile(f"[{_ARABIC_RANGE}]")

# A BCT filename: kind (cir, note, ...), year, number, then the language. "Cir_2018_10_fr.pdf"
# gives kind=cir, year=2018, number=10, language=fr. (?P<name>...) names a part of the match.
_SOURCE_ID = re.compile(
    r"(?i)^(?P<kind>cir|note|cb|nb|ci)[_ -]?(?P<year>\d{4})"
    r"[_ -]?(?P<number>\d+).*?[_ -]?(?P<language>fr|ar)\.pdf$"
)


# The patterns below find a circular named in a question, each in one way of writing it.
# A filename typed in the question: "Cir_2018_10".
_FILENAME_QUERY_ID = re.compile(
    r"(?i)\b(?P<kind>cir|note|cb|nb|ci)[_ -](?P<year>\d{4})"
    r"[_ -](?P<number>\d+)(?:[_ -](?:fr|ar))?(?:\.pdf)?\b"
)


# French or English: "circulaire 2018-10", "circulaire n° 10 de 2018", "circular 2018/10".
_FRENCH_QUERY_ID = re.compile(
    r"(?i)\b(?P<kind>circulaire|circular|note)\s+(?:BCT\s+)?"
    r"(?:(?:n(?:[°ºo])?\.?|numero|number)\s*)?"
    r"(?:(?P<year>\d{4})\s*[-/]\s*(?P<number>\d+)"
    r"|(?P<number_first>\d+)\s*(?:de|/)\s*(?P<year_last>\d{4}))"
)


# "Selon 2018-10" / "According to 2018-10": after such a word, the kind may be left out.
_CITED_BARE_ID = re.compile(
    r"(?i)\b(?:selon|d['']apres|dapres|according\s+to)\s+"
    r"(?:(?:la\s+|the\s+)?(?P<kind>circulaire|circular|note)\s+)?"
    r"(?:BCT\s+)?"
    r"(?:(?:n(?:[°ºo])?\.?|numero|number)\s*)?"
    r"(?P<year>\d{4})\s*[-/]\s*(?P<number>\d+)\b"
)


# Arabic, the official form: "المنشور عدد 10 لسنة 2018" (circular number 10 of year 2018).
_ARABIC_QUERY_ID = re.compile(
    r"(?P<kind>المنشور|منشور|المذكرة|مذكرة)\s*"
    r"(?:عدد|رقم)\s*(?P<number>\d+)\s*"
    r"(?:لسنة|سنة|لعام|عام)\s*(?P<year>\d{4})"
)


# Arabic with the year-number form: "حسب المنشور 2020-03", "وفقا للمنشور عدد 2016-01".
_ARABIC_YEAR_NUMBER_ID = re.compile(
    r"(?P<kind>المنشور|منشور|المذكرة|مذكرة)\s*"
    r"(?:عدد|رقم)?\s*"
    r"(?P<year>\d{4})\s*[-/]\s*(?P<number>\d+)"
)


# Arabic "حسب 2020-03" (according to 2020-03), without the kind word, like "Selon 2018-10".
_ARABIC_CITED_BARE_ID = re.compile(
    r"(?:حسب|وفقاً?|طبقاً?|بموجب)\s+"
    r"(?:(?P<kind>المنشور|منشور|المذكرة|مذكرة)\s+)?"
    r"(?:عدد|رقم)?\s*"
    r"(?P<year>\d{4})\s*[-/]\s*(?P<number>\d+)"
)


# A year alone ("publiée en 2019", "لسنة 2019"): a weaker hint, used only when no number is named.
_FRENCH_YEAR = re.compile(
    r"(?i)\b(?:en|type|publiee?\s+en|emise?\s+en|adoptee?\s+en)"
    r"\s+(?P<year>(?:19|20)\d{2})\b"
)


_ARABIC_YEAR = re.compile(
    r"(?:لسنة|سنة|لعام|عام)\s*(?P<year>(?:19|20)\d{2})(?!\d)"
)


# "la prochaine circulaire", "سيصدر": about a text that does not exist yet, so no year hint.
_FUTURE = re.compile(
    r"(?i)\b(?:futur|future|prochain|prochaine|sera\s+publie)\b|"
    r"سيصدر|ستصدر|القادم|القادمة"
)


def _ascii_fold(text: str) -> str:
    """Remove accents ("numéro" -> "numero") so the patterns need only one spelling."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )


def parse_source_identity(source: str) -> dict[str, Any] | None:
    """{"kind", "year", "number", "language"} from a BCT filename, or None if it is not one."""
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


# Every way of naming a kind, mapped to the two kinds filenames use: "cir" (circular), "note".
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
    """One identity dict, the same shape whichever pattern found it."""
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
    """The circular (or year) the question names, or None. The patterns are tried from the most
    precise to the weakest; the first match wins."""
    # Every kind of dash ("2018–10", "2018—10") becomes a plain hyphen first.
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

    cited_bare = _CITED_BARE_ID.search(folded)
    if cited_bare:
        values = cited_bare.groupdict()
        return _query_identity(
            kind=values["kind"] or "cir",
            year=values["year"],
            number=values["number"],
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

    arabic_year_number = _ARABIC_YEAR_NUMBER_ID.search(query)
    if arabic_year_number:
        values = arabic_year_number.groupdict()
        return _query_identity(
            kind=values["kind"],
            year=values["year"],
            number=values["number"],
            reason="explicit_instrument_identity",
        )

    arabic_cited = _ARABIC_CITED_BARE_ID.search(query)
    if arabic_cited:
        values = arabic_cited.groupdict()
        return _query_identity(
            kind=values["kind"] or "cir",
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
    """A label like "[document kind=cir; year=2018; number=10; language=fr]"."""
    return (
        f"[document kind={identity['kind']}; year={identity['year']}; "
        f"number={identity['number']}; language={identity['language']}]"
    )


def build_identity_reranker_documents(
    documents: list[Document], query_identity: dict[str, Any] | None
) -> list[Document]:
    """When the question names a circular, put each chunk's own identity label in front of its
    text, so the reranker can see which chunks come from the named circular."""
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


def explicit_instrument_identity(query: str) -> dict[str, Any] | None:
    """The named circular only when kind, year and number are all known (a year alone is too
    vague to pull a document to the top)."""
    identity = parse_query_identity(query)
    if identity and identity.get("kind") and identity.get("number") is not None:
        return identity
    return None


def source_matches_identity(source: str, identity: dict[str, Any]) -> bool:
    """Is this PDF the named circular (same kind, year and number, any language)?"""
    source_identity = parse_source_identity(str(source))
    if not source_identity:
        return False
    return all(
        source_identity[key] == identity[key] for key in ("kind", "year", "number")
    )


# Section labels a question can name: a table or chapter code ("IV-2", "II-1-B"), or an article,
# chapter, section, annex or table with its number ("Article 3", "annexe 2", "tableau 5").
_SECTION_LABEL = re.compile(
    r"(?<![\w-])([IVXLC]{1,6}-\d{1,3}(?:-[A-Z])?)(?![\w-])"
    r"|\b((?:article|chapitre|section|annexe|tableau)\s+(?:n\s*°\s*)?(?:\d+|premier|1er))\b",
    re.IGNORECASE,
)


def named_section_labels(query: str) -> list[str]:
    """Section labels the question names, as written ("IV-2", "article 3")."""
    return [code or " ".join(phrase.split()) for code, phrase in _SECTION_LABEL.findall(query or "")]


def prefer_named_section_hits(ranked: list[tuple[Document, float]], query: str) -> list[tuple[Document, float]]:
    """A question naming a section ("tableau IV-2", "article 3") reads the passages of that section
    first: look-alike tables (the same rows for banks and for leasing companies) or articles are
    told apart by the label in each chunk's context header (ingestion/chunk.py)."""
    labels = named_section_labels(query)
    if not ranked or not labels:
        return ranked
    patterns = [re.compile(r"(?<![\w-])" + re.escape(label).replace(r"\ ", r"\s+") + r"(?![\w-])", re.IGNORECASE)
                for label in labels]

    def names_it(document: Document) -> bool:
        context = str(document.metadata.get("context") or "")
        return any(pattern.search(context) for pattern in patterns)

    matched = [item for item in ranked if names_it(item[0])]
    if not matched:
        return ranked
    return matched + [item for item in ranked if not names_it(item[0])]


def prefer_named_instrument_hits(
    ranked: list[tuple[Document, float]],
    query_identity: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[tuple[Document, float]]:
    """Chunks from the circular(s) the question names go first, before texts that only mention
    them; the order inside each group stays the reranker's."""
    if not ranked or not query_identity:
        return ranked
    identities = (
        query_identity if isinstance(query_identity, list) else [query_identity]
    )
    identities = [
        item
        for item in identities
        if item and item.get("kind") is not None and item.get("number") is not None
    ]
    if not identities:
        return ranked
    matched: list[tuple[Document, float]] = []
    rest: list[tuple[Document, float]] = []
    for document, score in ranked:
        source = str(document.metadata.get("source", ""))
        if any(source_matches_identity(source, identity) for identity in identities):
            matched.append((document, score))
        else:
            rest.append((document, score))
    if not matched:
        return ranked
    return matched + rest


def _instrument_year(document: Document) -> int:
    """The year in the chunk's filename (0 if none)."""
    identity = parse_source_identity(str(document.metadata.get("source", "")))
    return int(identity["year"]) if identity else 0


# "Avant la circulaire 2025-13" asks for the rules before that circular (the prior regime);
# "avant le 26 mars" or "avant l'entrée en vigueur de 2026-04" asks for the named circular's own rules.
_HISTORICAL_QUERY = re.compile(
    r"(?i)(?:\b(?:avant|before)|قبل)\s+(?:(?:la|le|the)\s+)?"
    r"(?:(?:circulaire|circular|note|منشور)\s+)?(?:(?:n[°o.]?|عدد)\s*)?(?:19|20)\d{2}\s*[-/]\s*\d{1,3}"
    r"|\bancien\s+r[eé]gime\b|\bprevious\s+regime\b|النظام\s*السابق"
)
# "Engagement existant avant 2026-04": something done before the circular, so the question is
# about that circular's transitional rule, not about the prior regime.
_DONE_BEFORE = re.compile(
    r"(?i)\b(?:pris|existants?|conclus?|entam[eé]e?s?|ex[eé]cut[eé]e?s?|sign[eé]e?s?|made|signed|concluded)"
    r"\s+(?:\w+\s+){0,3}(?:avant|before)\b"
)


def is_historical_cutoff_query(query: str) -> bool:
    """Does the question ask what applied before a named circular?"""
    raw = (query or "").translate(str.maketrans({character: "-" for character in "‐‑‒–—−"}))
    if _DONE_BEFORE.search(raw):
        return False
    return bool(_HISTORICAL_QUERY.search(raw) or _HISTORICAL_QUERY.search(_ascii_fold(raw)))


# Any "2018-10" in the question, read as a circular when nothing says otherwise.
_YEAR_NUMBER_TOKEN = re.compile(
    r"(?i)\b(?P<year>(?:19|20)\d{2})\s*[-/]\s*(?P<number>\d{1,3})\b"
)


def query_instrument_refs(query: str) -> list[dict[str, Any]]:
    """Every circular the question names (a question can name several), without repeats."""
    query = query.translate(str.maketrans({character: "-" for character in "‐‑‒–—−"}))
    primary = explicit_instrument_identity(query)
    refs: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    if primary:
        key = (primary["kind"], primary["year"], primary["number"])
        seen.add(key)
        refs.append(primary)
    folded = _ascii_fold(query)
    for match in _YEAR_NUMBER_TOKEN.finditer(folded):
        identity = _query_identity(
            kind="cir",
            year=match.group("year"),
            number=match.group("number"),
            reason="explicit_instrument_identity",
        )
        key = (identity["kind"], identity["year"], identity["number"])
        if key in seen:
            continue
        seen.add(key)
        refs.append(identity)
    return refs


def prefer_historical_hits(
    ranked: list[tuple[Document, float]],
    query: str,
) -> list[tuple[Document, float]]:
    """For "avant la circulaire 2025-13", move 2025-13's own pages after the others: the answer
    must come from the texts that applied before it."""
    if not ranked or not is_historical_cutoff_query(query):
        return ranked
    named = query_instrument_refs(query)
    if not named:
        return ranked
    # The circulars named after "avant" are the cutoff: their pages go to the end.
    demote_keys = {(item["kind"], item["year"], item["number"]) for item in named}
    keep: list[tuple[Document, float]] = []
    demoted: list[tuple[Document, float]] = []
    for document, score in ranked:
        identity = parse_source_identity(str(document.metadata.get("source", "")))
        if identity and (identity["kind"], identity["year"], identity["number"]) in demote_keys:
            demoted.append((document, score))
        else:
            keep.append((document, score))
    return keep + demoted if demoted and keep else ranked


def _page_key(document: Document) -> tuple[str, int]:
    """(lower-case PDF path, page number): identifies a page (-1 when the page is unknown)."""
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
    """Does the text contain any Arabic letter?"""
    return bool(ARABIC.search(query))


def _chunk_order(document: Document) -> int:
    """The chunk's position on its page (to put the pieces back in reading order)."""
    try:
        return int(document.metadata.get("chunk_index", document.metadata.get("flat_part", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def _representation_page_key(document: Document):
    """A page key that also tells apart the PDF's text ("native") from text read off an image."""
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
    """Join consecutive page chunks, removing the chunker's overlap when it is visible.
    The chunker's overlap is at most 400 characters (it goes back to a sentence start)."""
    for size in range(min(len(right), 450), 19, -1):
        if left.endswith(right[:size]):
            return left + right[size:]
    return left + "\n" + right


def _chunks_for_ranked_hit(document: Document, pages: dict) -> list[Document] | None:
    exact = pages.get(_representation_page_key(document))
    if exact:
        return exact
    # The ranked chunk may come from an older reading of the page (another "representation"):
    # take the same page's current chunks, preferring the PDF's own text.
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
    """Give the answer step the page around each ranked chunk, not only the chunk.

    Search found the right page, but the answer may sit in the next chunk of that page. The
    page is rebuilt from its own indexed chunks, growing outward from the found chunk (one
    chunk left, one right, ...) until max_chars, so very long pages stay bounded.
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
            # The page was re-read since this chunk was ranked: grow from the page start instead.
            center = 0
        # left..right is the window of chunks kept so far; widen it while it fits in max_chars.
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
    """Rebuild one page of a PDF from its indexed chunks (for a neighbour page). template is the
    ranked chunk it neighbours: the new page copies its metadata, then its own page fields."""
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
        "context": chunks[0].metadata.get("context"),  # this page's own header, not the template's
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
    """Add the next (or previous) page of the same PDF when it is not already there.

    A rule often continues on the next page (an article's list, its exceptions). Only pages of
    the same PDF are added, at most one per ranked page and max_total_extras in all.
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
            extras.append((neighbor_doc, float(score) * 0.99))  # just below the page it continues
            present.add((source, neighbor))
            added += 1
    return list(ranked) + extras


def expand_answer_pages(
    ranked: list[tuple[Document, float]],
    pages: dict,
    *,
    max_chars: int = 8000,
) -> list[tuple[Document, float]]:
    """What the answer model reads: each ranked chunk grown to its page, then neighbour pages."""
    expanded = expand_ranked_pages(ranked, pages, max_chars=max_chars)
    return expand_adjacent_instrument_pages(expanded, pages, max_chars=max_chars)
