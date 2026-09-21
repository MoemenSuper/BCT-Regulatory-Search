"""Answer-only identity and literal gates. Never use benchmark labels or reorder retrieval."""
import re
import unicodedata
from difflib import SequenceMatcher

from graph_contract import is_relationship_query, is_temporal_rule_query
from retrieval_selection import _ARABIC_RANGE as _AR, parse_query_identity, parse_source_identity


_TYPOGRAPHY = str.maketrans({**{c: "-" for c in "‐‑‒–—−"}, "’": "'", "‘": "'",
                           **{c: str(unicodedata.decimal(c)) for c in "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"}})
# Arabic tatweel and harakat: presentation marks that OCR emits inconsistently.
_ARABIC_MARKS = re.compile(r"[\u0640\u064b-\u065f\u0670]")
_NEGATION = re.compile(r"(?<![\w-])(?:n'|(?:ne|pas|non|sans|jamais|aucun|aucune|not|no|never|لا|لم|لن|ليس|ليست|غير|دون)(?![\w-]))", re.I)


def plain(text):
    # Formatting equivalence only: never reverse digits, remove punctuation,
    # skip words or fuzzy-match a quote.
    return " ".join(unicodedata.normalize("NFC", text).translate(_TYPOGRAPHY).split())


def _compact(text):
    """Drop whitespace and Arabic presentation marks for locating, never for display."""
    return _ARABIC_MARKS.sub("", text).replace(" ", "")


def _negations(text):
    return sorted(_NEGATION.findall(text.casefold()))


def _near_match(haystack, needle):
    """Bounded near-match on compact text: ≥95% of the quote's characters align
    contiguously, the span stays within the quote's length, digits are identical.
    Locating tolerates OCR letter noise, never a changed number."""
    if len(needle) < 20:
        return None
    blocks = [b for b in SequenceMatcher(None, haystack, needle, autojunk=False).get_matching_blocks() if b.size]
    if not blocks:
        return None
    matched = sum(b.size for b in blocks)
    start, end = blocks[0].a, blocks[-1].a + blocks[-1].size
    if matched < 0.95 * len(needle) or end - start > len(needle) * 1.05 + 3:
        return None
    if re.findall(r"\d+", haystack[start:end]) != re.findall(r"\d+", needle):
        return None
    return start, end


def source_quote(quote, text):
    """Recover the original excerpt, including its typography, after matching.

    Locating tolerates whitespace/punctuation spacing and small OCR letter noise;
    the returned quotation is always the page's own characters.
    """
    needle = plain(quote)
    if not needle:
        raise ValueError("empty_quote")
    # Track offsets through whitespace/typographic normalization. NFC is applied
    # per grapheme so decomposed accents do not lose their original offsets.
    chars, starts, ends = [], [], []
    for match in re.finditer(r"\s+|[^\s][\u0300-\u036f]*", text):
        token = " " if match[0].isspace() else unicodedata.normalize("NFC", match[0]).translate(_TYPOGRAPHY)
        for char in token:
            chars.append(char)
            starts.append(match.start())
            ends.append(match.end())
    normalized = "".join(chars)
    offset = normalized.find(needle)
    if offset >= 0:
        return text[starts[offset]:ends[offset + len(needle) - 1]]

    # Whitespace-insensitive pass: "premier :La Banque" vs "premier : La Banque",
    # "2018 ," vs "2018,". Offsets map compact positions back to the page.
    keep = [i for i, c in enumerate(normalized) if c != " " and not _ARABIC_MARKS.match(c)]
    compact = "".join(normalized[i] for i in keep)
    compact_needle = _compact(needle)
    if not compact_needle:
        raise ValueError("empty_quote")
    span = None
    offset = compact.find(compact_needle)
    if offset >= 0:
        span = offset, offset + len(compact_needle)
    else:
        segments = [_compact(s.strip()) for s in re.split(r"(?:\[?…\]?|\[?\.{3}\]?)", needle)]
        if len(segments) >= 2 and all(len(s) >= 10 for s in segments):
            cursor, first = 0, None
            for segment in segments:
                index = compact.find(segment, cursor)
                if index < 0:
                    first = None
                    break
                first = index if first is None else first
                cursor = index + len(segment)
            if first is not None:
                # Return the entire original span, including any omitted condition.
                span = first, cursor
    if span is None:
        span = _near_match(compact, compact_needle)
        # A flipped polarity ("ne peut pas" -> "peut") is within 5% of the characters
        # but must never anchor a claim to the page.
        if span is not None and _negations(normalized[keep[span[0]]:keep[span[1] - 1] + 1]) != _negations(needle):
            span = None
    if span is None:
        raise ValueError("quote_not_found")
    return text[starts[keep[span[0]]]:ends[keep[span[1] - 1]]]


def numeric_literals(text):
    # Preserve leading zeroes and decimal scale. 10.000 may be a decimal or a
    # thousands grouping: do not guess. The writer can copy the source notation.
    normalized = plain(text)
    normalized = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in normalized)
    normalized = re.sub(r"(?<!\d)\d{1,3}(?: \d{3})+(?!\d)",
                        lambda m: m[0].replace(" ", ""), normalized)
    return set(re.findall(r"\d+(?:[.,٫]\d+)?", normalized.replace(",", ".").replace("٫", ".")))


def supported_numbers(text):
    numbers = numeric_literals(text)
    # Common standalone word/digit equivalences, not arithmetic or unit conversion.
    words = (
        "zero zéro صفر", "one un une واحد واحدة", "two deux اثنان اثنين", "three trois ثلاثة ثلاث",
        "four quatre أربعة أربع", "five cinq خمسة خمس", "six six ستة ست", "seven sept سبعة سبع",
        "eight huit ثمانية ثمان", "nine neuf تسعة تسع", "ten dix عشرة عشر", "eleven onze",
        "twelve douze", "thirteen treize", "fourteen quatorze", "fifteen quinze", "sixteen seize",
    )
    folded = plain(text).casefold()
    for value, aliases in enumerate(words):
        if re.search(r"(?<![\w-])(?:" + "|".join(aliases.split()) + r")(?![\w-])", folded):
            numbers.add(str(value))
    return numbers


def explicit_identity(question):
    identity = parse_query_identity(question)
    if identity and identity["number"] is not None:
        return identity
    # Arabic also commonly writes the year-number form rather than 'عدد ... لسنة'.
    match = re.search(r"(?:المنشور|منشور|المذكرة|مذكرة)\s*(?:عدد|رقم)?\s*(\d{4})\s*[-/]\s*(\d+)", plain(question))
    if match:
        return dict(kind="note" if "مذكرة" in match[0] else "cir", year=int(match[1]), number=int(match[2]))
    return None


def direct_identity(question):
    # A question about amendments, history or comparisons needs the related
    # documents too. A direct contents question must stay on its named instrument.
    if is_temporal_rule_query(question) or is_relationship_query(question) or re.search(
        r"\b(?:compar\w*|entre|between|évolu\w*)\b|مقارنة|بين|تعديل|تغيير|استبدل",
        question, re.I,
    ):
        return None
    identities = re.findall(r"\b\d{4}\s*[-/]\s*\d{1,3}\b", plain(question))
    if len(set(identities)) > 1:
        return None
    return explicit_identity(question)


def identity_matches(source, target):
    identity = parse_source_identity(source)
    aliases = {"cb": "cir", "ci": "cir", "nb": "note"}
    return bool(identity and identity["year"] == target["year"] and identity["number"] == target["number"]
                and aliases.get(identity["kind"], identity["kind"]) == aliases.get(target["kind"], target["kind"]))


def evidence_problem(record):
    """Ingestion-detected extraction conflicts make a passage unusable."""
    if record.get("numeric_conflict") or record.get("extraction_conflict"):
        return "extraction_conflict"
    return None


def evidence_warning(record):
    """Observable OCR noise in the page header. Identity still comes from the trusted
    filename and quotes stay digit-exact, so the body may support claims; the model
    is told never to repair the header. This is not a complete OCR detector."""
    identity = parse_source_identity(record["source"])
    if not identity:
        return None
    text = plain(record["text"])
    header = re.match(r"\s*(?:CIRCULAIRE|NOTE)\b.{0,150}?n\s*[°ºo]\s*(\d{4})\s*[-/]\s*(\d+)", text, re.I)
    if header and (int(header[1]), int(header[2])) != (identity["year"], identity["number"]):
        return "source_header_conflict"
    arabic = _ARABIC_MARKS.sub("", text[:450])
    # The instrument's own title ("منشور إلى البنوك عدد 4 لسنة 2016"); a body
    # cross-reference is prefixed ("بالمنشور عدد 6 لسنة 2008") and is not a header.
    header_ar = re.search(rf"(?<![{_AR}])(?:مذكرة|منشور).{{0,100}}?(?:عدد|رقم)\s*(\d+)\s*لسنة\s*(\d{{4}})", arabic)
    if header_ar and (int(header_ar[2]), int(header_ar[1])) != (identity["year"], identity["number"]):
        return "source_header_conflict"
    # Impossible Gregorian years in a contemporary instrument are a corruption
    # signal (a broken font map yields "لسنة 6112" for 2016), never an invitation
    # to silently transpose or replace the digits.
    dated = r"(?:لسنة|جانفي|فيفري|مارس|أفريل|ماي|جوان|جويلية|أوت|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s*(\d{4})(?!\d)"
    for match in re.finditer(dated, _ARABIC_MARKS.sub("", text)):
        if not 1900 <= int(match[1]) <= identity["year"] + 10:
            return "implausible_gregorian_year"
    return None


def strip_instrument_references(text, records):
    """Remove 'YEAR-NUMBER' / 'عدد N لسنة YEAR' references to the cited instruments.

    The identity is trusted filename metadata, so naming it in a claim is not a rule
    number that must be quoted. Only the cited instruments' own identifiers are removed.
    Counterpart ids from pinned/graph temporal_* metadata are trusted the same way.
    Bare sequence numbers alone (e.g. '41 dinars') are NOT trusted — only YEAR-NUMBER
    reference forms are stripped.
    """
    for year, number in _cited_instrument_year_numbers(records):
        text = re.sub(rf"(?<!\d){year}\s*[-/]\s*0*{number}(?!\d)", " ", text)
        text = re.sub(rf"(?:cir|note)\s*:\s*{year}\s*:\s*0*{number}(?!\d)", " ", text, flags=re.I)
        text = re.sub(rf"(?:nombre|رقم|عدد)\s*0*{number}\s*لسنة\s*{year}(?!\d)", " ", text)
    return text


def trusted_years(question, records):
    """Years the claim may name without quoting: the question's own and the cited
    instruments' filename years (plus temporal counterpart years). Trusted metadata."""
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", plain(question)))
    for year, _number in _cited_instrument_year_numbers(records):
        years.add(str(year))
    return years


_SCENARIO_DATE = re.compile(
    r"(?:"
    # French: 26 mars 2026
    r"\b(\d{1,2})\s+"
    r"(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre)\s+"
    r"(\d{4})\b"
    r"|"
    # Arabic (Tunisian + MSA): 26 مارس 2026 / 26 جانفي 2026
    r"(\d{1,2})\s*"
    r"(?:جانفي|يناير|فيفري|فبراير|مارس|أفريل|أبريل|افريل|"
    r"ماي|مايو|جوان|يونيو|جويلية|يوليو|أوت|أغسطس|اوت|"
    r"سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر)\s*"
    r"(\d{4})"
    r")",
    re.I,
)


def question_scenario_numbers(question: str) -> set[str]:
    """Calendar/scenario numbers from the user question (French or Arabic).

    These frame the hypothetical (e.g. 'avant le 26 mars 2026' / 'قبل 26 مارس 2026')
    and may be restated in a claim without appearing in the supporting quotation.
    Regulatory quanta that are not in the question still require quote support.
    """
    found: set[str] = set()
    for match in _SCENARIO_DATE.finditer(plain(question or "")):
        day = match.group(1) or match.group(3)
        year = match.group(2) or match.group(4)
        if day:
            found.add(str(int(day)))
        if year:
            found.add(year)
    return found


def _cited_instrument_year_numbers(records):
    """(year, number) pairs trusted from cited filenames and temporal_* ids."""
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(year: int, number: int) -> None:
        key = (year, number)
        if key in seen:
            return
        seen.add(key)
        pairs.append(key)

    for record in records:
        identity = parse_source_identity(record.get("source") or "")
        if identity:
            add(int(identity["year"]), int(identity["number"]))
        for key in ("temporal_source_id", "temporal_target_id"):
            match = re.match(
                r"(?:cir|note):(\d{4}):(\d+)$",
                str(record.get(key) or "").strip(),
                re.I,
            )
            if match:
                add(int(match.group(1)), int(match.group(2)))
    return pairs


def claim_asserts_unverified_applicability(text: str) -> bool:
    """True when the claim asserts present force/currentness under unverified scope.

    Negative / document-scoped wording ("n'est plus en vigueur", "abrogée") is allowed
    when backed by a locating quote; affirmative "est en vigueur" / "currently" is not.
    """
    cleaned = re.sub(
        r"\b(?:n['’]est\s+plus|n['’]est\s+pas|plus|no\s+longer|not)\s+"
        r"(?:en\s+vigueur|in\s+force|applicable)\b|"
        r"\b(?:abrogée?s?|remplacée?s?|modifiée?s?|repealed|superseded)\b|"
        r"(?:لم\s+تعد|ليست\s+سارية|ملغاة|معوضة)",
        " ",
        text or "",
        flags=re.I,
    )
    return bool(
        re.search(
            r"\b(?:actuel(?:le(?:ment)?)?s?|currently|current|today|now|"
            r"en\s+vigueur|in\s+force)\b|"
            r"(?:ساري|سارية|الساري|النافذ|الحالي|حالي)",
            cleaned,
            re.I,
        )
    )


_NEGATIVE_AMENDMENT_CLAIM = re.compile(
    r"(?i)(?:"
    r"(?:aucun|aucune|pas\s+de|nul(?:le)?)\s+(?:\w+\s+){0,5}"
    r"(?:texte|circulaire|disposition|amendement|modification)\s+(?:\w+\s+){0,8}"
    r"(?:ult[eé]rieur|post[eé]rieur|plus\s+r[eé]cent|subs[eé]quent|"
    r"modifi|abrog|remplac)|"
    r"(?:ne\s+)?(?:modifie|abroge|remplace)\s+pas|"
    r"n['’]a\s+pas\s+(?:[eé]t[eé]\s+)?(?:modifi|abrog|remplac)|"
    r"(?:no|without)\s+(?:later|subsequent|further)\s+"
    r"(?:text|circular|amendment|modification)|"
    r"does\s+not\s+(?:modify|amend|abrogate|replace)|"
    r"nothing\s+(?:later|subsequent)\s+(?:modifies|amends)|"
    r"no\s+later\s+text|"
    r"لا\s+يوجد\s+(?:أي\s+)?نص\s+(?:لاحق|معدل)|"
    r"لم\s+(?:يعدل|تعد|يلغ|تعدل)"
    r")"
)


def claim_asserts_unsupported_negative_amendment(text: str) -> bool:
    """True when the claim asserts that no later text modifies/abrogates an instrument.

    Absence from retrieval is not proof; only a literal quote of that assertion may
    support it.
    """
    return bool(_NEGATIVE_AMENDMENT_CLAIM.search(text or ""))

