"""Answer-only identity and literal gates. Never use benchmark labels or reorder retrieval."""
import re
import unicodedata

from graph_contract import is_relationship_query, is_temporal_rule_query
from retrieval_selection import parse_query_identity, parse_source_identity


_TYPOGRAPHY = str.maketrans({**{c: "-" for c in "‐‑‒–—−"}, "’": "'", "‘": "'",
                           **{c: str(unicodedata.decimal(c)) for c in "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"}})


def plain(text):
    # Formatting equivalence only: never reverse digits, remove punctuation,
    # skip words or fuzzy-match a quote.
    return " ".join(unicodedata.normalize("NFC", text).translate(_TYPOGRAPHY).split())


def source_quote(quote, text):
    """Recover the original excerpt, including its typography, after matching."""
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
    if offset < 0:
        segments = [s.strip() for s in re.split(r"(?:\[?…\]?|\[?\.{3}\]?)", needle)]
        if len(segments) < 2 or any(len(s) < 12 for s in segments):
            raise ValueError("quote_not_found")
        cursor, offsets = 0, []
        for segment in segments:
            index = normalized.find(segment, cursor)
            if index < 0:
                raise ValueError("quote_not_found")
            offsets.append(index)
            cursor = index + len(segment)
        # Return the entire original span, including any omitted condition.
        return text[starts[offsets[0]]:ends[cursor - 1]]
    return text[starts[offset]:ends[offset + len(needle) - 1]]


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
    """Reject observable identity corruption; this is not a complete OCR detector."""
    identity = parse_source_identity(record["source"])
    if not identity:
        return None
    text = plain(record["text"])
    header = re.match(r"\s*(?:CIRCULAIRE|NOTE)\b.{0,150}?n\s*[°ºo]\s*(\d{4})\s*[-/]\s*(\d+)", text, re.I)
    if header and (int(header[1]), int(header[2])) != (identity["year"], identity["number"]):
        return "source_header_conflict"
    arabic = re.sub(r"[\u0640\u064b-\u065f\u0670]", "", text[:450])
    header_ar = re.search(r"(?:مذكرة|منشور).{0,100}?(?:عدد|رقم)\s*(\d+)\s*لسنة\s*(\d{4})", arabic)
    if header_ar and (int(header_ar[2]), int(header_ar[1])) != (identity["year"], identity["number"]):
        return "source_header_conflict"
    # Impossible Gregorian years in a contemporary instrument are a corruption
    # signal, never an invitation to silently transpose or replace the digits.
    for match in re.finditer(r"(?:جانفي|فيفري|مارس|أفريل|ماي|جوان|جويلية|أوت|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+(\d{4})", arabic):
        if int(match[1]) > identity["year"] + 100:
            return "implausible_gregorian_year"
    if record.get("numeric_conflict") or record.get("extraction_conflict"):
        return "extraction_conflict"
    return None
