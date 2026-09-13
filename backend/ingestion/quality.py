from __future__ import annotations

import re
from dataclasses import dataclass

from retrieval_selection import ARABIC, _ARABIC_RANGE

_WORD = re.compile(rf"[A-Za-zÀ-ÖØ-öø-ÿ{_ARABIC_RANGE}0-9٠-٩]")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_ARABIC_TOKEN = re.compile(rf"^[{_ARABIC_RANGE}]+$")
_MOJIBAKE = ("�", "Ã", "Ø", "Ù")
_SENSITIVE = re.compile(
    r"(?:\d|[٠-٩]|%|٪|\b(?:19|20)\d{2}\b|\b\d{1,2}[./-]\d{1,2}[./-](?:\d{2}|\d{4})\b|"
    r"(?:دينار|دنانير|مليون|ألف|pour\s*cent|percent|taux|montant|date|أجل|نسبة|مبلغ))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageQuality:
    score: float
    requires_fallback: bool
    flags: tuple[str, ...]
    latin_character_ratio: float
    single_arabic_token_ratio: float


def assess_page_quality(text: str, block_count: int) -> PageQuality:
    """Cheap corruption screening derived from the experimental branch.

    It intentionally catches only obvious native-extraction failures. Silent Arabic
    numeric/date corruption is handled separately by the visual-routing policy.
    """
    stripped = text.strip()
    if not stripped:
        return PageQuality(0.0, True, ("no_native_text",), 0.0, 0.0)

    visible = [character for character in stripped if not character.isspace()]
    useful = sum(bool(_WORD.match(character)) for character in visible)
    useful_ratio = useful / max(len(visible), 1)
    mojibake_ratio = sum(stripped.count(token) for token in _MOJIBAKE) / max(len(stripped), 1)
    latin_ratio = sum(
        "a" <= character.casefold() <= "z" or "\u00c0" <= character <= "\u024f"
        for character in visible
    ) / max(len(visible), 1)
    arabic_tokens = [token for token in _TOKEN.findall(stripped) if _ARABIC_TOKEN.match(token)]
    single_arabic_ratio = (
        sum(len(token) == 1 for token in arabic_tokens) / len(arabic_tokens)
        if arabic_tokens
        else 0.0
    )

    flags: list[str] = []
    score = 1.0
    if len(stripped) < 40:
        flags.append("very_little_native_text")
        score -= 0.55
    elif len(stripped) < 100:
        flags.append("little_native_text")
        score -= 0.25
    if block_count == 0:
        flags.append("no_native_blocks")
        score -= 0.4
    if useful_ratio < 0.45:
        flags.append("low_alphanumeric_ratio")
        score -= 0.35
    if mojibake_ratio > 0.02:
        flags.append("encoding_artifacts")
        score -= 0.35
    if latin_ratio > 0.20 and arabic_tokens:
        flags.append("latin_heavy_arabic_page")
    if single_arabic_ratio >= 0.10:
        flags.append("fragmented_arabic_tokens")

    score = max(0.0, min(score, 1.0))
    requires_fallback = score < 0.55
    return PageQuality(
        score=score,
        requires_fallback=requires_fallback,
        flags=tuple(flags),
        latin_character_ratio=latin_ratio,
        single_arabic_token_ratio=single_arabic_ratio,
    )


def contains_sensitive_literals(text: str) -> bool:
    """Whether a page contains literals for which a one-character OCR error matters."""
    return bool(_SENSITIVE.search(text))


def arabic_character_ratio(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    count = sum(bool(ARABIC.search(c)) for c in visible)
    return count / len(visible)
