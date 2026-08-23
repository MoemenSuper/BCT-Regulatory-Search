import re
from dataclasses import dataclass


_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ\u0600-\u06ff0-9٠-٩]")
_MOJIBAKE = ("�", "Ã", "Ø", "Ù")


@dataclass(frozen=True)
class PageQuality:
    score: float
    requires_fallback: bool
    flags: list[str]


def assess_page_quality(text: str, block_count: int) -> PageQuality:
    """Conservatively identify pages whose native extraction needs OCR."""
    stripped = text.strip()
    flags: list[str] = []
    if not stripped:
        return PageQuality(0.0, True, ["no_native_text"])

    visible = [character for character in stripped if not character.isspace()]
    useful = sum(bool(_WORD.match(character)) for character in visible)
    useful_ratio = useful / max(len(visible), 1)
    mojibake_ratio = sum(stripped.count(token) for token in _MOJIBAKE) / max(len(stripped), 1)
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

    score = max(0.0, min(score, 1.0))
    return PageQuality(score, score < 0.55, flags)
