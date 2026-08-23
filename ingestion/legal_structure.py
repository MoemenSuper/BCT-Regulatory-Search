import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class StructureType(str, Enum):
    TITLE = "title"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    ARTICLE = "article"
    ANNEX = "annex"


@dataclass
class StructureMatch:
    type: StructureType
    heading: str
    body: str = ""


ARTICLE_PATTERN_FR = re.compile(
    r"^\s*(Article\s+(?:premier|1er|\d+(?:\s*(?:bis|ter|quater))?)"
    r"(?:\s*\([^)]*\))?)\s*[:\-–—]?\s*(.*)$",
    re.IGNORECASE,
)
TITLE_PATTERN_FR = re.compile(r"^\s*(TITRE\s+[IVXLCDM\d]+\b.*)$", re.IGNORECASE)
CHAPTER_PATTERN_FR = re.compile(
    r"^\s*(CHAPITRE\s+(?:premier|1er|\d+|[IVXLCDM]+)\b.*)$", re.IGNORECASE
)
SECTION_PATTERN_FR = re.compile(
    r"^\s*(SECTION\s+(?:première|premier|1er|\d+|[IVXLCDM]+)\b.*)$", re.IGNORECASE
)
SUBSECTION_PATTERN_FR = re.compile(
    r"^\s*(SOUS[-\s]?SECTION\s+(?:première|premier|1er|\d+|[IVXLCDM]+)\b.*)$",
    re.IGNORECASE,
)
ANNEX_PATTERN_FR = re.compile(r"^\s*(ANNEXE\b.*)$", re.IGNORECASE)

ARTICLE_PATTERN_AR = re.compile(
    r"^\s*((?:الفصل|فصل|المادة|مادة)\s+(?:[\d٠-٩]+|الأول(?:ى)?|الثاني(?:ة)?|الثالث(?:ة)?))"
    r"\s*[:\-–—]?\s*(.*)$",
    re.IGNORECASE,
)
TITLE_PATTERN_AR = re.compile(r"^\s*((?:العنوان|الباب)\s+.+)$", re.IGNORECASE)
CHAPTER_PATTERN_AR = re.compile(r"^\s*((?:الفصل\s+الفرعي)\s+.+)$", re.IGNORECASE)
SECTION_PATTERN_AR = re.compile(r"^\s*((?:القسم|الجزء)\s+.+)$", re.IGNORECASE)
SUBSECTION_PATTERN_AR = re.compile(r"^\s*((?:القسم\s+الفرعي|الفرع)\s+.+)$", re.IGNORECASE)
ANNEX_PATTERN_AR = re.compile(r"^\s*((?:الملحق|ملحق)\b.*)$", re.IGNORECASE)


def _recognize(
    text: str,
    article_pattern: re.Pattern[str],
    patterns: list[tuple[StructureType, re.Pattern[str]]],
) -> StructureMatch | None:
    normalized = " ".join(text.split())
    match = article_pattern.match(normalized)
    if match:
        return StructureMatch(
            type=StructureType.ARTICLE,
            heading=match.group(1).strip(),
            body=match.group(2).strip(),
        )

    for structure_type, pattern in patterns:
        match = pattern.match(normalized)
        if match:
            return StructureMatch(type=structure_type, heading=match.group(1).strip())
    return None


def recognize_french_structure(text: str) -> StructureMatch | None:
    return _recognize(text, ARTICLE_PATTERN_FR, [
        (StructureType.TITLE, TITLE_PATTERN_FR),
        (StructureType.CHAPTER, CHAPTER_PATTERN_FR),
        (StructureType.SUBSECTION, SUBSECTION_PATTERN_FR),
        (StructureType.SECTION, SECTION_PATTERN_FR),
        (StructureType.ANNEX, ANNEX_PATTERN_FR),
    ])


def recognize_arabic_structure(text: str) -> StructureMatch | None:
    return _recognize(text, ARTICLE_PATTERN_AR, [
        (StructureType.TITLE, TITLE_PATTERN_AR),
        (StructureType.CHAPTER, CHAPTER_PATTERN_AR),
        (StructureType.SUBSECTION, SUBSECTION_PATTERN_AR),
        (StructureType.SECTION, SECTION_PATTERN_AR),
        (StructureType.ANNEX, ANNEX_PATTERN_AR),
    ])


def recognizer_for_language(language: str) -> Callable[[str], StructureMatch | None]:
    if language == "fr":
        return recognize_french_structure
    if language == "ar":
        return recognize_arabic_structure
    return lambda _text: None


@dataclass
class HierarchyState:
    title: str | None = None
    chapter: str | None = None
    section: str | None = None
    subsection: str | None = None
    article: str | None = None
    annex: str | None = None

    def update(self, structure: StructureMatch) -> None:
        if structure.type == StructureType.TITLE:
            self.title = structure.heading
            self.chapter = self.section = self.subsection = self.article = self.annex = None
        elif structure.type == StructureType.CHAPTER:
            self.chapter = structure.heading
            self.section = self.subsection = self.article = None
        elif structure.type == StructureType.SECTION:
            self.section = structure.heading
            self.subsection = self.article = None
        elif structure.type == StructureType.SUBSECTION:
            self.subsection = structure.heading
            self.article = None
        elif structure.type == StructureType.ARTICLE:
            self.article = structure.heading
        elif structure.type == StructureType.ANNEX:
            self.title = self.chapter = self.section = self.subsection = self.article = None
            self.annex = structure.heading

    def heading_path(self) -> list[str]:
        if self.annex:
            return [self.annex]
        return [
            value
            for value in [self.title, self.chapter, self.section, self.subsection, self.article]
            if value is not None
        ]
