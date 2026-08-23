import re
from dataclasses import dataclass
from enum import Enum

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


# Regex patterns to identify specific document sections
ARTICLE_PATTERN_FR = re.compile(
    r"^\s*"
    r"(Article\s+"
    r"(?:premier|1er|\d+(?:\s*(?:bis|ter|quater))?)"
    r"(?:\s*\([^)]*\))?"
    r")"
    r"\s*[:\-–—]?\s*"
    r"(.*)$",
    re.IGNORECASE,
)


TITLE_PATTERN_FR = re.compile(
    r"^\s*(TITRE\s+[IVXLCDM\d]+\b.*)$",
    re.IGNORECASE,
)


CHAPTER_PATTERN_FR = re.compile(
    r"^\s*(CHAPITRE\s+"
    r"(?:premier|1er|\d+|[IVXLCDM]+)"
    r"\b.*)$",
    re.IGNORECASE,
)


SECTION_PATTERN_FR = re.compile(
    r"^\s*(SECTION\s+"
    r"(?:première|premier|1er|\d+|[IVXLCDM]+)"
    r"\b.*)$",
    re.IGNORECASE,
)


SUBSECTION_PATTERN_FR = re.compile(
    r"^\s*(SOUS[-\s]?SECTION\s+"
    r"(?:première|premier|1er|\d+|[IVXLCDM]+)"
    r"\b.*)$",
    re.IGNORECASE,
)


ANNEX_PATTERN_FR = re.compile(
    r"^\s*(ANNEXE\b.*)$",
    re.IGNORECASE,
)


def recognize_french_structure(text: str) -> StructureMatch | None:

    match = ARTICLE_PATTERN_FR.match(text)

    if match:
        return StructureMatch(
            type = StructureType.ARTICLE,
            heading = match.group(1).strip(),
            body = match.group(2).strip(),
        )

    patterns = [
        (StructureType.TITLE, TITLE_PATTERN_FR),
        (StructureType.CHAPTER, CHAPTER_PATTERN_FR),
        (StructureType.SUBSECTION, SUBSECTION_PATTERN_FR),
        (StructureType.SECTION, SECTION_PATTERN_FR),
        (StructureType.ANNEX, ANNEX_PATTERN_FR),
    ]

    for structure_type, pattern in patterns:
        match = pattern.match(text)

        if match:
            return StructureMatch(
                type = structure_type,
                heading = match.group(1).strip(),
            )

    return None


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
            self.chapter = None
            self.section = None
            self.subsection = None
            self.article = None
            self.annex = None

        elif structure.type == StructureType.CHAPTER:
            self.chapter = structure.heading
            self.section = None
            self.subsection = None
            self.article = None

        elif structure.type == StructureType.SECTION:
            self.section = structure.heading
            self.subsection = None
            self.article = None

        elif structure.type == StructureType.SUBSECTION:
            self.subsection = structure.heading
            self.article = None

        elif structure.type == StructureType.ARTICLE:
            self.article = structure.heading

        elif structure.type == StructureType.ANNEX:
            self.title = None
            self.chapter = None
            self.section = None
            self.subsection = None
            self.article = None
            # When we find a new annex, we need a new hierarchy
            self.annex = structure.heading