from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "documents"
DEFAULT_DATASET = ROOT / "evaluation_queries.json"
MOJIBAKE_MARKERS = ("Ã", "Â", "Ø", "Ù", "â€", "ï¿½", "�")
GENERIC_QUERY_MARKERS = (
    "quel est l'objet",
    "quelle est la date de publication",
    "circulaire",
    "cette note",
    "ce document",
    "selon l'article",
    "selon l’article",
    "ما هو موضوع",
    "ما هو تاريخ صدور",
    "هذا المنشور",
    "هذه المذكرة",
    "cette circulaire",
    "la présente circulaire",
    "circulaire n°",
    "circulaire no",
    "ما هو موضوع",
    "ما هو تاريخ صدور",
    "المنشور عدد",
    "المذكرة عدد",
)
UNSUPPORTED_ANSWER_MARKERS = (
    "information n'est pas disponible",
    "information n’est pas disponible",
    "n'est pas spécifié",
    "n’est pas spécifié",
    "aucun délai spécifié",
    "pas de montant spécifié",
    "pas de critère spécifié",
    "à déterminer",
    "[x]",
)
VALID_LANGUAGES = {"fr", "ar"}
VALID_EVIDENCE_METHODS = {"text_extraction", "visual_review"}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("’", "'").replace("‘", "'").replace("ـ", "")
    value = value.replace("‑", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def numbers(value: str) -> list[str]:
    matches = re.findall(
        r"\d+(?:[ .\u00a0\u202f]\d{3})*(?:[.,]\d+)?",
        unicodedata.normalize("NFKC", value),
    )
    canonical: list[str] = []
    for match in matches:
        is_grouped_integer = bool(
            re.fullmatch(r"\d{1,3}(?:[ .\u00a0\u202f]\d{3})+", match)
        )
        compact = re.sub(r"[ \u00a0\u202f]", "", match)
        if is_grouped_integer:
            compact = compact.replace(".", "")
        else:
            compact = compact.replace(",", ".")
        canonical.append(compact)
    return canonical


def load_page_texts(path: Path) -> list[str]:
    document = fitz.open(path)
    return [page.get_text("text") for page in document]


def validate(
    dataset_path: Path,
    require_all_sources: bool,
    list_missing_sources: bool = False,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        items = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [f"cannot load {dataset_path}: {error}"], warnings
    if not isinstance(items, list):
        return ["dataset root must be a JSON array"], warnings

    corpus_paths = {path.name: path for path in DOCUMENTS.rglob("*.pdf")}
    page_cache: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    seen_queries: dict[str, str] = {}
    represented_sources: set[str] = set()

    for index, item in enumerate(items):
        label = f"item {index}"
        if not isinstance(item, dict):
            errors.append(f"{label}: must be an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            errors.append(f"{label}: id must be a non-empty string")
            item_id = label
        elif item_id in seen_ids:
            errors.append(f"{item_id}: duplicate id")
        else:
            seen_ids.add(item_id)
        label = str(item_id)

        query = item.get("query")
        if not isinstance(query, str) or len(query.strip()) < 15:
            errors.append(f"{label}: query must contain at least 15 characters")
            query = ""
        normalized_query = normalize(query)
        if normalized_query in seen_queries:
            errors.append(f"{label}: duplicates the query in {seen_queries[normalized_query]}")
        elif normalized_query:
            seen_queries[normalized_query] = label
        if any(marker in normalized_query for marker in GENERIC_QUERY_MARKERS):
            errors.append(f"{label}: uses a generic document-lookup question")

        language = item.get("language")
        if language not in VALID_LANGUAGES:
            errors.append(f"{label}: language must be 'fr' or 'ar'")
        for field in ("query", "expected_answer", "evidence_quote"):
            value = item.get(field)
            if isinstance(value, str) and any(marker in value for marker in MOJIBAKE_MARKERS):
                errors.append(f"{label}: {field} contains mojibake")

        relevant = item.get("relevant")
        if not isinstance(relevant, bool):
            errors.append(f"{label}: relevant must be a boolean")
            continue
        if not relevant:
            for field in ("expected_source", "expected_page", "expected_answer", "evidence_quote"):
                if item.get(field) is not None:
                    errors.append(f"{label}: irrelevant item must set {field} to null")
            if item.get("expected_behavior") not in {"abstain", "clarify", "reject_out_of_scope"}:
                errors.append(f"{label}: irrelevant item needs a valid expected_behavior")
            continue

        source = item.get("expected_source")
        page_number = item.get("expected_page")
        answer = item.get("expected_answer")
        evidence = item.get("evidence_quote")
        evidence_method = item.get("evidence_method")
        if source not in corpus_paths:
            errors.append(f"{label}: expected source is absent from corpus: {source!r}")
            continue
        represented_sources.add(source)
        if not isinstance(page_number, int):
            errors.append(f"{label}: expected_page must be an integer")
            continue
        if source not in page_cache:
            page_cache[source] = load_page_texts(corpus_paths[source])
        if not 1 <= page_number <= len(page_cache[source]):
            errors.append(
                f"{label}: expected_page {page_number} is outside 1..{len(page_cache[source])}"
            )
            continue
        if not isinstance(answer, str) or not answer.strip():
            errors.append(f"{label}: expected_answer must be a non-empty string")
            continue
        if any(marker in normalize(answer) for marker in UNSUPPORTED_ANSWER_MARKERS):
            errors.append(f"{label}: relevant answer states that the requested fact is unavailable")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(f"{label}: evidence_quote must be a non-empty string")
            continue
        if evidence_method not in VALID_EVIDENCE_METHODS:
            errors.append(f"{label}: invalid evidence_method {evidence_method!r}")
        if "**" in answer or "[" in answer or "]" in answer or "=" in answer:
            errors.append(f"{label}: expected_answer contains generation or formatting artifacts")

        missing_numbers = [number for number in numbers(answer) if number not in numbers(evidence)]
        if missing_numbers:
            errors.append(
                f"{label}: answer numbers absent from evidence: {', '.join(missing_numbers)}"
            )
        if evidence_method == "text_extraction":
            page_text = normalize(page_cache[source][page_number - 1])
            if normalize(evidence) not in page_text:
                errors.append(f"{label}: extracted evidence does not occur on the cited page")

    missing_sources = sorted(set(corpus_paths) - represented_sources)
    if missing_sources:
        message = f"{len(missing_sources)} corpus PDFs have no relevant query"
        if list_missing_sources:
            message += ": " + ", ".join(missing_sources)
        (errors if require_all_sources else warnings).append(message)

    category_counts = Counter(
        item.get("category") for item in items if isinstance(item, dict) and item.get("relevant")
    )
    warnings.append(
        f"summary: {len(items)} items, {len(represented_sources)}/{len(corpus_paths)} sources, "
        f"categories={dict(category_counts)}"
    )
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--require-all-sources", action="store_true")
    parser.add_argument("--list-missing-sources", action="store_true")
    args = parser.parse_args()
    errors, warnings = validate(
        args.dataset,
        args.require_all_sources,
        args.list_missing_sources,
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print(f"validation failed with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
