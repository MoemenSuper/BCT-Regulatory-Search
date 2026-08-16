from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from groq import APIError, Groq, RateLimitError


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "documents"
DEFAULT_OUTPUT = Path(os.environ.get("TEMP", ROOT)) / "bct_evaluation_draft.json"
MODEL = os.environ.get("BCT_EVAL_MODEL", "openai/gpt-oss-120b")
MAX_BATCH_CHARS = int(os.environ.get("BCT_EVAL_MAX_CHARS", "12000"))
MAX_DOCUMENTS_PER_BATCH = int(os.environ.get("BCT_EVAL_MAX_DOCUMENTS", "3"))
MAX_COMPLETION_TOKENS = int(os.environ.get("BCT_EVAL_MAX_COMPLETION", "2200"))
MAX_ATTEMPTS = int(os.environ.get("BCT_EVAL_MAX_ATTEMPTS", "6"))
MAX_QUESTIONS_PER_SOURCE = int(os.environ.get("BCT_EVAL_MAX_QUESTIONS_PER_SOURCE", "0"))
MAX_SOURCE_CHARS = int(os.environ.get("BCT_EVAL_MAX_SOURCE_CHARS", "0"))

CATEGORIES = {
    "amount_or_rate",
    "deadline_or_duration",
    "eligibility_or_scope",
    "required_action",
    "prohibition_or_limit",
    "exception_or_condition",
    "definition",
    "procedure_or_documents",
    "reporting_or_control",
    "effective_date",
    "other_operational_rule",
}

SYSTEM_PROMPT = """You create a high-precision retrieval evaluation set from public Tunisian Central Bank documents.

Return one JSON object with a `queries` array. Treat the requested number as a maximum for each source, not a quota. Write each question and answer in the document's language.

Each item must contain exactly these fields:
- expected_source: exact filename from a SOURCE marker
- query: a natural, self-contained user question
- category: one of amount_or_rate, deadline_or_duration, eligibility_or_scope, required_action, prohibition_or_limit, exception_or_condition, definition, procedure_or_documents, reporting_or_control, effective_date, other_operational_rule
- expected_page: the 1-based PDF page number from a PAGE marker
- expected_answer: a concise, complete answer
- evidence_start_line: first numbered line supporting the answer
- evidence_end_line: last numbered line supporting the answer

Hard rules:
1. A bank employee, compliance officer, company, farmer, exporter, customer, or researcher should plausibly ask the question without knowing that a particular source document exists.
2. Test operative facts: amounts, rates, deadlines, durations, eligibility, limits, exceptions, definitions, required actions, procedures, documents, reporting, or controls.
3. Never ask for a document's title, general subject, publication date, circular number, note number, article number, signatory, or legal citations in the preamble.
4. In this main dataset, do not use the words `circulaire`, `note`, `document`, `article`, `منشور`, `مذكرة`, `وثيقة`, or a source identifier in a question. Do not use deictic wording such as `nouveau`, `récent`, `cette règle`, `هذا`, or `هذه`. Name the concrete rate, activity, transaction, institution, crop, obligation, or procedure instead.
5. Keep any effective date, sector, transaction type, stakeholder, or other scope needed to prevent an old amendment from sounding like an unqualified current rule. Do not identify the source indirectly through the issuer, signature date, or wording such as `la décision du Gouverneur du ...`.
6. One cited PDF page must fully support the answer. Do not combine facts from different pages.
7. Cite one contiguous range of numbered lines from one page. Include enough lines to support all parts of the answer. The program will copy those lines into evidence_quote; do not write an evidence quote yourself.
8. Preserve every number, unit, currency, percentage, date, negation, condition, exception, table column, and dry/irrigated distinction. Label every number in the answer. Write the answer as a grammatical sentence with a subject and verb; never use `=` or return an unexplained number or range.
9. Do not infer legal advice, current validity, supersession, or facts absent from the supplied text.
10. Avoid duplicate facts and questions that differ only by wording. Combine an amount or rate with its effective date in one question when both describe the same narrow change.
11. Prefer questions with enough concrete wording to retrieve the source, while avoiding the source identifier as a shortcut.
12. If a source lacks enough reliable, distinct, user-useful facts for the requested count, return fewer questions. A rule's amendment mechanism, replaced clause number, signatory, and preamble are not useful facts.
13. Every question must contain the regulated topic needed to understand it in isolation. A question such as `À partir de quelle date les nouvelles conditions de banque entrent-elles en vigueur ?` is too broad. A valid version names the exact change, such as `Quel taux annuel de rémunération de l'épargne s'applique à compter du 2 janvier 2026 ?`.

Quality examples:
- `Je cultive des tomates de saison. Quel est le plafond du crédit par hectare et à quelle date arrive-t-il à échéance ?`
- `Une banque doit-elle accorder automatiquement la totalité du barème prévu pour un crédit agricole ?`
- `ما هي المدة القصوى لإعادة جدولة أصل الدين الفلاحي، وما النسبة الواجب دفعها عند تقديم مطلب التسوية؟`

Bad examples:
- `Quel est l'objet de la circulaire n°2021-07 ?`
- `Que prévoit cette circulaire ?`
- `Quel est le barème pour les amandiers ?` when the table distinguishes dry and irrigated farming.
"""

COMPACT_SYSTEM_PROMPT = """Create grounded evaluation questions from Tunisian Central Bank text. Return only JSON: {"queries": [...]}. Write in the source language.

Each item has exactly: expected_source, query, category, expected_page, expected_answer, evidence_start_line, evidence_end_line. Category is one of: amount_or_rate, deadline_or_duration, eligibility_or_scope, required_action, prohibition_or_limit, exception_or_condition, definition, procedure_or_documents, reporting_or_control, effective_date, other_operational_rule.

Ask a self-contained question a bank employee, business, customer, farmer, exporter, or compliance officer would naturally ask without knowing the source exists. Ask about an operative amount, rate, deadline, eligibility rule, limit, exception, required action, procedure, document, report, or control. Never ask for a title, subject, publication date, source number, article number, signatory, or legal citation. Do not say circular, note, document, this rule, new, or recent. Include the exact activity and scope needed to retrieve the rule.

The concise answer and every number, unit, date, condition, and negation must be fully supported by one contiguous line range on one supplied PDF page. Do not infer current validity or combine pages. The REQUESTED QUESTIONS value is a strict maximum for each source. Return fewer items if reliable operational evidence is absent."""

ACTIVE_SYSTEM_PROMPT = (
    COMPACT_SYSTEM_PROMPT
    if os.environ.get("BCT_EVAL_COMPACT", "").lower() in {"1", "true", "yes"}
    else SYSTEM_PROMPT
)


@dataclass(frozen=True)
class SourcePart:
    source: str
    language: str
    requested_count: int
    pages: tuple[tuple[int, str], ...]

    @property
    def char_count(self) -> int:
        return sum(len(text) for _, text in self.pages)


def language_for(path: Path) -> str:
    return "ar" if path.stem.lower().endswith("_ar") else "fr"


def requested_question_count(page_count: int) -> int:
    if page_count == 1:
        count = 1
    elif page_count <= 3:
        count = 2
    elif page_count <= 10:
        count = 4
    elif page_count <= 25:
        count = 8
    else:
        count = 12
    return min(count, MAX_QUESTIONS_PER_SOURCE) if MAX_QUESTIONS_PER_SOURCE else count


def split_document(path: Path) -> list[SourcePart]:
    document = fitz.open(path)
    pages = tuple((index + 1, page.get_text("text").strip()) for index, page in enumerate(document))
    usable_pages = tuple((number, text) for number, text in pages if text)
    if not usable_pages:
        return []

    if MAX_SOURCE_CHARS:
        selected_pages: list[tuple[int, str]] = []
        remaining = MAX_SOURCE_CHARS
        for number, text in usable_pages:
            if remaining <= 0:
                break
            selected_pages.append((number, text[:remaining]))
            remaining -= len(text)
        usable_pages = tuple(selected_pages)

    target = requested_question_count(len(document))
    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_chars = 0
    for page in usable_pages:
        page_chars = len(page[1])
        if current and current_chars + page_chars > MAX_BATCH_CHARS:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(page)
        current_chars += page_chars
    if current:
        groups.append(current)

    allocations = [max(1, round(target * sum(len(text) for _, text in group) / sum(len(text) for _, text in usable_pages))) for group in groups]
    while sum(allocations) > target:
        largest = max(range(len(allocations)), key=lambda index: allocations[index])
        if allocations[largest] == 1:
            break
        allocations[largest] -= 1
    while sum(allocations) < target:
        largest = max(range(len(groups)), key=lambda index: sum(len(text) for _, text in groups[index]))
        allocations[largest] += 1

    return [
        SourcePart(path.name, language_for(path), count, tuple(group))
        for group, count in zip(groups, allocations, strict=True)
    ]


def make_batches(parts: list[SourcePart]) -> list[list[SourcePart]]:
    batches: list[list[SourcePart]] = []
    current: list[SourcePart] = []
    current_chars = 0
    for part in parts:
        if current and (
            current_chars + part.char_count > MAX_BATCH_CHARS
            or len(current) >= MAX_DOCUMENTS_PER_BATCH
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(part)
        current_chars += part.char_count
    if current:
        batches.append(current)
    return batches


def render_batch(batch: list[SourcePart]) -> str:
    sections: list[str] = [
        f"[[TOTAL MAXIMUM QUESTIONS {sum(part.requested_count for part in batch)}]]"
    ]
    for part in batch:
        sections.append(
            f"[[SOURCE {part.source} | LANGUAGE {part.language} | REQUESTED QUESTIONS {part.requested_count}]]"
        )
        for page_number, text in part.pages:
            numbered_lines = "\n".join(
                f"L{line_number:04d}|{line}"
                for line_number, line in enumerate(text.splitlines(), start=1)
            )
            sections.append(
                f"[[SOURCE {part.source} | PDF PAGE {page_number}]]\n{numbered_lines}"
            )
    return "\n\n".join(sections)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("’", "'").replace("‘", "'").replace("ـ", "")
    return re.sub(r"\s+", " ", value).strip().casefold()


def parse_line_number(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"L?(\d+)", value.strip(), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def canonical_numbers(value: str) -> list[str]:
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


def validate_item(item: dict[str, Any], batch: list[SourcePart]) -> list[str]:
    errors: list[str] = []
    required = {
        "expected_source",
        "query",
        "category",
        "expected_page",
        "expected_answer",
        "evidence_start_line",
        "evidence_end_line",
    }
    missing = required - set(item)
    if missing:
        return [f"missing fields: {sorted(missing)}"]

    source = item["expected_source"]
    matching_parts = [part for part in batch if part.source == source]
    if not matching_parts:
        return [f"source is not in batch: {source!r}"]

    if item["category"] not in CATEGORIES:
        errors.append(f"invalid category: {item['category']!r}")
    parsed_page = parse_line_number(item["expected_page"])
    if parsed_page is not None:
        item["expected_page"] = parsed_page
    if not isinstance(item["expected_page"], int):
        errors.append("expected_page is not an integer")
        return errors

    matching_pages = {
        page_number: text.splitlines()
        for part in matching_parts
        for page_number, text in part.pages
    }
    page_lines = matching_pages.get(item["expected_page"])
    if page_lines is None:
        errors.append(f"page {item['expected_page']} is not in the supplied source part")
    else:
        start = parse_line_number(item["evidence_start_line"])
        end = parse_line_number(item["evidence_end_line"])
        if start is None or end is None:
            errors.append("evidence line numbers must be integers")
        elif start < 1 or end < start or end > len(page_lines):
            errors.append(
                f"invalid evidence line range {start}-{end}; page has {len(page_lines)} lines"
            )
        elif end - start > 24:
            errors.append("evidence range exceeds 25 lines")
        else:
            evidence = "\n".join(page_lines[start - 1 : end])
            if len(normalize_text(evidence)) < 12:
                errors.append("evidence range is too short")
            answer_numbers = canonical_numbers(str(item["expected_answer"]))
            evidence_numbers = canonical_numbers(evidence)
            missing_numbers = [number for number in answer_numbers if number not in evidence_numbers]
            if missing_numbers:
                repaired_range: tuple[int, int] | None = None
                for added_lines in range(1, 5):
                    for lines_before in range(added_lines + 1):
                        candidate_start = max(1, start - lines_before)
                        candidate_end = min(
                            len(page_lines), end + (added_lines - lines_before)
                        )
                        candidate_evidence = "\n".join(
                            page_lines[candidate_start - 1 : candidate_end]
                        )
                        candidate_numbers = canonical_numbers(candidate_evidence)
                        if all(number in candidate_numbers for number in answer_numbers):
                            repaired_range = (candidate_start, candidate_end)
                            break
                    if repaired_range:
                        break
                if repaired_range:
                    item["evidence_start_line"], item["evidence_end_line"] = repaired_range
                else:
                    errors.append(
                        "the answer contains numbers absent from nearby evidence lines: "
                        + ", ".join(missing_numbers)
                    )

    query = normalize_text(str(item["query"]))
    forbidden = (
        "quel est l'objet",
        "quelle est la date de publication",
        "la présente circulaire",
        "cette circulaire",
        "circulaire",
        "cette note",
        "ce document",
        "selon la note",
        "selon l'article",
        "nouvelle disposition",
        "nouvelle règle",
        "nouveau taux",
        "décision du gouverneur",
        "ما هو موضوع",
        "ما هو تاريخ صدور",
        "هذا المنشور",
        "هذه المذكرة",
        "المنشور عدد",
        "المذكرة عدد",
    )
    relaxed_query_checks = os.environ.get("BCT_EVAL_RELAX_QUERY_CHECKS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not relaxed_query_checks and any(phrase in query for phrase in forbidden):
        errors.append("question uses forbidden generic or context-dependent wording")
    if len(query) < 20:
        errors.append("question is too short to be self-contained")
    if not str(item["expected_answer"]).strip():
        errors.append("answer is empty")
    elif "=" in str(item["expected_answer"]):
        errors.append("answer uses an equals sign instead of a complete sentence")
    return errors


def materialize_evidence(item: dict[str, Any], batch: list[SourcePart]) -> dict[str, Any]:
    page_lines = next(
        text.splitlines()
        for part in batch
        if part.source == item["expected_source"]
        for page_number, text in part.pages
        if page_number == item["expected_page"]
    )
    start = parse_line_number(item["evidence_start_line"])
    end = parse_line_number(item["evidence_end_line"])
    if start is None or end is None:
        raise ValueError("invalid evidence line numbers")
    evidence_lines = page_lines[start - 1 : end]
    while evidence_lines and not evidence_lines[0].strip():
        evidence_lines.pop(0)
    while evidence_lines and not evidence_lines[-1].strip():
        evidence_lines.pop()
    if len(evidence_lines) > 1 and re.fullmatch(r"\s*\d+\s*", evidence_lines[0]):
        evidence_lines.pop(0)
    enriched = dict(item)
    enriched["evidence_quote"] = "\n".join(evidence_lines).strip()
    return enriched


def call_model(client: Groq, batch: list[SourcePart], max_attempts: int = 3) -> list[dict[str, Any]]:
    user_content = render_batch(batch)
    feedback = ""
    last_rate_limit: RateLimitError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            request_options: dict[str, Any] = {}
            if MODEL.startswith("openai/gpt-oss"):
                request_options["reasoning_effort"] = os.environ.get(
                    "BCT_EVAL_REASONING_EFFORT", "medium"
                )
            elif MODEL.startswith("qwen/"):
                request_options["reasoning_effort"] = "none"
                request_options["include_reasoning"] = False
            if os.environ.get("BCT_EVAL_JSON_MODE", "").lower() in {"1", "true", "yes"}:
                request_options["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": ACTIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content + feedback},
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                **request_options,
            )
            response_text = response.choices[0].message.content
            object_start = response_text.find("{")
            object_end = response_text.rfind("}")
            if object_start < 0 or object_end < object_start:
                raise ValueError("response does not contain a JSON object")
            payload = json.loads(response_text[object_start : object_end + 1])
            items = payload.get("queries")
            if not isinstance(items, list):
                raise ValueError("response lacks a queries array")
            failures: list[tuple[int, list[str]]] = []
            for index, item in enumerate(items):
                item_errors = (
                    ["item is not an object"]
                    if not isinstance(item, dict)
                    else validate_item(item, batch)
                )
                if item_errors:
                    failures.append((index, item_errors))
            if not failures:
                covered_sources = {
                    item["expected_source"] for item in items if isinstance(item, dict)
                }
                missing_sources = {part.source for part in batch} - covered_sources
                if not missing_sources:
                    return [materialize_evidence(item, batch) for item in items]
                failures.extend(
                    (-1, [f"response omitted source: {source}"])
                    for source in sorted(missing_sources)
                )
            failed_indexes = {index for index, _ in failures}
            valid_items = [
                item
                for index, item in enumerate(items)
                if index not in failed_indexes and isinstance(item, dict)
            ]
            covered_sources = {item["expected_source"] for item in valid_items}
            required_sources = {part.source for part in batch}
            if valid_items and required_sources <= covered_sources:
                print(
                    f"accepting {len(valid_items)} validated items and dropping "
                    f"{len(failures)} invalid items",
                    flush=True,
                )
                return [materialize_evidence(item, batch) for item in valid_items]
            print(
                "validation failures: " + json.dumps(failures, ensure_ascii=False),
                flush=True,
            )
            print(
                "failed items: "
                + json.dumps(
                    [
                        items[index]
                        for index, _ in failures
                        if 0 <= index < len(items) and isinstance(items[index], dict)
                    ],
                    ensure_ascii=False,
                ),
                flush=True,
            )
            feedback = (
                "\n\nYour previous response failed validation. Regenerate the complete JSON response and fix these errors:\n"
                + json.dumps(failures, ensure_ascii=False)
            )
        except RateLimitError as error:
            last_rate_limit = error
            retry_match = re.search(
                r"try again in ([0-9.]+)(ms|s)", str(error), re.IGNORECASE
            )
            if retry_match:
                retry_value = float(retry_match.group(1))
                if retry_match.group(2).lower() == "ms":
                    retry_value /= 1000
                wait_seconds = min(90, max(1, int(retry_value) + 1))
            else:
                wait_seconds = min(90, 10 * attempt)
            print(f"rate limited; waiting {wait_seconds}s: {error}", flush=True)
            time.sleep(wait_seconds)
        except (APIError, json.JSONDecodeError, ValueError) as error:
            if attempt == max_attempts:
                raise
            wait_seconds = min(30, 3 * attempt)
            print(f"attempt {attempt} failed; waiting {wait_seconds}s: {error}", flush=True)
            time.sleep(wait_seconds)
    if last_rate_limit is not None:
        raise last_rate_limit
    raise RuntimeError("model did not produce a valid batch")


def stable_id(item: dict[str, Any], sequence: int) -> str:
    stem = Path(item["expected_source"]).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"{stem}_{item['category']}_{sequence:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--year", help="Only process documents whose parent directory contains this year")
    parser.add_argument("--source", action="append", help="Only process an exact PDF filename; repeatable")
    parser.add_argument("--exclude-source", action="append", help="Skip an exact PDF filename; repeatable")
    parser.add_argument("--language", choices=("fr", "ar"), help="Only process one source language")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_dotenv(dotenv_path=ROOT / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is missing", file=sys.stderr)
        return 2

    paths = sorted(DOCUMENTS.rglob("*.pdf"), key=lambda path: str(path).casefold())
    if args.year:
        paths = [path for path in paths if args.year in path.parent.name]
    if args.source:
        requested_sources = set(args.source)
        paths = [path for path in paths if path.name in requested_sources]
    if args.exclude_source:
        excluded_sources = set(args.exclude_source)
        paths = [path for path in paths if path.name not in excluded_sources]
    if args.language:
        paths = [path for path in paths if language_for(path) == args.language]
    if args.limit:
        paths = paths[: args.limit]

    parts = [part for path in paths for part in split_document(path)]
    batches = make_batches(parts)
    existing: list[dict[str, Any]] = []
    completed_signatures: set[tuple[str, tuple[int, ...]]] = set()
    failed_parts: list[dict[str, Any]] = []
    failed_signatures: set[tuple[str, tuple[int, ...]]] = set()
    if args.output.exists():
        checkpoint = json.loads(args.output.read_text(encoding="utf-8"))
        existing = checkpoint.get("queries", [])
        completed_signatures = {
            (entry["source"], tuple(entry["pages"]))
            for entry in checkpoint.get("completed_parts", [])
        }
        failed_parts = checkpoint.get("failed_parts", [])
        failed_signatures = {
            (entry["source"], tuple(entry["pages"])) for entry in failed_parts
        }

    client = Groq(
        api_key=os.environ["GROQ_API_KEY"],
        timeout=float(os.environ.get("BCT_EVAL_REQUEST_TIMEOUT", "90")),
        max_retries=0,
    )
    for batch_index, batch in enumerate(batches, start=1):
        pending = [
            part
            for part in batch
            if (part.source, tuple(number for number, _ in part.pages)) not in completed_signatures
            and (part.source, tuple(number for number, _ in part.pages)) not in failed_signatures
        ]
        if not pending:
            continue
        print(
            f"batch {batch_index}/{len(batches)}: "
            + ", ".join(f"{part.source}({part.requested_count})" for part in pending),
            flush=True,
        )
        try:
            generated = call_model(client, pending, max_attempts=MAX_ATTEMPTS)
        except RateLimitError as error:
            print(f"provider quota exhausted; checkpoint preserved: {error}", flush=True)
            return 3
        except Exception as error:
            print(f"batch failed and will require manual review: {error}", flush=True)
            for part in pending:
                signature = (part.source, tuple(number for number, _ in part.pages))
                failed_signatures.add(signature)
                failed_parts.append(
                    {
                        "source": part.source,
                        "pages": list(signature[1]),
                        "error": str(error),
                    }
                )
        else:
            existing.extend(generated)
            for part in pending:
                completed_signatures.add((part.source, tuple(number for number, _ in part.pages)))
        checkpoint = {
            "model": MODEL,
            "queries": existing,
            "completed_parts": [
                {"source": source, "pages": list(pages)}
                for source, pages in sorted(completed_signatures)
            ],
            "failed_parts": failed_parts,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"saved {len(existing)} questions to {args.output}", flush=True)

    counts: dict[str, int] = {}
    final: list[dict[str, Any]] = []
    for item in existing:
        source = item["expected_source"]
        counts[source] = counts.get(source, 0) + 1
        final.append(
            {
                "id": stable_id(item, counts[source]),
                "query": item["query"].strip(),
                "language": language_for(Path(source)),
                "category": item["category"],
                "relevant": True,
                "expected_source": source,
                "expected_page": item["expected_page"],
                "expected_answer": item["expected_answer"].strip(),
                "evidence_quote": item["evidence_quote"].strip(),
                "evidence_method": "text_extraction",
            }
        )
    args.output.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures_output = args.output.with_suffix(".failures.json")
    failures_output.write_text(
        json.dumps(failed_parts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"complete: {len(final)} questions; {len(failed_parts)} parts require manual review",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
