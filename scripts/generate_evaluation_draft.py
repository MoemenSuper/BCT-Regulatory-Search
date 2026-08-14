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


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "documents"
DEFAULT_OUTPUT = Path(os.environ.get("TEMP", ROOT)) / "bct_evaluation_draft.json"
MODEL = "openai/gpt-oss-120b"
MAX_BATCH_CHARS = 55_000
MAX_DOCUMENTS_PER_BATCH = 8

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

Return one JSON object with a `queries` array. Use the requested number of questions for each source. Write each question and answer in the document's language.

Each item must contain exactly these fields:
- expected_source: exact filename from a SOURCE marker
- query: a natural, self-contained user question
- category: one of amount_or_rate, deadline_or_duration, eligibility_or_scope, required_action, prohibition_or_limit, exception_or_condition, definition, procedure_or_documents, reporting_or_control, effective_date, other_operational_rule
- expected_page: the 1-based PDF page number from a PAGE marker
- expected_answer: a concise, complete answer
- evidence_quote: one contiguous supporting passage copied from that same page

Hard rules:
1. A bank employee, compliance officer, company, farmer, exporter, customer, or researcher should plausibly ask the question without knowing the document number.
2. Test operative facts: amounts, rates, deadlines, durations, eligibility, limits, exceptions, definitions, required actions, procedures, documents, reporting, or controls.
3. Never ask for a document's title, general subject, publication date, circular number, note number, article number, signatory, or legal citations in the preamble.
4. Never use context-dependent wording such as `cette circulaire`, `la présente circulaire`, `ce document`, `cette note`, `هذا المنشور`, or `هذه المذكرة` unless the question also names the concrete regulated activity.
5. Keep any date, sector, transaction type, stakeholder, or other scope needed to prevent an old amendment from sounding like an unqualified current rule.
6. One cited PDF page must fully support the answer. Do not combine facts from different pages.
7. Copy evidence_quote from one page. You may normalize whitespace, but do not paraphrase the evidence.
8. Preserve every number, unit, currency, percentage, date, negation, condition, exception, table column, and dry/irrigated distinction. Label every number in the answer. Never return an unexplained number or range.
9. Do not infer legal advice, current validity, supersession, or facts absent from the supplied text.
10. Avoid duplicate facts and questions that differ only by wording.
11. Prefer questions with enough concrete wording to retrieve the source, while avoiding the source identifier as a shortcut.
12. If a source lacks enough reliable operative facts for the requested count, return fewer questions rather than inventing content.

Quality examples:
- `Je cultive des tomates de saison. Quel est le plafond du crédit par hectare et à quelle date arrive-t-il à échéance ?`
- `Une banque doit-elle accorder automatiquement la totalité du barème prévu pour un crédit agricole ?`
- `ما هي المدة القصوى لإعادة جدولة أصل الدين الفلاحي، وما النسبة الواجب دفعها عند تقديم مطلب التسوية؟`

Bad examples:
- `Quel est l'objet de la circulaire n°2021-07 ?`
- `Que prévoit cette circulaire ?`
- `Quel est le barème pour les amandiers ?` when the table distinguishes dry and irrigated farming.
"""


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
        return 2
    if page_count <= 3:
        return 3
    if page_count <= 10:
        return 4
    if page_count <= 25:
        return 6
    return 8


def split_document(path: Path) -> list[SourcePart]:
    document = fitz.open(path)
    pages = tuple((index + 1, page.get_text("text").strip()) for index, page in enumerate(document))
    usable_pages = tuple((number, text) for number, text in pages if text)
    if not usable_pages:
        return []

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
    sections: list[str] = []
    for part in batch:
        sections.append(
            f"[[SOURCE {part.source} | LANGUAGE {part.language} | REQUESTED QUESTIONS {part.requested_count}]]"
        )
        for page_number, text in part.pages:
            sections.append(f"[[SOURCE {part.source} | PDF PAGE {page_number}]]\n{text}")
    return "\n\n".join(sections)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("’", "'").replace("‘", "'").replace("ـ", "")
    return re.sub(r"\s+", " ", value).strip().casefold()


def validate_item(item: dict[str, Any], batch: list[SourcePart]) -> list[str]:
    errors: list[str] = []
    required = {"expected_source", "query", "category", "expected_page", "expected_answer", "evidence_quote"}
    missing = required - set(item)
    if missing:
        return [f"missing fields: {sorted(missing)}"]

    source = item["expected_source"]
    matching_parts = [part for part in batch if part.source == source]
    if not matching_parts:
        return [f"source is not in batch: {source!r}"]

    if item["category"] not in CATEGORIES:
        errors.append(f"invalid category: {item['category']!r}")
    if not isinstance(item["expected_page"], int):
        errors.append("expected_page is not an integer")
        return errors

    matching_pages = {
        page_number: text
        for part in matching_parts
        for page_number, text in part.pages
    }
    page_text = matching_pages.get(item["expected_page"])
    if page_text is None:
        errors.append(f"page {item['expected_page']} is not in the supplied source part")
    else:
        evidence = normalize_text(str(item["evidence_quote"]))
        if len(evidence) < 12:
            errors.append("evidence quote is too short")
        elif evidence not in normalize_text(page_text):
            errors.append("normalized evidence quote does not occur on the cited page")

    query = normalize_text(str(item["query"]))
    forbidden = (
        "quel est l'objet",
        "quelle est la date de publication",
        "la présente circulaire",
        "cette circulaire",
        "cette note",
        "ce document",
        "ما هو موضوع",
        "ما هو تاريخ صدور",
        "هذا المنشور",
        "هذه المذكرة",
    )
    if any(phrase in query for phrase in forbidden):
        errors.append("question uses forbidden generic or context-dependent wording")
    if len(query) < 20:
        errors.append("question is too short to be self-contained")
    if not str(item["expected_answer"]).strip():
        errors.append("answer is empty")
    return errors


def call_model(client: Groq, batch: list[SourcePart], max_attempts: int = 6) -> list[dict[str, Any]]:
    user_content = render_batch(batch)
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                reasoning_effort="medium",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content + feedback},
                ],
                max_completion_tokens=12_000,
            )
            payload = json.loads(response.choices[0].message.content)
            items = payload.get("queries")
            if not isinstance(items, list):
                raise ValueError("response lacks a queries array")
            failures = [
                (index, validate_item(item, batch))
                for index, item in enumerate(items)
                if not isinstance(item, dict) or validate_item(item, batch)
            ]
            if not failures:
                return items
            feedback = (
                "\n\nYour previous response failed validation. Regenerate the complete JSON response and fix these errors:\n"
                + json.dumps(failures, ensure_ascii=False)
            )
        except RateLimitError as error:
            wait_seconds = min(90, 10 * attempt)
            print(f"rate limited; waiting {wait_seconds}s: {error}", flush=True)
            time.sleep(wait_seconds)
        except (APIError, json.JSONDecodeError, ValueError) as error:
            if attempt == max_attempts:
                raise
            wait_seconds = min(30, 3 * attempt)
            print(f"attempt {attempt} failed; waiting {wait_seconds}s: {error}", flush=True)
            time.sleep(wait_seconds)
    raise RuntimeError("model did not produce a valid batch")


def stable_id(item: dict[str, Any], sequence: int) -> str:
    stem = Path(item["expected_source"]).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"{stem}_{item['category']}_{sequence:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--year", help="Only process documents whose parent directory contains this year")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    load_dotenv(dotenv_path=ROOT / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is missing", file=sys.stderr)
        return 2

    paths = sorted(DOCUMENTS.rglob("*.pdf"), key=lambda path: str(path).casefold())
    if args.year:
        paths = [path for path in paths if args.year in path.parent.name]
    if args.limit:
        paths = paths[: args.limit]

    parts = [part for path in paths for part in split_document(path)]
    batches = make_batches(parts)
    existing: list[dict[str, Any]] = []
    completed_signatures: set[tuple[str, tuple[int, ...]]] = set()
    if args.output.exists():
        checkpoint = json.loads(args.output.read_text(encoding="utf-8"))
        existing = checkpoint.get("queries", [])
        completed_signatures = {
            (entry["source"], tuple(entry["pages"]))
            for entry in checkpoint.get("completed_parts", [])
        }

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    for batch_index, batch in enumerate(batches, start=1):
        pending = [
            part
            for part in batch
            if (part.source, tuple(number for number, _ in part.pages)) not in completed_signatures
        ]
        if not pending:
            continue
        print(
            f"batch {batch_index}/{len(batches)}: "
            + ", ".join(f"{part.source}({part.requested_count})" for part in pending),
            flush=True,
        )
        generated = call_model(client, pending)
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
            }
        )
    args.output.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {len(final)} questions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
