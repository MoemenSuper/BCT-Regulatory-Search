"""Query-class → which document kinds may prove claims.

Classification never rejects the question; it only chooses grounding permission.
uncertain / classifier failure → regulatory grounding (fail closed on authority).
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from document_authority import DocKind

QueryClass = Literal[
    "regulatory_rule",
    "statistical_fact",
    "internal_procedure",
    "mixed",
    "uncertain",
]

QUERY_CLASSES: tuple[str, ...] = (
    "regulatory_rule",
    "statistical_fact",
    "internal_procedure",
    "mixed",
    "uncertain",
)


class QueryAuthorityClass(BaseModel):
    query_class: QueryClass
    confidence: Literal["high", "low"] = "low"
    rationale: str = Field(default="", max_length=240)


# Curated gold bank: hard negatives first. Dynamic demos pick by token overlap.
_GOLD: tuple[tuple[str, QueryClass, str], ...] = (
    (
        "Quel est le plafond de financement pour les importations non prioritaires selon la circulaire 2026-04 ?",
        "regulatory_rule",
        "Named circulaire + plafond = obligation/limit, not a bulletin series.",
    ),
    (
        "Quel était le volume d'exportations de la Tunisie en 2024 ?",
        "statistical_fact",
        "Asks for an observed volume/time series, not a legal plafond.",
    ),
    (
        "Quelle est la procédure interne pour valider un dossier de change ?",
        "internal_procedure",
        "Explicit internal procedure wording.",
    ),
    (
        "Selon la circulaire 2016-01, quelle procédure suivre en interne pour ouvrir le marché ?",
        "mixed",
        "Binding instrument plus internal how-to.",
    ),
    (
        "Quel est le taux applicable aux opérations de change ?",
        "regulatory_rule",
        "Hard negative: 'taux' here is a regulatory rate/plafond, not a bulletin statistic.",
    ),
    (
        "Montrez l'évolution du taux directeur sur le graphique du bulletin 2023.",
        "statistical_fact",
        "Bulletin + graphique/évolution = statistical chart fact.",
    ),
    (
        "Combien d'agences bancaires étaient recensées dans les statistiques 2022 ?",
        "statistical_fact",
        "Recensement / statistiques = observed count.",
    ),
    (
        "Le mémo interne précise-t-il les pièces à joindre au dossier ?",
        "internal_procedure",
        "Mémo interne + pièces = internal checklist.",
    ),
    (
        "La note 2024-03 impose-t-elle un délai de déclaration ?",
        "regulatory_rule",
        "Note réglementaire + impose/délai = primary rule.",
    ),
    (
        "Comparez l'obligation de la circulaire et le chiffre publié dans le bulletin.",
        "mixed",
        "Needs both a rule and a published statistic.",
    ),
    (
        "What is the maximum financing line under circular 2025-13?",
        "regulatory_rule",
        "English circular ceiling question.",
    ),
    (
        "What was Tunisia's inflation rate in the 2024 statistical bulletin?",
        "statistical_fact",
        "English bulletin time-series fact.",
    ),
    (
        "ما هو السقف المنصوص عليه في المنشور 2026-04؟",
        "regulatory_rule",
        "Arabic circular ceiling.",
    ),
    (
        "ما هو حجم الصادرات حسب النشرة الإحصائية؟",
        "statistical_fact",
        "Arabic statistical bulletin volume.",
    ),
    (
        "c'est quoi ce truc",
        "uncertain",
        "Underspecified; do not reject — default regulatory grounding.",
    ),
)


def allowed_doc_kinds(query_class: object | None) -> frozenset[DocKind]:
    value = str(query_class or "uncertain").strip().casefold()
    if value == "statistical_fact":
        return frozenset({"statistical", "regulatory"})
    if value == "internal_procedure":
        return frozenset({"internal", "regulatory"})
    if value == "mixed":
        return frozenset({"regulatory", "statistical", "internal"})
    # regulatory_rule and uncertain (and unknown): primary regulatory only
    return frozenset({"regulatory"})


def evidence_kind_allowed(query_class: object | None, *, doc_kind: object | None) -> bool:
    from document_authority import normalize_doc_kind

    return normalize_doc_kind(doc_kind) in allowed_doc_kinds(query_class)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w\u0600-\u06FF]+", text.casefold()) if len(token) > 2}


def select_demonstrations(question: str, *, k: int = 6) -> list[tuple[str, QueryClass, str]]:
    """Cheap dynamic few-shot: gold examples with highest token overlap, class-diverse."""
    q = _tokens(question)
    scored = []
    for example, label, note in _GOLD:
        overlap = len(q & _tokens(example))
        scored.append((overlap, example, label, note))
    scored.sort(key=lambda row: (-row[0], row[1]))
    chosen: list[tuple[str, QueryClass, str]] = []
    seen_labels: set[str] = set()
    # First pass: best per class for coverage of hard boundaries.
    for overlap, example, label, note in scored:
        if label in seen_labels:
            continue
        chosen.append((example, label, note))
        seen_labels.add(label)
        if len(chosen) >= min(k, len(QUERY_CLASSES)):
            break
    # Fill remaining slots by raw score.
    for overlap, example, label, note in scored:
        if (example, label, note) in chosen:
            continue
        chosen.append((example, label, note))
        if len(chosen) >= k:
            break
    return chosen


def _prompt(question: str, demos: list[tuple[str, QueryClass, str]]) -> str:
    lines = [
        "Classify the user question for claim-grounding authority only.",
        "Never refuse. Never answer the regulatory question. Output JSON only.",
        "Classes:",
        "- regulatory_rule: obligations, délais, plafonds, régimes, selon circulaire/note",
        "- statistical_fact: observed volumes, rates in bulletins, chart/table figures, recensements",
        "- internal_procedure: mémo/procédure interne, checklists, internal workflow",
        "- mixed: needs both a binding rule and a stat/internal fact",
        "- uncertain: underspecified; still classify uncertain (system will ground as regulatory)",
        "Hard negatives: 'taux' alone often regulatory (plafond); bulletin/graphique/évolution → statistical.",
        "Examples:",
    ]
    for index, (example, label, note) in enumerate(demos, 1):
        lines.append(f"{index}. Q: {example}")
        lines.append(f"   A: {{\"query_class\":\"{label}\",\"confidence\":\"high\",\"rationale\":\"{note}\"}}")
    lines.append(f"Question: {question}")
    lines.append('JSON: {"query_class":"...","confidence":"high|low","rationale":"short"}')
    return "\n".join(lines)


def default_query_authority() -> dict:
    return QueryAuthorityClass(
        query_class="uncertain",
        confidence="low",
        rationale="classifier_unavailable",
    ).model_dump(mode="json")


def classify_query_authority(llm, question: str) -> dict:
    """Return a QueryAuthorityClass dict. Fail closed to uncertain (regulatory grounding)."""
    demos = select_demonstrations(question)
    prompt = _prompt(question, demos)
    try:
        raw = llm.invoke(prompt)
        content = getattr(raw, "content", raw)
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content
            )
        text = str(content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Prefer first JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        payload = json.loads(text)
        parsed = QueryAuthorityClass.model_validate(payload)
        if parsed.query_class not in QUERY_CLASSES:
            return default_query_authority()
        return parsed.model_dump(mode="json")
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
        return default_query_authority()
    except Exception:
        return default_query_authority()
