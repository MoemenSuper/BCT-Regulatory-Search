"""Deterministic post-retrieval answer-safety checkpoint.

The runtime decision surface is deliberately limited to the question, frozen
query state, retrieved evidence, structured claims, and citations. Evaluation
gold is used only after each safety decision has been made.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.document_identity_candidate_experiment import (
    parse_query_identity,
    parse_source_identity,
)
from experiments.gold_evidence_answer_experiment import automatic_answer_audit
from experiments.numeric_fidelity_stress import critical_identifiers
from experiments.retrieved_context_answer_experiment import (
    _query_states,
    retrieved_structured_diagnostics,
)
from experiments.structured_answer_experiment import _aggregate


EXPERIMENT_ID = "post-retrieval-answer-safety-development-v1"
ANSWER_SUITE_SHA256 = (
    "5E3D840B6DF8FDF3046AA2A5A67B84DF8CC0E0E61D4EC810E5D51446C89917A1"
)
COMBINED_CHECKPOINT_A_SHA256 = (
    "5CBAAC050DADBB49174BE0C9F829219F51BEE9E5EB9E25668A561CBA0562067C"
)
NEGATIVE_QUERY_STATE_SHA256 = (
    "C846D913DDCFB996392A5F48A194FC62ACF60152429B9A9D9473E6FA8A542BCE"
)
RELEVANT_QUERY_STATE_SHA256 = (
    "B14D820A398C6B131713714B4DE9985A6E8F89B78374FF4836BC58F9A88317F4"
)

_CONTEXT_LABEL = re.compile(
    r"(?i)\b(?:à titre de comparaison|a titre de comparaison|comparativement|"
    r"contexte|pour comparaison|by comparison|for context)\b|"
    r"للمقارنة|على سبيل المقارنة|للسياق"
)
_CURRENT_QUERY = re.compile(
    r"(?i)\b(?:aujourd['’]hui|actuellement|à ce jour|a ce jour|"
    r"encore applicable|le plus récent|la plus récente|latest|currently|"
    r"in force)\b|حالياً|حاليا|الآن|الأحدث"
)
_IN_FORCE_QUERY = re.compile(r"(?i)\ben vigueur\b|ساري(?:ة)? المفعول")
_EXPLICIT_CURRENT_SUPPORT = re.compile(
    r"(?i)\b(?:demeure|reste|est toujours|sont toujours|encore)\s+en vigueur\b|"
    r"\ben vigueur\s+(?:au|à la date du|a la date du|à ce jour|a ce jour)\b|"
    r"\bactuellement\s+en vigueur\b|"
    r"ما ?زال(?:ت)? ساري(?:ة)? المفعول|ساري(?:ة)? المفعول حتى|"
    r"ساري(?:ة)? المفعول حالياً"
)
_LEGAL_ROLE_INVERSION = re.compile(
    r"(?i)\b(?:abrog\w*|décid\w*|decid\w*|fix\w*|ordonn\w*)\b[^.]{0,80}"
    r"\bpar\s+(?:l['’]\s*)?(?:avis|comité|comite)\b"
)
_MERE_ADVISORY_RECITAL = re.compile(
    r"(?i)\bvu\s+(?:l['’]\s*)?avis\b[\s\S]{0,160}\b(?:décide|decide)\b"
)
_INCOMPLETE_ENDING = re.compile(
    r"(?i)(?::|：|comme suit|suivantes?|التالية|كما يلي)\s*$"
)
_FRENCH_REQUIREMENT = re.compile(
    r"(?i)\bqu(?:el|els|elle|elles)\s+([a-zà-ÿ][a-zà-ÿ'’-]*)"
)
_ARABIC_REQUIREMENT = re.compile(r"(?:ما\s+هي|ما\s+هو|وما|ومتى|وكم|وكيف|وأي)\s+([^\s؟?،,]+)")
_DOMAIN_TERMS = re.compile(
    r"(?i)\b(?:banque|bancaire|billet|allocation|transfert|circulaire|"
    r"réglement|reglement|chèque|cheque|dinar|franc|devise)\w*\b|"
    r"بنك|مصرف|ورقة نقدية|أوراق نقدية|فرنك|تداول|سحب|تحويل|منشور|مذكرة|شيك"
)
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_STOPWORDS = {
    "avec",
    "ainsi",
    "cette",
    "dans",
    "depuis",
    "doit",
    "elle",
    "elles",
    "entre",
    "pour",
    "sont",
    "sur",
    "tous",
    "toutes",
    "une",
    "the",
    "and",
    "from",
    "that",
    "this",
    "على",
    "إلى",
    "التي",
    "هذا",
    "هذه",
    "يجب",
}


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text)).casefold()
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in _WORD.findall(_fold(text))
        if word not in _STOPWORDS and len(word) >= 3
    }


def _numeric_literals(text: str) -> set[str]:
    characters = []
    for character in unicodedata.normalize("NFKC", str(text)):
        if character.isdigit():
            characters.append(str(unicodedata.digit(character)))
        elif character == "\u066b":
            characters.append(".")
        elif character == "\u066c":
            continue
        else:
            characters.append(character)
    normalized = "".join(characters)
    normalized = re.sub(
        r"(?<=\d)[\s\u00a0\u202f.,](?=\d{3}(?:\D|$))", "", normalized
    )
    normalized = normalized.replace(",", ".")
    return set(re.findall(r"\d+(?:\.\d+)?", normalized))


def _safe_abstention(language: str, reason: str) -> dict[str, Any]:
    answer = (
        "لا تكفي الأدلة المسترجعة لإثبات هذه النقطة بصورة موثوقة، لذلك لا يمكن تقديم إجابة مؤكدة."
        if language == "ar"
        else "Les preuves récupérées ne permettent pas d'établir ce point de façon fiable; je ne peux donc pas donner une réponse affirmative."
    )
    return {
        "status": "insufficient_evidence",
        "answer": answer,
        "claims": [],
        "citations": [],
        "safety_reason": reason,
    }


def _clarification(language: str, missing_detail: str) -> dict[str, Any]:
    detail = missing_detail.strip()
    if language == "fr":
        detail = {
            "type of transfer": "type de transfert",
            "instrument type": "type d'instrument",
            "document type": "type de document",
        }.get(detail.casefold(), detail)
    if language == "ar":
        answer = "يرجى تحديد الأداة أو العملية التنظيمية المقصودة"
        if detail:
            answer += f"، وخصوصاً: {detail}"
        answer += "."
    else:
        answer = "Veuillez préciser l'instrument ou l'opération réglementaire visée"
        if detail:
            answer += f", notamment le {detail}"
        answer += "."
    return {
        "status": "clarification_needed",
        "answer": answer,
        "claims": [],
        "citations": [],
        "safety_reason": "multiple_plausible_instruments",
    }


def _identity_key(identity: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if identity is None:
        return None
    return identity.get("kind"), identity.get("year"), identity.get("number")


def _identity_matches_requested(
    source_identity: dict[str, Any] | None, requested: dict[str, Any]
) -> bool:
    if source_identity is None:
        return False
    return all(
        requested.get(field) is None or source_identity.get(field) == requested[field]
        for field in ("kind", "year", "number")
    )


def _linked_evidence(
    claim: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]] | None:
    evidence_ids = claim.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        return None
    if any(evidence_id not in evidence_by_id for evidence_id in evidence_ids):
        return None
    return [evidence_by_id[evidence_id] for evidence_id in evidence_ids]


def _adequate_linked_evidence(claim_text: str, linked: list[dict[str, Any]]) -> bool:
    evidence_text = "\n".join(
        f"{item.get('source', '')}\n{item.get('text', '')}" for item in linked
    )
    claim_numbers = _numeric_literals(claim_text)
    if not claim_numbers.issubset(_numeric_literals(evidence_text)):
        return False
    claim_identifiers = critical_identifiers(claim_text)
    if not claim_identifiers.issubset(critical_identifiers(evidence_text)):
        return False
    claim_words = _content_words(claim_text)
    evidence_words = _content_words(evidence_text)
    if not claim_words:
        return bool(claim_numbers)
    overlap = claim_words & evidence_words
    return bool(overlap)


def _asks_current_status(question: str) -> bool:
    if _CURRENT_QUERY.search(question):
        return True
    if not _IN_FORCE_QUERY.search(question):
        return False
    folded = _fold(question)
    has_historical_anchor = bool(re.search(r"\b(?:19|20)\d{2}\b", folded)) or any(
        marker in folded for marker in ("apres", "avant", "depuis le")
    )
    return not has_historical_anchor


def _material_requirements(question: str) -> set[str]:
    matches = [*_FRENCH_REQUIREMENT.findall(question), *_ARABIC_REQUIREMENT.findall(question)]
    requirements = set()
    for match in matches:
        value = _fold(match).strip("'’- ")
        if value.endswith("es") and len(value) > 5:
            value = value[:-2]
        elif value.endswith("s") and len(value) > 4:
            value = value[:-1]
        requirements.add(value)
    return requirements


def _missing_material_requirements(question: str, response: dict[str, Any]) -> list[str]:
    requirements = _material_requirements(question)
    if len(requirements) < 2:
        return []
    response_text = _fold(
        "\n".join(
            [str(response.get("answer", ""))]
            + [str(claim.get("text", "")) for claim in response.get("claims", [])]
        )
    )
    return sorted(requirement for requirement in requirements if requirement not in response_text)


def _answer_separately_labels_claim_identity(
    *,
    answer: str,
    claim_text: str,
    year: int,
    number: int,
    require_number: bool,
) -> bool:
    claim_numbers = _numeric_literals(claim_text)
    clauses = re.split(
        r"(?<!\d),(?!\d)|[;\n]|\b(?:et|and)\b", _fold(answer)
    )
    for clause in clauses:
        if not re.search(rf"(?<!\d){year}(?!\d)", clause):
            continue
        if require_number and not re.search(rf"(?<!\d){number}(?!\d)", clause):
            continue
        if claim_numbers.issubset(_numeric_literals(clause)):
            return True
    return False


def compose_incomplete_answer_from_claims(response: dict[str, Any]) -> dict[str, Any]:
    """Complete a visibly unfinished answer using only its existing claim text."""
    if response.get("status") != "answered" or not _INCOMPLETE_ENDING.search(
        str(response.get("answer", ""))
    ):
        return response
    claims = [str(claim.get("text", "")).strip() for claim in response.get("claims", [])]
    if not claims or any(not claim for claim in claims):
        return response
    completed = copy.deepcopy(response)
    completed["answer"] = str(response["answer"]).rstrip() + "\n- " + "\n- ".join(claims)
    return completed


def _validation_failure(language: str, action: str, **details: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return _safe_abstention(language, action), {"action": action, **details}


def apply_answer_safety(
    *,
    question: str,
    language: str,
    response: dict[str, Any],
    evidence: list[dict[str, Any]],
    query_state: dict[str, str],
    generated_status_before_query_state: str | None = None,
    answer_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply runtime-observable answer guards and return the action audit."""
    if response.get("status") != "answered":
        false_preemption = (
            response.get("status") == "out_of_scope"
            and generated_status_before_query_state == "insufficient_evidence"
            and answer_path == "retrieved_top5_abstention_then_query_state"
            and bool(_DOMAIN_TERMS.search(question))
        )
        if false_preemption:
            return _validation_failure(
                language,
                "correct_false_query_state_preemption",
                prior_status="out_of_scope",
            )
        return response, {"action": "keep_nonanswer"}

    evidence_by_id = {str(item.get("evidence_id")): item for item in evidence}
    evidence_identities = {
        key
        for item in evidence
        if (key := _identity_key(parse_source_identity(str(item.get("source", "")))))
        is not None
    }
    requested = parse_query_identity(question)

    if (
        query_state.get("ambiguity") == "missing_discriminating_detail"
        and requested is None
        and len(evidence_identities) > 1
    ):
        return (
            _clarification(language, str(query_state.get("missing_detail", ""))),
            {
                "action": "request_clarification_multiple_instruments",
                "plausible_instrument_count": len(evidence_identities),
            },
        )

    if _asks_current_status(question):
        all_evidence_text = "\n".join(str(item.get("text", "")) for item in evidence)
        if not _EXPLICIT_CURRENT_SUPPORT.search(all_evidence_text):
            return _validation_failure(
                language, "fail_closed_current_status_not_established"
            )

    claims = response.get("claims")
    citations = response.get("citations")
    if not isinstance(claims, list) or not claims or not isinstance(citations, list):
        return _validation_failure(language, "fail_closed_invalid_structured_links")

    cited_ids: set[str] = set()
    for citation in citations:
        evidence_id = str(citation.get("evidence_id", ""))
        linked = evidence_by_id.get(evidence_id)
        if (
            linked is None
            or Path(str(citation.get("source", ""))).name.casefold()
            != Path(str(linked.get("source", ""))).name.casefold()
            or int(citation.get("page", -10**9)) != int(linked.get("page", -1))
        ):
            return _validation_failure(language, "fail_closed_citation_evidence_mismatch")
        cited_ids.add(evidence_id)

    requested_primary_found = False
    all_linked_ids: set[str] = set()
    claim_version_rows: list[dict[str, Any]] = []
    for claim in claims:
        claim_text = str(claim.get("text", ""))
        linked = _linked_evidence(claim, evidence_by_id)
        if linked is None:
            return _validation_failure(language, "fail_closed_invalid_structured_links")
        linked_ids = {str(item["evidence_id"]) for item in linked}
        all_linked_ids |= linked_ids
        if not linked_ids.issubset(cited_ids):
            return _validation_failure(language, "fail_closed_uncited_claim_evidence")

        linked_identity_keys = {
            key
            for item in linked
            if (key := _identity_key(parse_source_identity(str(item.get("source", "")))))
            is not None
        }
        independently_supported = False
        if len(linked_identity_keys) > 1:
            claim_literals = _numeric_literals(claim_text) | critical_identifiers(
                claim_text
            )
            independently_supported = bool(claim_literals) and all(
                claim_literals.issubset(
                    _numeric_literals(f"{item.get('source', '')}\n{item.get('text', '')}")
                    | critical_identifiers(
                        f"{item.get('source', '')}\n{item.get('text', '')}"
                    )
                )
                for item in linked
            )
            if not independently_supported:
                return _validation_failure(language, "fail_closed_mixed_instrument_claim")
        claim_version_rows.append(
            {
                "claim_text": claim_text,
                "identity_keys": linked_identity_keys,
                "cross_version_corroboration": independently_supported,
            }
        )

        if requested is not None:
            matches = all(
                _identity_matches_requested(
                    parse_source_identity(str(item.get("source", ""))), requested
                )
                for item in linked
            )
            if matches:
                requested_primary_found = True
            elif not _CONTEXT_LABEL.search(claim_text):
                return _validation_failure(
                    language, "fail_closed_requested_instrument_mismatch"
                )

        linked_text = "\n".join(str(item.get("text", "")) for item in linked)
        if _LEGAL_ROLE_INVERSION.search(claim_text) and _MERE_ADVISORY_RECITAL.search(
            linked_text
        ):
            return _validation_failure(language, "fail_closed_legal_role_inversion")
        if not _adequate_linked_evidence(claim_text, linked):
            return _validation_failure(language, "fail_closed_inadequate_linked_evidence")

    response_identity_keys = {
        identity_key
        for row in claim_version_rows
        for identity_key in row["identity_keys"]
    }
    if len(response_identity_keys) > 1:
        response_years = {identity_key[1] for identity_key in response_identity_keys}
        for row in claim_version_rows:
            identity_keys = row["identity_keys"]
            if len(identity_keys) > 1 and row["cross_version_corroboration"]:
                continue
            if len(identity_keys) != 1:
                return _validation_failure(
                    language, "fail_closed_unlabelled_cross_claim_version_mixing"
                )
            kind, year, number = next(iter(identity_keys))
            claim_folded = _fold(row["claim_text"])
            year_labelled = bool(re.search(rf"(?<!\d){year}(?!\d)", claim_folded))
            number_labelled = bool(
                re.search(rf"(?<!\d){number}(?!\d)", claim_folded)
            )
            directly_labelled = year_labelled and (
                len(response_years) > 1 or number_labelled
            )
            answer_labelled = _answer_separately_labels_claim_identity(
                answer=str(response.get("answer", "")),
                claim_text=row["claim_text"],
                year=year,
                number=number,
                require_number=len(response_years) == 1,
            )
            if not directly_labelled and not answer_labelled:
                return _validation_failure(
                    language,
                    "fail_closed_unlabelled_cross_claim_version_mixing",
                    unlabelled_identity={"kind": kind, "year": year, "number": number},
                )

    if requested is not None and not requested_primary_found:
        return _validation_failure(language, "fail_closed_missing_requested_instrument_claim")

    answer_text = str(response.get("answer", ""))
    answer_literals = _numeric_literals(answer_text) | critical_identifiers(answer_text)
    linked_text = "\n".join(
        f"{evidence_by_id[evidence_id].get('source', '')}\n"
        f"{evidence_by_id[evidence_id].get('text', '')}"
        for evidence_id in all_linked_ids
    )
    linked_literals = _numeric_literals(linked_text) | critical_identifiers(linked_text)
    if not answer_literals.issubset(linked_literals):
        return _validation_failure(language, "fail_closed_unsupported_numeric_literal")

    completed = compose_incomplete_answer_from_claims(response)
    if _INCOMPLETE_ENDING.search(str(completed.get("answer", ""))):
        return _validation_failure(language, "fail_closed_incomplete_answer_without_claims")

    missing_requirements = _missing_material_requirements(question, completed)
    if missing_requirements:
        return _validation_failure(
            language,
            "fail_closed_material_subquestion_unsupported",
            missing_requirements=missing_requirements,
        )

    action = "complete_answer_from_existing_claims" if completed is not response else "keep"
    return completed, {"action": action}


def recompose_records(
    original_records: list[dict[str, Any]], replacements: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    original_ids = [record["id"] for record in original_records]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError("Input records must contain unique IDs")
    if not set(replacements).issubset(original_ids):
        raise ValueError("Replacement IDs must be a subset of input IDs")
    return [replacements.get(record["id"], record) for record in original_records]


def run_post_retrieval_safety(
    *,
    suite_path: Path,
    combined_checkpoint_a_path: Path,
    negative_query_state_path: Path,
    relevant_query_state_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    input_hashes = {
        "answer_suite_sha256": sha256_file(suite_path),
        "combined_checkpoint_a_sha256": sha256_file(combined_checkpoint_a_path),
        "negative_query_state_sha256": sha256_file(negative_query_state_path),
        "relevant_query_state_sha256": sha256_file(relevant_query_state_path),
    }
    expected_hashes = {
        "answer_suite_sha256": ANSWER_SUITE_SHA256,
        "combined_checkpoint_a_sha256": COMBINED_CHECKPOINT_A_SHA256,
        "negative_query_state_sha256": NEGATIVE_QUERY_STATE_SHA256,
        "relevant_query_state_sha256": RELEVANT_QUERY_STATE_SHA256,
    }
    mismatches = [
        name for name, expected in expected_hashes.items() if input_hashes[name] != expected
    ]
    if mismatches:
        raise ValueError(f"Frozen post-retrieval inputs differ: {mismatches}")

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    combined = json.loads(combined_checkpoint_a_path.read_text(encoding="utf-8"))
    negative = json.loads(negative_query_state_path.read_text(encoding="utf-8"))
    relevant = json.loads(relevant_query_state_path.read_text(encoding="utf-8"))
    states = _query_states(suite, negative, relevant)
    cases = {case["id"]: case for case in suite["cases"]}
    if set(cases) != {record["id"] for record in combined["records"]}:
        raise ValueError("Suite and combined checkpoint A records do not align")

    replacements: dict[str, dict[str, Any]] = {}
    action_counts: Counter[str] = Counter()
    for record in combined["records"]:
        case = cases[record["id"]]
        guarded, safety_audit = apply_answer_safety(
            question=case["query"],
            language=case["language"],
            response=record["response"],
            evidence=record["retrieved_evidence"],
            query_state=states[record["id"]],
            generated_status_before_query_state=record.get(
                "generated_status_before_query_state"
            ),
            answer_path=record.get("answer_path"),
        )
        action = safety_audit["action"]
        action_counts[action] += 1
        if guarded is record["response"]:
            continue
        replacement = {
            **record,
            "response": guarded,
            "automatic_audit": automatic_answer_audit(case, guarded["answer"]),
            "structured_diagnostics": retrieved_structured_diagnostics(
                case, guarded, record["retrieved_evidence"]
            ),
            "post_retrieval_safety": {
                **safety_audit,
                "response_before_safety": record["response"],
            },
        }
        replacements[record["id"]] = replacement

    records = recompose_records(combined["records"], replacements)
    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "configuration": {
            "runtime_inputs": [
                "question",
                "frozen_query_state",
                "retrieved_evidence",
                "claim_links",
                "citations",
            ],
            "evaluation_gold_available_to_runtime_guard": False,
            "hosted_calls": 0,
            "untouched_records": "copied by value from frozen checkpoint A input",
        },
        "inputs": input_hashes,
        "counts": {
            "all_cases": len(records),
            "changed_cases": len(replacements),
            "untouched_cases": len(records) - len(replacements),
            "actions": dict(sorted(action_counts.items())),
        },
        "automatic_metrics": {
            "overall": _aggregate(records),
            "fr": _aggregate([record for record in records if record["language"] == "fr"]),
            "ar": _aggregate([record for record in records if record["language"] == "ar"]),
        },
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "post_retrieval_answer_safety_v1.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--combined-checkpoint-a", type=Path, required=True)
    parser.add_argument("--negative-query-state", type=Path, required=True)
    parser.add_argument("--relevant-query-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_post_retrieval_safety(
        suite_path=args.suite,
        combined_checkpoint_a_path=args.combined_checkpoint_a,
        negative_query_state_path=args.negative_query_state,
        relevant_query_state_path=args.relevant_query_state,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
