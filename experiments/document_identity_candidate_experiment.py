"""Gold-blind document-identity reranker-input experiment."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.ocr_fusion_retrieval import (
    OCR_BM25_K,
    OCR_DENSE_K,
    _deserialize_candidates,
    _load_search_representation,
    _merge_candidates,
    _ranks,
    _retrieve,
    is_arabic_query,
)
from experiments.provisional_validation_retrieval import (
    NATIVE_MANIFEST_SHA256,
    OCR_MANIFEST_SHA256,
    diversify_ranked_pages,
)
from reranker import create_reranker, score_documents


EVALUATION_SHA256 = "00964BA335B759D01BA42CED75FC6AE10F082AE4D45499426CEA69F3F1DF3CA1"
NATIVE_CANDIDATES_SHA256 = (
    "A061D88BEE3F4C221EFC8612AC13272C83AA1C8A2EEA475409C2DC6E0CF65B33"
)
DIVERSITY_RESULT_SHA256 = (
    "1A0ABFD6D97CDAC7AC102987B1E274DD7958295A9E18D2577B694C9F566CDE40"
)
TARGETED_CATALOG_SHA256 = (
    "296C670D8CCBB5B46D3830D19AA98337468B0F11931CA9D5C86797A48931B99C"
)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EXPERIMENT_ID = "document-identity-reranker-prefix-development-v1"

_SOURCE_ID = re.compile(
    r"(?i)^(?P<kind>cir|note|cb|nb|ci)[_ -]?(?P<year>\d{4})"
    r"[_ -]?(?P<number>\d+).*?[_ -]?(?P<language>fr|ar)\.pdf$"
)
_FILENAME_QUERY_ID = re.compile(
    r"(?i)\b(?P<kind>cir|note|cb|nb|ci)[_ -](?P<year>\d{4})"
    r"[_ -](?P<number>\d+)(?:[_ -](?:fr|ar))?(?:\.pdf)?\b"
)
_FRENCH_QUERY_ID = re.compile(
    r"(?i)\b(?P<kind>circulaire|note)\s+"
    r"(?:n(?:[°ºo])?|numero)\s*"
    r"(?:(?P<year>\d{4})\s*[-/]\s*(?P<number>\d+)"
    r"|(?P<number_first>\d+)\s*(?:de|/)\s*(?P<year_last>\d{4}))"
)
_ARABIC_QUERY_ID = re.compile(
    r"(?P<kind>المنشور|منشور|المذكرة|مذكرة)\s*"
    r"(?:عدد|رقم)\s*(?P<number>\d+)\s*"
    r"(?:لسنة|سنة|لعام|عام)\s*(?P<year>\d{4})"
)
_FRENCH_YEAR = re.compile(
    r"(?i)\b(?:en|type|publiee?\s+en|emise?\s+en|adoptee?\s+en)"
    r"\s+(?P<year>(?:19|20)\d{2})\b"
)
_ARABIC_YEAR = re.compile(
    r"(?:لسنة|سنة|لعام|عام)\s*(?P<year>(?:19|20)\d{2})(?!\d)"
)
_FUTURE = re.compile(
    r"(?i)\b(?:futur|future|prochain|prochaine|sera\s+publie)\b|"
    r"سيصدر|ستصدر|القادم|القادمة"
)


def _ascii_fold(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )


def parse_source_identity(source: str) -> dict[str, Any] | None:
    filename = str(source).replace("\\", "/").rsplit("/", 1)[-1]
    match = _SOURCE_ID.match(filename)
    if not match:
        return None
    values = match.groupdict()
    return {
        "kind": values["kind"].casefold(),
        "year": int(values["year"]),
        "number": int(values["number"]),
        "language": values["language"].casefold(),
    }


def _query_identity(
    *, kind: str | None, year: str | int, number: str | int | None, reason: str
) -> dict[str, Any]:
    normalized_kind = kind.casefold() if kind is not None else None
    normalized_kind = {
        "circulaire": "cir",
        "المنشور": "cir",
        "منشور": "cir",
        "المذكرة": "note",
        "مذكرة": "note",
    }.get(normalized_kind, normalized_kind)
    return {
        "kind": normalized_kind,
        "year": int(year),
        "number": int(number) if number is not None else None,
        "route_reason": reason,
    }


def parse_query_identity(query: str) -> dict[str, Any] | None:
    """Return only identity signals observable in the runtime query text."""
    filename_match = _FILENAME_QUERY_ID.search(query)
    if filename_match:
        values = filename_match.groupdict()
        return _query_identity(
            kind=values["kind"],
            year=values["year"],
            number=values["number"],
            reason="explicit_instrument_identity",
        )

    folded = _ascii_fold(query)
    french_match = _FRENCH_QUERY_ID.search(folded)
    if french_match:
        values = french_match.groupdict()
        return _query_identity(
            kind=values["kind"],
            year=values["year"] or values["year_last"],
            number=values["number"] or values["number_first"],
            reason="explicit_instrument_identity",
        )

    arabic_match = _ARABIC_QUERY_ID.search(query)
    if arabic_match:
        values = arabic_match.groupdict()
        return _query_identity(
            kind=values["kind"],
            year=values["year"],
            number=values["number"],
            reason="explicit_instrument_identity",
        )

    if _FUTURE.search(folded) or _FUTURE.search(query):
        return None
    years = {
        int(match.group("year"))
        for pattern, text in ((_FRENCH_YEAR, folded), (_ARABIC_YEAR, query))
        for match in pattern.finditer(text)
    }
    if len(years) != 1:
        return None
    return _query_identity(
        kind=None,
        year=years.pop(),
        number=None,
        reason="explicit_source_year",
    )


def _identity_prefix(identity: dict[str, Any]) -> str:
    return (
        f"[document kind={identity['kind']}; year={identity['year']}; "
        f"number={identity['number']}; language={identity['language']}]"
    )


def build_identity_reranker_documents(
    candidates: list[dict[str, Any]], query_identity: dict[str, Any] | None
) -> list[Document]:
    if query_identity is None:
        return [candidate["document"] for candidate in candidates]
    documents = []
    for candidate in candidates:
        original = candidate["document"]
        identity = parse_source_identity(str(original.metadata.get("source", "")))
        if identity is None:
            documents.append(original)
            continue
        documents.append(
            Document(
                page_content=f"{_identity_prefix(identity)}\n{original.page_content}",
                metadata=dict(original.metadata),
            )
        )
    return documents


def ranked_signature(
    ranked: list[tuple[dict[str, Any], float]], *, limit: int | None = None
) -> list[dict[str, Any]]:
    selected = ranked if limit is None else ranked[:limit]
    return [
        {
            "source": Path(str(candidate["document"].metadata.get("source", ""))).name,
            "page": int(candidate["document"].metadata.get("page", -1)),
            "score": float(score),
        }
        for candidate, score in selected
    ]


def _rank(
    reranker: Any,
    query: str,
    candidates: list[dict[str, Any]],
    documents: list[Document],
) -> list[tuple[dict[str, Any], float]]:
    scored = score_documents(reranker, query, documents)
    ranked = sorted(
        zip(candidates, (float(score) for _document, score in scored)),
        key=lambda item: item[1],
        reverse=True,
    )
    return diversify_ranked_pages(ranked)


def _signature_ranks(
    signature: list[dict[str, Any]], expected_source: str, expected_page: int
) -> tuple[int | None, int | None]:
    source_rank = exact_page_rank = None
    expected_name = Path(expected_source).name.casefold()
    for index, item in enumerate(signature, start=1):
        if Path(item["source"]).name.casefold() != expected_name:
            continue
        if source_rank is None:
            source_rank = index
        if int(item["page"]) == int(expected_page) and exact_page_rank is None:
            exact_page_rank = index
    return source_rank, exact_page_rank


def _hit(rank: int | None, cutoff: int) -> bool:
    return rank is not None and rank <= cutoff


def _arm_metrics(records: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    relevant = [record for record in records if record["relevant"]]
    page_ranks = [record[f"{arm}_page_rank"] for record in relevant]
    source_ranks = [record[f"{arm}_source_rank"] for record in relevant]
    if not relevant:
        return {"n": 0}
    return {
        "n": len(relevant),
        "source_top1": sum(_hit(rank, 1) for rank in source_ranks) / len(relevant),
        "source_top5": sum(_hit(rank, 5) for rank in source_ranks) / len(relevant),
        "exact_page_top1": sum(_hit(rank, 1) for rank in page_ranks) / len(relevant),
        "exact_page_top5": sum(_hit(rank, 5) for rank in page_ranks) / len(relevant),
        "exact_page_top20": sum(_hit(rank, 20) for rank in page_ranks) / len(relevant),
        "mrr_page": statistics.mean(
            1.0 / rank if rank is not None else 0.0 for rank in page_ranks
        ),
    }


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [record for record in records if record["relevant"]]
    repairs = [
        record["id"]
        for record in relevant
        if not _hit(record["control_page_rank"], 5)
        and _hit(record["candidate_page_rank"], 5)
    ]
    regressions = [
        record["id"]
        for record in relevant
        if _hit(record["control_page_rank"], 5)
        and not _hit(record["candidate_page_rank"], 5)
    ]
    return {
        "control": _arm_metrics(records, "control"),
        "candidate": _arm_metrics(records, "candidate"),
        "repairs_at_5": repairs,
        "regressions_at_5": regressions,
        "net_at_5": len(repairs) - len(regressions),
    }


def _score_records(
    runtime_records: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    temporal_roles: dict[str, str],
    frozen_diversity: dict[str, Any],
) -> list[dict[str, Any]]:
    cases = {case["id"]: case for case in evaluation}
    scored = []
    for runtime in runtime_records:
        case = cases[runtime["id"]]
        relevant = bool(case["relevant"])
        control_source = control_page = candidate_source = candidate_page = None
        if relevant:
            control_source, control_page = _signature_ranks(
                runtime["control_ranked"],
                str(case["expected_source"]),
                int(case["expected_page"]),
            )
            candidate_source, candidate_page = _signature_ranks(
                runtime["candidate_ranked"],
                str(case["expected_source"]),
                int(case["expected_page"]),
            )
            frozen_page = frozen_diversity["rank_records"][case["id"]]["diverse_page"]
            if control_page != frozen_page:
                raise ValueError(
                    f"Control diverse rank drift for {case['id']}: "
                    f"expected {frozen_page}, reproduced {control_page}"
                )
        scored.append(
            {
                "id": runtime["id"],
                "language": case["language"],
                "relevant": relevant,
                "routed": runtime["query_identity"] is not None,
                "route_reason": (
                    runtime["query_identity"]["route_reason"]
                    if runtime["query_identity"] is not None
                    else None
                ),
                "temporal_role": temporal_roles.get(runtime["id"]),
                "control_source_rank": control_source,
                "control_page_rank": control_page,
                "candidate_source_rank": candidate_source,
                "candidate_page_rank": candidate_page,
                "top5_changed": runtime["control_ranked"][:5]
                != runtime["candidate_ranked"][:5],
                "control_top5": runtime["control_ranked"][:5],
                "candidate_top5": runtime["candidate_ranked"][:5],
            }
        )
    return scored


def run_experiment(
    *,
    evaluation_path: Path,
    native_candidates_path: Path,
    native_manifest_path: Path,
    ocr_manifest_path: Path,
    diversity_result_path: Path,
    targeted_catalog_path: Path,
    routing_output_path: Path,
    result_output_path: Path,
) -> dict[str, Any]:
    frozen = (
        (evaluation_path, EVALUATION_SHA256, "evaluation"),
        (native_candidates_path, NATIVE_CANDIDATES_SHA256, "native candidates"),
        (native_manifest_path, NATIVE_MANIFEST_SHA256, "native manifest"),
        (ocr_manifest_path, OCR_MANIFEST_SHA256, "OCR manifest"),
        (diversity_result_path, DIVERSITY_RESULT_SHA256, "diversity result"),
        (targeted_catalog_path, TARGETED_CATALOG_SHA256, "targeted catalog"),
    )
    for path, expected_hash, label in frozen:
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen {label} hash differs")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    native_cache = json.loads(native_candidates_path.read_text(encoding="utf-8"))
    if set(native_cache) != {case["id"] for case in evaluation}:
        raise ValueError("Native candidate cache must exactly cover development IDs")
    ocr = _load_search_representation(
        json.loads(ocr_manifest_path.read_text(encoding="utf-8"))
    )
    reranker = create_reranker()
    runtime_records = []
    control_seconds = identity_seconds = 0.0

    for index, case in enumerate(evaluation, start=1):
        base = _deserialize_candidates(native_cache[case["id"]])
        ocr_candidates = (
            _retrieve(ocr, case["query"], OCR_DENSE_K, OCR_BM25_K)
            if is_arabic_query(case["query"])
            else []
        )
        candidates = _merge_candidates((base, ocr_candidates))
        control_started = time.perf_counter()
        control = _rank(
            reranker,
            case["query"],
            candidates,
            [candidate["document"] for candidate in candidates],
        )
        control_seconds += time.perf_counter() - control_started

        query_identity = parse_query_identity(case["query"])
        if query_identity is None:
            candidate_ranked = control
        else:
            candidate_started = time.perf_counter()
            candidate_ranked = _rank(
                reranker,
                case["query"],
                candidates,
                build_identity_reranker_documents(candidates, query_identity),
            )
            identity_seconds += time.perf_counter() - candidate_started
        control_signature = ranked_signature(control)
        candidate_signature = ranked_signature(candidate_ranked)
        if query_identity is None and control_signature != candidate_signature:
            raise ValueError(f"Un-routed rank drift for {case['id']}")
        runtime_records.append(
            {
                "id": case["id"],
                "query": case["query"],
                "query_identity": query_identity,
                "candidate_count": len(candidates),
                "base_candidate_count": len(base),
                "ocr_candidate_count": len(ocr_candidates),
                "control_ranked": control_signature,
                "candidate_ranked": candidate_signature,
            }
        )
        print(
            f"[identity-rerank {index}/{len(evaluation)}] {case['id']} "
            f"routed={query_identity is not None}",
            flush=True,
        )

    routing_artifact = {
        "status": "frozen_before_gold_scoring",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "policy": {
            "routed_signals": ["explicit_instrument_identity", "explicit_source_year"],
            "reranker_prefix": (
                "candidate filename kind, year, number, and language; original text unchanged"
            ),
            "unrouted": "exact control reranker documents and ranks",
            "gold_fields_used": [],
        },
        "inputs": {label.replace(" ", "_"): sha256_file(path) for path, _hash, label in frozen},
        "counts": {
            "queries": len(runtime_records),
            "routed": sum(record["query_identity"] is not None for record in runtime_records),
            "route_reasons": dict(
                Counter(
                    record["query_identity"]["route_reason"]
                    for record in runtime_records
                    if record["query_identity"] is not None
                )
            ),
        },
        "records": runtime_records,
    }
    write_json_atomic(routing_output_path, routing_artifact)

    frozen_diversity = json.loads(diversity_result_path.read_text(encoding="utf-8"))
    catalog = json.loads(targeted_catalog_path.read_text(encoding="utf-8"))
    temporal_roles = {
        case["id"]: case["role"]
        for case in catalog["suites"]["temporal_near_duplicate"]["cases"]
    }
    scored = _score_records(
        runtime_records, evaluation, temporal_roles, frozen_diversity
    )
    groups = {
        "overall": _group_metrics(scored),
        "fr": _group_metrics([record for record in scored if record["language"] == "fr"]),
        "ar": _group_metrics([record for record in scored if record["language"] == "ar"]),
        "routed": _group_metrics([record for record in scored if record["routed"]]),
        "temporal_failures": _group_metrics(
            [record for record in scored if record["temporal_role"] == "failure"]
        ),
        "temporal_controls": _group_metrics(
            [record for record in scored if record["temporal_role"] == "control"]
        ),
    }
    no_decrease = {
        language: groups[language]["candidate"]["exact_page_top5"]
        >= groups[language]["control"]["exact_page_top5"]
        for language in ("overall", "fr", "ar")
    }
    gate = {
        "at_least_two_temporal_failure_repairs": (
            len(groups["temporal_failures"]["repairs_at_5"]) >= 2
        ),
        "zero_temporal_control_regressions": (
            not groups["temporal_controls"]["regressions_at_5"]
        ),
        "overall_french_arabic_page5_non_decrease": all(no_decrease.values()),
        "unrouted_rankings_identical": all(
            record["control_ranked"] == record["candidate_ranked"]
            for record in runtime_records
            if record["query_identity"] is None
        ),
        "fixed_candidate_budgets_and_pipeline": True,
        "gold_blind_routing_and_ranking": True,
    }
    decision = "KEEP" if all(gate.values()) else "REJECT"
    result = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "Compact filename-derived identity prefixes improve explicit source/version "
            "selection without regressing routed controls."
        ),
        "changed_variable": "reranker_input_identity_prefix_for_runtime_routed_queries",
        "configuration": {
            "dense_top_k": 20,
            "bm25_top_k": 15,
            "arabic_ocr_dense_top_k": OCR_DENSE_K,
            "arabic_ocr_bm25_top_k": OCR_BM25_K,
            "reranker": RERANKER_MODEL,
            "final_selection": "stable source/page diverse top 5",
            "hosted_calls": 0,
        },
        "inputs": {
            "routing_receipt_sha256": sha256_file(routing_output_path),
            **{label.replace(" ", "_"): sha256_file(path) for path, _hash, label in frozen},
        },
        "routing": routing_artifact["counts"],
        "metrics": groups,
        "changed_top5_ids": [record["id"] for record in scored if record["top5_changed"]],
        "repairs": groups["overall"]["repairs_at_5"],
        "regressions": groups["overall"]["regressions_at_5"],
        "latency_seconds": {
            "control_reranking_total": control_seconds,
            "identity_reranking_total": identity_seconds,
            "identity_reranking_mean_per_routed_query": (
                identity_seconds / routing_artifact["counts"]["routed"]
                if routing_artifact["counts"]["routed"]
                else 0.0
            ),
        },
        "gate": gate,
        "decision": decision,
        "validation_metrics": {
            "status": "not_run",
            "reason": "Provisional answer validation remains closed.",
        },
        "answer_calls": {
            "status": "eligible_only_on_keep" if decision == "KEEP" else "not_run",
            "count": 0,
        },
        "records": scored,
    }
    write_json_atomic(result_output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--native-candidates", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--ocr-manifest", type=Path, required=True)
    parser.add_argument("--diversity-result", type=Path, required=True)
    parser.add_argument("--targeted-catalog", type=Path, required=True)
    parser.add_argument("--routing-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(
        evaluation_path=args.evaluation,
        native_candidates_path=args.native_candidates,
        native_manifest_path=args.native_manifest,
        ocr_manifest_path=args.ocr_manifest,
        diversity_result_path=args.diversity_result,
        targeted_catalog_path=args.targeted_catalog,
        routing_output_path=args.routing_output,
        result_output_path=args.result_output,
    )


if __name__ == "__main__":
    main()
