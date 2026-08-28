"""Evaluate the enriched graph-temporal slice without pretending GraphRAG exists."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.artifacts import sha256_file, write_json_atomic
from experiments.document_identity_candidate_experiment import (
    build_identity_reranker_documents,
    parse_query_identity,
    ranked_signature,
)
from experiments.ocr_fusion_retrieval import (
    OCR_BM25_K,
    OCR_DENSE_K,
    _load_search_representation,
    _merge_candidates,
    _retrieve,
    is_arabic_query,
)
from experiments.provisional_validation_retrieval import diversify_ranked_pages
from experiments.retrieval_ablations import BM25_K, DENSE_K
from regulatory_graph.fixtures import circular_2016_03_fr_bundle
from regulatory_graph.neo4j_store import Neo4jRegulatoryGraph
from reranker import create_reranker, score_documents


EXPERIMENT_ID = "graph-temporal-enriched-benchmark-v1"
ENRICHED_EVALUATION_SHA256 = (
    "6DACE8538C9D5F47EFCA5E038DC4853B66A283803C1B0CF06A8B11DBD8DC99EB"
)
NATIVE_MANIFEST_SHA256 = (
    "920159F84E87353256899370A2E8B7854BAB87FBBBDC05184FED7C3657B439C1"
)
OCR_MANIFEST_SHA256 = (
    "8CBD9A468B2D017DB46218252B555231460A64381245786CCA020C84F3CD05AB"
)
EXPECTED_CASE_COUNT = 50
CUTOFFS = (1, 5, 10, 20)

_OFFICE_2018_QUOTE = (
    "Article 3 : L’autorisation d’exercice de l’activité de change manuel par "
    "l’ouverture d’un bureau de change est personnelle et incessible. Une même "
    "personne physique ne peut bénéficier de plus d’une autorisation. Celle-ci "
    "habilite son titulaire à exercer l’activité de change manuel exclusivement "
    "dans le bureau de change qui y est indiqué. Elle ne permet, en aucun cas, "
    "l’exercice de ladite activité par plus d’un bureau de change."
)
_START_2018_QUOTE = (
    "La personne physique ayant obtenu l’autorisation doit, dans un délai ne "
    "dépassant pas trois mois à partir de la date de l’autorisation, procéder à "
    "l’exercice effectif de son activité et transmettre à la Banque Centrale de "
    "Tunisie, par tout moyen laissant trace écrite, dans un délai maximum de 3 "
    "jours ouvrables à compter de la date d’entrée en activité, une déclaration, "
    "établie selon le modèle objet de l’annexe n°3 à la présente circulaire."
)
_AUDITED_CORRECTIONS = {
    "graph_fr_2019_07_multi_office_02": _OFFICE_2018_QUOTE,
    "graph_fr_2019_07_start_deadline_03": _START_2018_QUOTE,
}


def select_graph_temporal_cases(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        copy.deepcopy(row)
        for row in rows
        if row.get("requires_graph") is True
        and row.get("evaluation_slice") == "graph_temporal"
    ]
    ids = [str(case.get("id", "")) for case in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate graph-temporal evaluation IDs")
    if any(not case_id for case_id in ids):
        raise ValueError("graph-temporal evaluation case requires an ID")
    return selected


def apply_audited_corrections(
    cases: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corrected = copy.deepcopy(list(cases))
    receipts = []
    for case in corrected:
        case_id = case["id"]
        replacement_quote = _AUDITED_CORRECTIONS.get(case_id)
        if replacement_quote is None:
            continue
        source_entry = next(
            (
                item
                for item in case.get("expected_sources", [])
                if item.get("source") == "Cir_2018_07_fr.pdf"
            ),
            None,
        )
        if source_entry is None or source_entry.get("pages") != [2]:
            raise ValueError(
                f"{case_id} no longer matches audited page 2 source expectation"
            )
        evidence_entry = next(
            (
                item
                for item in case.get("evidence_quotes", [])
                if item.get("source") == "Cir_2018_07_fr.pdf"
            ),
            None,
        )
        if evidence_entry is None or evidence_entry.get("page") != 2:
            raise ValueError(
                f"{case_id} no longer matches audited page 2 evidence expectation"
            )
        before_quote = evidence_entry.get("quote")
        source_entry["pages"] = [3]
        evidence_entry["page"] = 3
        evidence_entry["quote"] = replacement_quote
        if case.get("evidence_quote") == before_quote:
            case["evidence_quote"] = replacement_quote
        receipts.append(
            {
                "id": case_id,
                "source": "Cir_2018_07_fr.pdf",
                "before_pages": [2],
                "after_pages": [3],
                "reason": "rule is visibly present on PDF page 3, not page 2",
                "replacement_quote": replacement_quote,
            }
        )
    return corrected, receipts


def _source_key(value: Any) -> str:
    return Path(str(value)).name.casefold()


def _required_sources(case: dict[str, Any]) -> set[str]:
    return {
        _source_key(item["source"])
        for item in case.get("expected_sources", [])
    }


def _required_page_pairs(case: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (_source_key(item["source"]), int(page))
        for item in case.get("expected_sources", [])
        for page in item.get("pages", [])
    }


def _page_pairs(items: Iterable[dict[str, Any]]) -> list[tuple[str, int]]:
    return [(_source_key(item["source"]), int(item["page"])) for item in items]


def _first_rank(values: list[Any], target: Any) -> int | None:
    try:
        return values.index(target) + 1
    except ValueError:
        return None


def score_case(
    case: dict[str, Any],
    candidate_pages: list[dict[str, Any]],
    ranked_pages: list[dict[str, Any]],
) -> dict[str, Any]:
    required_sources = _required_sources(case)
    required_pairs = _required_page_pairs(case)
    ranked_pairs = _page_pairs(ranked_pages)
    ranked_sources = [source for source, _page in ranked_pairs]
    candidate_pair_set = set(_page_pairs(candidate_pages))
    primary = (_source_key(case["expected_source"]), int(case["expected_page"]))
    result: dict[str, Any] = {
        "primary_page_rank": _first_rank(ranked_pairs, primary),
        "required_source_count": len(required_sources),
        "required_page_pair_count": len(required_pairs),
        "missing_required_page_pairs_from_candidates": [
            {"source": source, "page": page}
            for source, page in sorted(required_pairs - candidate_pair_set)
        ],
    }
    for cutoff in CUTOFFS:
        found_sources = set(ranked_sources[:cutoff]) & required_sources
        found_pairs = set(ranked_pairs[:cutoff]) & required_pairs
        result[f"required_source_recall_at_{cutoff}"] = (
            len(found_sources) / len(required_sources) if required_sources else 1.0
        )
        result[f"complete_required_sources_at_{cutoff}"] = (
            found_sources == required_sources
        )
        result[f"required_page_pair_recall_at_{cutoff}"] = (
            len(found_pairs) / len(required_pairs) if required_pairs else 1.0
        )
        result[f"complete_required_page_pairs_at_{cutoff}"] = (
            found_pairs == required_pairs
        )
    return result


def graph_readiness(
    cases: Iterable[dict[str, Any]],
    *,
    local_graph_sources: set[str],
    runtime_retriever_exists: bool,
    answer_context_assembler_exists: bool,
) -> dict[str, Any]:
    normalized_graph_sources = {_source_key(source) for source in local_graph_sources}
    case_rows = []
    for case in cases:
        required = _required_sources(case)
        present = required & normalized_graph_sources
        case_rows.append(
            {
                "id": case["id"],
                "required_sources": sorted(required),
                "present_sources": sorted(present),
                "missing_sources": sorted(required - present),
                "any_required_source": bool(present),
                "all_required_sources": present == required,
            }
        )
    all_count = sum(row["all_required_sources"] for row in case_rows)
    gate = {
        "all_cases_have_all_required_sources": all_count == len(case_rows),
        "runtime_graph_retriever_exists": runtime_retriever_exists,
        "answer_context_assembler_exists": answer_context_assembler_exists,
    }
    return {
        "local_graph_sources": sorted(normalized_graph_sources),
        "case_count": len(case_rows),
        "cases_with_any_required_source": sum(
            row["any_required_source"] for row in case_rows
        ),
        "cases_with_all_required_sources": all_count,
        "runtime_retriever_exists": runtime_retriever_exists,
        "answer_context_assembler_exists": answer_context_assembler_exists,
        "gate": gate,
        "decision": "GRAPH_ARM_READY" if all(gate.values()) else "GRAPH_ARM_NOT_READY",
        "records": case_rows,
    }


def _candidate_pages(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: set[tuple[str, int]] = set()
    for candidate in candidates:
        metadata = candidate["document"].metadata
        source = Path(str(metadata.get("source", ""))).name
        raw_pages = metadata.get("pages")
        if isinstance(raw_pages, str):
            raw_pages = [part for part in raw_pages.split(",") if part]
        if not raw_pages:
            raw_pages = [metadata.get("page", -1)]
        pages.update((source, int(page)) for page in raw_pages)
    return [
        {"source": source, "page": page}
        for source, page in sorted(pages, key=lambda item: (item[0].casefold(), item[1]))
    ]


def _rank(
    reranker: Any,
    query: str,
    candidates: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], float]]:
    query_identity = parse_query_identity(query)
    documents = build_identity_reranker_documents(candidates, query_identity)
    scored = score_documents(reranker, query, documents)
    ranked = sorted(
        zip(candidates, (float(score) for _document, score in scored)),
        key=lambda item: item[1],
        reverse=True,
    )
    return diversify_ranked_pages(ranked)


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    metrics: dict[str, Any] = {"n": len(records)}
    primary_ranks = [record["primary_page_rank"] for record in records]
    metrics["primary_page_at_1"] = sum(rank == 1 for rank in primary_ranks) / len(records)
    metrics["primary_page_at_5"] = sum(
        rank is not None and rank <= 5 for rank in primary_ranks
    ) / len(records)
    for cutoff in (5, 10, 20):
        metrics[f"complete_required_sources_at_{cutoff}"] = sum(
            record[f"complete_required_sources_at_{cutoff}"] for record in records
        ) / len(records)
        metrics[f"complete_required_page_pairs_at_{cutoff}"] = sum(
            record[f"complete_required_page_pairs_at_{cutoff}"] for record in records
        ) / len(records)
        metrics[f"mean_required_page_pair_recall_at_{cutoff}"] = statistics.mean(
            record[f"required_page_pair_recall_at_{cutoff}"] for record in records
        )
    return metrics


def _graph_capabilities() -> tuple[bool, bool]:
    runtime_names = ("retrieve_for_query", "search_for_query", "expand_query")
    context_names = ("assemble_answer_context", "build_answer_context")
    return (
        any(callable(getattr(Neo4jRegulatoryGraph, name, None)) for name in runtime_names),
        any(callable(getattr(Neo4jRegulatoryGraph, name, None)) for name in context_names),
    )


def run_experiment(
    *,
    evaluation_path: Path,
    native_manifest_path: Path,
    ocr_manifest_path: Path,
    slice_output_path: Path,
    result_output_path: Path,
) -> dict[str, Any]:
    frozen = (
        (evaluation_path, ENRICHED_EVALUATION_SHA256, "enriched evaluation"),
        (native_manifest_path, NATIVE_MANIFEST_SHA256, "native manifest"),
        (ocr_manifest_path, OCR_MANIFEST_SHA256, "OCR manifest"),
    )
    for path, expected_hash, label in frozen:
        actual = sha256_file(path)
        if actual != expected_hash:
            raise ValueError(f"Frozen {label} hash differs: {actual}")

    rows = json.loads(evaluation_path.read_text(encoding="utf-8"))
    selected = select_graph_temporal_cases(rows)
    if len(selected) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CASE_COUNT} graph-temporal cases, found {len(selected)}"
        )
    corrected, corrections = apply_audited_corrections(selected)
    if {item["id"] for item in corrections} != set(_AUDITED_CORRECTIONS):
        raise ValueError("The two frozen benchmark corrections were not both applied")

    slice_artifact = {
        "status": "frozen_before_scoring",
        "experiment_id": EXPERIMENT_ID,
        "source_evaluation_sha256": sha256_file(evaluation_path),
        "selection": {
            "requires_graph": True,
            "evaluation_slice": "graph_temporal",
            "runtime_fields_used": [],
        },
        "corrections": corrections,
        "case_count": len(corrected),
        "cases": corrected,
    }
    write_json_atomic(slice_output_path, slice_artifact)

    native = _load_search_representation(
        json.loads(native_manifest_path.read_text(encoding="utf-8"))
    )
    ocr = _load_search_representation(
        json.loads(ocr_manifest_path.read_text(encoding="utf-8"))
    )
    reranker = create_reranker()
    records = []
    for index, case in enumerate(corrected, start=1):
        started = time.perf_counter()
        native_candidates = _retrieve(native, case["query"], DENSE_K, BM25_K)
        ocr_candidates = (
            _retrieve(ocr, case["query"], OCR_DENSE_K, OCR_BM25_K)
            if is_arabic_query(case["query"])
            else []
        )
        candidates = _merge_candidates((native_candidates, ocr_candidates))
        diverse = _rank(reranker, case["query"], candidates)
        signature = ranked_signature(diverse)
        scored = score_case(case, _candidate_pages(candidates), signature)
        missing_candidates = scored["missing_required_page_pairs_from_candidates"]
        missing_below_20 = [
            pair
            for pair in (
                {"source": source, "page": page}
                for source, page in sorted(_required_page_pairs(case))
            )
            if pair not in missing_candidates
            and (_source_key(pair["source"]), int(pair["page"]))
            not in set(_page_pairs(signature[:20]))
        ]
        records.append(
            {
                "id": case["id"],
                "language": case["language"],
                "category": case["category"],
                "query_identity": parse_query_identity(case["query"]),
                **scored,
                "required_page_pairs_reranked_below_20": missing_below_20,
                "candidate_count": len(candidates),
                "native_candidate_count": len(native_candidates),
                "ocr_candidate_count": len(ocr_candidates),
                "latency_seconds": time.perf_counter() - started,
                "top20": signature[:20],
            }
        )
        print(
            f"[graph-temporal-control {index}/{len(corrected)}] "
            f"{case['id']} primary={scored['primary_page_rank']}",
            flush=True,
        )

    bundle = circular_2016_03_fr_bundle()
    local_sources = {
        edition.filename
        for edition in bundle.source_editions
        if any(
            instrument.uid == edition.instrument_uid and instrument.corpus_present
            for instrument in bundle.instruments
        )
    }
    runtime_retriever, context_assembler = _graph_capabilities()
    graph = graph_readiness(
        corrected,
        local_graph_sources=local_sources,
        runtime_retriever_exists=runtime_retriever,
        answer_context_assembler_exists=context_assembler,
    )
    latency = [record["latency_seconds"] for record in records]
    result = {
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "decision": graph["decision"],
        "inputs": {
            "enriched_evaluation_sha256": sha256_file(evaluation_path),
            "derived_slice_sha256": sha256_file(slice_output_path),
            "native_manifest_sha256": sha256_file(native_manifest_path),
            "ocr_manifest_sha256": sha256_file(ocr_manifest_path),
        },
        "selection": {
            "case_count": len(corrected),
            "languages": dict(Counter(case["language"] for case in corrected)),
            "categories": dict(Counter(case["category"] for case in corrected)),
            "audited_correction_count": len(corrections),
        },
        "configuration": {
            "native_dense_k": DENSE_K,
            "native_bm25_k": BM25_K,
            "arabic_ocr_dense_k": OCR_DENSE_K,
            "arabic_ocr_bm25_k": OCR_BM25_K,
            "reranker": "BAAI/bge-reranker-v2-m3",
            "document_identity_prefix": "retained runtime-observable policy",
            "source_page_diversity": True,
            "hosted_calls": 0,
            "answer_calls": 0,
        },
        "control_retrieval": {
            "overall": _group_metrics(records),
            "fr": _group_metrics([row for row in records if row["language"] == "fr"]),
            "ar": _group_metrics([row for row in records if row["language"] == "ar"]),
            "candidate_missing_page_pair_count": sum(
                len(row["missing_required_page_pairs_from_candidates"])
                for row in records
            ),
            "reranked_below_20_page_pair_count": sum(
                len(row["required_page_pairs_reranked_below_20"])
                for row in records
            ),
            "latency_seconds": {
                "total": sum(latency),
                "mean": statistics.mean(latency),
                "median": statistics.median(latency),
            },
        },
        "current_graph_readiness": graph,
        "limitations": [
            "The graph fixture covers one real circular and is not a corpus graph.",
            "No runtime query-to-graph retriever or graph answer-context assembler exists.",
            "The control arm measures retrieval only, not answer correctness.",
            "The supplied gold received an automated audit plus targeted visual review, not independent human adjudication.",
            "Provisional validation and final holdout remained unopened.",
        ],
        "records": records,
    }
    write_json_atomic(result_output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--ocr-manifest", type=Path, required=True)
    parser.add_argument("--slice-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    result = run_experiment(
        evaluation_path=args.evaluation,
        native_manifest_path=args.native_manifest,
        ocr_manifest_path=args.ocr_manifest,
        slice_output_path=args.slice_output,
        result_output_path=args.result_output,
    )
    print(json.dumps({"decision": result["decision"], "metrics": result["control_retrieval"]}, indent=2))


if __name__ == "__main__":
    main()

