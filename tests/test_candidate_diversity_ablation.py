from langchain_core.documents import Document

from experiments.candidate_diversity_ablation import (
    build_slim_result,
    diversify_ranked_pages,
    summarize_diversity,
)


def _candidate(source, page, text):
    return {"document": Document(page_content=text, metadata={"source": source, "page": page})}


def test_page_diversity_keeps_highest_scored_chunk_and_stable_order():
    ranked = [
        (_candidate("A.pdf", 1, "best a"), 0.9),
        (_candidate("B.pdf", 2, "best b"), 0.8),
        (_candidate("a.PDF", 1, "second a"), 0.7),
        (_candidate("C.pdf", 3, "best c"), 0.6),
    ]

    diverse = diversify_ranked_pages(ranked)

    assert [(item[0]["document"].page_content, item[1]) for item in diverse] == [
        ("best a", 0.9),
        ("best b", 0.8),
        ("best c", 0.6),
    ]


def test_summary_reports_monotonic_page_cutoff_repair():
    records = [
        {
            "id": "repair",
            "language": "ar",
            "relevant": True,
            "current_page": 5,
            "fusion_page": 6,
            "diverse_page": 5,
            "duplicate_page_candidate_count": 1,
        },
        {
            "id": "same",
            "language": "fr",
            "relevant": True,
            "current_page": 1,
            "fusion_page": 1,
            "diverse_page": 1,
            "duplicate_page_candidate_count": 0,
        },
    ]

    summary = summarize_diversity(records)

    assert summary["overall"]["diverse_vs_fusion_at_5"] == {
        "repairs": 1,
        "regressions": 0,
        "net": 1,
    }
    assert summary["overall"]["diverse"]["exact_page_top5"] == 1.0
    assert summary["queries_with_page_duplicates"] == 1


def test_slim_result_retains_all_ranks_and_only_changed_full_records():
    records = [
        {"id": "changed", "language": "ar", "current_page": 5, "fusion_page": 6, "diverse_page": 5},
        {"id": "same", "language": "fr", "current_page": 1, "fusion_page": 1, "diverse_page": 1},
    ]
    full = {
        "status": "complete",
        "timestamp": "now",
        "experiment_id": "id",
        "decision": "KEEP_FOR_UNSEEN_VALIDATION",
        "deployment_status": "PROHIBITED_PENDING_UNSEEN_VALIDATION",
        "hypothesis": "hypothesis",
        "predeclared_gate": "gate",
        "configuration": {},
        "inputs": {},
        "summary": {"changed_page_ranks": ["changed"]},
        "latency_seconds": {},
        "limitations": [],
        "records": records,
    }

    slim = build_slim_result(full, full_result_sha256="hash")

    assert set(slim["rank_records"]) == {"changed", "same"}
    assert slim["changed_rank_records"] == [records[0]]
    assert slim["artifact_hashes"]["full_result_sha256"] == "hash"
