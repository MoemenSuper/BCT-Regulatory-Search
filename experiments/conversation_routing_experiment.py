import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from conversation import route_message
from llm import create_llm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "experiments/stress_suites/conversation_routing_development_v1.json"
DEFAULT_OUTPUT = ROOT / "experiments/results/conversation_routing_development_v1.json"


def memory_fixture(name):
    first_turn = {
        "user_message": "What did Circular 2019-07 change?",
        "standalone_query": "changes made by Circular 2019-07",
        "answer": "It amended the exchange-office rules.",
        "sources": [{"file": "Cir_2019_07_fr.pdf", "page": 3}],
        "graph_trace": {"status": "EXPANDED"},
    }
    second_turn = {
        "user_message": "Now explain Circular 2025-17.",
        "standalone_query": "reporting duties in Circular 2025-17",
        "answer": "It contains reporting duties.",
        "sources": [{"file": "Cir_2025_17_fr.pdf", "page": 2}],
        "graph_trace": {"status": "NO_EVIDENCE"},
    }
    fixtures = {
        "empty": {
            "topics": [], "first_topic": None, "current_topic": None, "turns": []
        },
        "one": {
            "topics": ["Circular 2019-07"],
            "first_topic": "Circular 2019-07",
            "current_topic": "Circular 2019-07",
            "turns": [first_turn],
        },
        "two_current": {
            "topics": ["Circular 2019-07", "Circular 2025-17"],
            "first_topic": "Circular 2019-07",
            "current_topic": "Circular 2025-17",
            "turns": [first_turn, second_turn],
        },
        "two_no_current": {
            "topics": ["Circular 2019-07", "Circular 2025-17"],
            "first_topic": "Circular 2019-07",
            "current_topic": None,
            "turns": [first_turn, second_turn],
        },
    }
    return fixtures[name]


def score_case(case, route):
    intent_correct = route.get("intent") == case["expected_intent"]
    rewrite = (route.get("rewrite_query") or "").casefold()
    groups = case.get("required_rewrite_groups", [])
    rewrite_correct = all(
        any(option.casefold() in rewrite for option in alternatives)
        for alternatives in groups
    )
    return {
        "intent_correct": intent_correct,
        "rewrite_correct": rewrite_correct,
        "passed": intent_correct and rewrite_correct,
    }


def run(suite_path=DEFAULT_SUITE, output_path=DEFAULT_OUTPUT):
    suite_path = Path(suite_path)
    output_path = Path(output_path)
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes.decode("utf-8"))
    llm = create_llm()
    records = []

    for case in suite["cases"]:
        started = time.perf_counter()
        try:
            route = route_message(llm, case["message"], memory_fixture(case["memory"]))
            error = None
        except Exception as exc:
            route = {"intent": "ERROR", "rewrite_query": None}
            error = f"{type(exc).__name__}: {exc}"
        score = score_case(case, route)
        records.append({
            "id": case["id"],
            "message": case["message"],
            "expected_intent": case["expected_intent"],
            "route": route,
            "score": score,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "error": error,
        })

    count = len(records)
    result = {
        "experiment_id": suite["suite_id"],
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suite_sha256": hashlib.sha256(suite_bytes).hexdigest().upper(),
        "model": getattr(llm, "model_name", "openai/gpt-oss-120b"),
        "case_count": count,
        "metrics": {
            "intent_accuracy": sum(r["score"]["intent_correct"] for r in records) / count,
            "rewrite_accuracy": sum(r["score"]["rewrite_correct"] for r in records) / count,
            "full_pass_rate": sum(r["score"]["passed"] for r in records) / count,
            "provider_error_count": sum(r["error"] is not None for r in records),
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.suite, args.output)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
