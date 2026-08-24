"""Summarize and gate the relevant-query false-preemption experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file, write_json_atomic


def evaluate_relevant_guard(result: dict[str, Any]) -> dict[str, Any]:
    records = result["records"]
    failures = [record for record in records if record["false_preempted"]]
    checks = {
        "frozen_relevant_case_count": len(records) == 32,
        "no_relevant_query_preempted": not failures,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failed_checks else "failed",
        "checks": checks,
        "failed_checks": failed_checks,
        "case_count": len(records),
        "proceed_to_retrieval_count": len(records) - len(failures),
        "false_preemption_count": len(failures),
        "false_preemptions": [
            {
                "id": record["id"],
                "decision": record["decision"],
                "query_state": record["query_state"],
            }
            for record in failures
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    receipt = {
        "experiment_id": "explicit-query-state-relevant-guard-development-v1",
        "input_sha256": sha256_file(args.result),
        **evaluate_relevant_guard(result),
        "decision": "KEEP" if not any(
            record["false_preempted"] for record in result["records"]
        ) else "REJECT",
        "conclusion": (
            "Standalone query-state preemption is safe on the frozen relevant queries."
            if not any(record["false_preempted"] for record in result["records"])
            else "Do not preempt retrieval with query state; use it only after evidence evaluation cannot answer."
        ),
    }
    write_json_atomic(args.output, receipt)
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
