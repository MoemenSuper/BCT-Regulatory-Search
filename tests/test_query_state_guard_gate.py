from experiments.query_state_guard_gate import evaluate_relevant_guard


def _result(false_ids=()):
    return {
        "records": [
            {
                "id": f"case-{index}",
                "decision": (
                    "out_of_scope"
                    if f"case-{index}" in false_ids
                    else "proceed_to_retrieval"
                ),
                "false_preempted": f"case-{index}" in false_ids,
                "query_state": {"scope": "clearly_unrelated"},
            }
            for index in range(32)
        ]
    }


def test_relevant_guard_requires_all_32_to_proceed():
    assert evaluate_relevant_guard(_result())["status"] == "passed"

    failed = evaluate_relevant_guard(_result({"case-7"}))
    assert failed["status"] == "failed"
    assert failed["false_preemption_count"] == 1
    assert failed["false_preemptions"][0]["id"] == "case-7"
