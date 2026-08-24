from copy import deepcopy

from experiments.selective_answer_gate import evaluate_selective_answer_gate


def _record(
    case_id: str,
    *,
    relevant: bool,
    clarification: bool = False,
    retried: bool = False,
):
    status = "clarification_needed" if clarification else (
        "answered" if relevant else "insufficient_evidence"
    )
    return {
        "id": case_id,
        "language": "fr",
        "relevant": relevant,
        "answer_path": (
            "single_full_page_context_retry"
            if retried
            else "verified_excerpt_claim_linked_answer"
        ),
        "response": {
            "status": status,
            "answer": "10 ABC1" if relevant else "Safe response",
            "claims": [{}] if relevant else [],
            "citations": [{}] if relevant else [],
        },
        "structured_diagnostics": {
            "exact_structured_citation": True,
            "claim_evidence_links_valid": True,
        },
        "automatic_audit": {
            "expected_numbers": ["10"],
            "matched_expected_numbers": ["10"],
            "expected_identifiers": ["ABC1"],
            "matched_expected_identifiers": ["ABC1"],
        },
        "answer_evaluation": {
            "answer_correct": True,
            "citation_correct": True,
            "grounded": True,
            "safe_response": True,
            "expected_behavior_met": True,
            "clarification_requested": clarification,
        },
    }


def _passing_result():
    relevant = [_record(f"relevant-{index}", relevant=True) for index in range(30)]
    relevant.extend(
        [
            _record(
                "cir_2019_02_fr_amount_or_rate_02", relevant=True, retried=True
            ),
            _record(
                "note_2022_16_ar_ceramic_importer_02", relevant=True, retried=True
            ),
        ]
    )
    negative = [
        _record(f"negative-{index}", relevant=False, clarification=index < 2)
        for index in range(8)
    ]
    return {"records": relevant + negative}


def test_frozen_selective_answer_gate_passes_complete_candidate():
    receipt = evaluate_selective_answer_gate(_passing_result())

    assert receipt["status"] == "passed"
    assert receipt["failed_checks"] == []


def test_frozen_selective_answer_gate_rejects_literal_omission():
    result = deepcopy(_passing_result())
    result["records"][0]["response"]["answer"] = "No literal is present."

    receipt = evaluate_selective_answer_gate(result)

    assert receipt["status"] == "failed"
    assert receipt["failed_checks"] == ["all_expected_literals_preserved"]


def test_literal_gate_accepts_equivalent_grouping_and_time_spelling():
    result = _passing_result()
    record = result["records"][0]
    record["response"]["answer"] = "Le montant est 100.000 DT, de 8h00 à 17h00."
    record["automatic_audit"].update(
        expected_numbers=["100000", "8", "17", "00"],
        expected_identifiers=["8h", "17h"],
    )

    assert evaluate_selective_answer_gate(result)["status"] == "passed"
