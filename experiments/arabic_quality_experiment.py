"""Development-only screening experiment for Arabic extraction corruption."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from experiments.artifacts import sha256_file, write_json_atomic

QUALITY_THRESHOLD = 0.55
LATIN_RATIO_THRESHOLD = 0.20
SINGLE_ARABIC_TOKEN_RATIO_THRESHOLD = 0.10


def _current_gate(diagnostics: dict[str, Any]) -> bool:
    return float(diagnostics["quality_score"]) < QUALITY_THRESHOLD


def _trigger_reasons(diagnostics: dict[str, Any]) -> set[str]:
    reasons = set()
    if _current_gate(diagnostics):
        reasons.add("current_low_quality")
    if float(diagnostics["latin_character_ratio"]) > LATIN_RATIO_THRESHOLD:
        reasons.add("latin_heavy")
    if (
        float(diagnostics["single_arabic_token_ratio"])
        >= SINGLE_ARABIC_TOKEN_RATIO_THRESHOLD
    ):
        reasons.add("fragmented_arabic_tokens")
    return reasons


def _proposed_gate(diagnostics: dict[str, Any]) -> bool:
    return bool(_trigger_reasons(diagnostics))


def _confusion(
    cases: list[dict[str, Any]], gate: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    true_positive = sum(
        case["role"] == "failure" and gate(case["diagnostics"]) for case in cases
    )
    false_negative = sum(
        case["role"] == "failure" and not gate(case["diagnostics"]) for case in cases
    )
    false_positive = sum(
        case["role"] == "control" and gate(case["diagnostics"]) for case in cases
    )
    true_negative = sum(
        case["role"] == "control" and not gate(case["diagnostics"]) for case in cases
    )
    positive = true_positive + false_negative
    negative = false_positive + true_negative
    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "recall": true_positive / positive if positive else 0.0,
        "false_positive_rate": false_positive / negative if negative else 0.0,
    }


def evaluate_arabic_gate(suite: dict[str, Any]) -> dict[str, Any]:
    """Compare the existing and proposed fallback gates on frozen Arabic cases."""
    cases = [case for case in suite["cases"] if case.get("language") == "ar"]
    current = _confusion(cases, _current_gate)
    proposed = _confusion(cases, _proposed_gate)
    reasons = Counter()
    for case in cases:
        for reason in _trigger_reasons(case["diagnostics"]):
            reasons[f"{case['role']}:{reason}"] += 1

    keep = (
        proposed["recall"] - current["recall"] >= 0.25
        and proposed["false_positive_rate"] <= 0.10
    )
    return {
        "configuration": {
            "quality_score_below": QUALITY_THRESHOLD,
            "latin_character_ratio_above": LATIN_RATIO_THRESHOLD,
            "single_arabic_token_ratio_at_least": SINGLE_ARABIC_TOKEN_RATIO_THRESHOLD,
            "decision_gate": "recall improvement >= 0.25 and false-positive rate <= 0.10",
        },
        "current_gate": current,
        "proposed_gate": proposed,
        "trigger_reasons": dict(sorted(reasons.items())),
        "decision": (
            "KEEP_FOR_CONTROLLED_FALLBACK_COMPARISON"
            if keep
            else "REJECT"
        ),
        "limitations": [
            "Development stress set only; this does not establish generalization.",
            "Thresholds were explored on this same development suite, so recall and false-positive rates are resubstitution estimates.",
            "Extraction-failure labels derive from benchmark evidence coverage and require independent validation.",
            "A better trigger does not prove that the available OCR or VLM fallback is more faithful.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.stress_suite.read_text(encoding="utf-8"))
    artifact = {
        "status": "complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stress_suite_sha256": sha256_file(args.stress_suite),
        **evaluate_arabic_gate(suite),
    }
    write_json_atomic(args.output, artifact)


if __name__ == "__main__":
    main()
