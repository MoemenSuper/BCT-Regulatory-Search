"""Render a model-output-free human review packet for a validation candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.artifacts import sha256_file


def render_review_packet(
    *, validation_path: Path, audit_path: Path, validation: list[dict[str, Any]], audit: dict[str, Any]
) -> str:
    candidate_hash = sha256_file(validation_path)
    if audit["inputs"]["validation_candidate_sha256"] != candidate_hash:
        raise ValueError("Audit and validation candidate hashes differ")
    lines = [
        "# Validation candidate v1 — independent human review packet",
        "",
        f"Candidate SHA-256: `{candidate_hash}`  ",
        f"Audit SHA-256: `{sha256_file(audit_path)}`  ",
        "Retrieval/model access: **not run**",
        "",
        "Review every case against the cited public BCT PDF. Do not inspect retrieval or model outputs. "
        "For relevant cases, verify source, page, evidence, answer, and unambiguous wording. "
        "For negative cases, verify the expected behavior. Record corrections directly under the case.",
        "",
        "## Reviewer attestation",
        "",
        "- Reviewer name: ____________________",
        "- Review date: ____________________",
        "- Candidate hash confirmed: [ ]",
        "- I reviewed source documents without seeing model outputs: [ ]",
        "- Overall decision: APPROVE / CORRECT / REJECT",
        "",
    ]
    for index, case in enumerate(validation, start=1):
        lines.extend([f"## {index}. `{case['id']}`", "", f"Language: `{case['language']}`  ", f"Category: `{case['category']}`  "])
        if case["relevant"]:
            lines.extend(
                [
                    f"Source: `{case['expected_source']}`, page `{case['expected_page']}`  ",
                    f"Question: {case['query']}  ",
                    f"Expected answer: {case['expected_answer']}  ",
                    f"Evidence: {case['evidence_quote']}",
                    "",
                    "- [ ] Source and page are correct",
                    "- [ ] Evidence supports the complete expected answer",
                    "- [ ] Question is clear and has one supported interpretation",
                    "- [ ] No material condition, exception, number, or identifier is omitted",
                ]
            )
        else:
            lines.extend(
                [
                    f"Question: {case['query']}  ",
                    f"Expected behavior: `{case['expected_behavior']}`",
                    "",
                    "- [ ] Expected behavior is correct",
                    "- [ ] The question does not require a positive regulatory answer from this corpus",
                ]
            )
        lines.extend(["- Case decision: APPROVE / CORRECT / REJECT", "- Correction/reason:", "", "---", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    content = render_review_packet(
        validation_path=args.validation,
        audit_path=args.audit,
        validation=validation,
        audit=audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
