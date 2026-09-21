"""Emit a stage-labeled stress report (pass / xfail / fail)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.stress.cases import CASES
from tests.stress.test_stress_suite import _KNOWN_GAPS, run_case


def main() -> int:
    rows = []
    for case in CASES:
        status = "pass"
        detail = ""
        try:
            run_case(case)
            if case.id in _KNOWN_GAPS:
                status = "xpass"
                detail = _KNOWN_GAPS[case.id]
        except AssertionError as exc:
            if case.id in _KNOWN_GAPS:
                status = "xfail"
                detail = str(exc).split("\n", 1)[0]
            else:
                status = "fail"
                detail = str(exc)
        rows.append(
            {
                "id": case.id,
                "stage": case.stage.value,
                "title": case.title,
                "status": status,
                "detail": detail,
            }
        )

    out = Path(__file__).resolve().parents[3] / "tmp" / "stress_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = {key: 0 for key in ("pass", "xfail", "xpass", "fail")}
    for row in rows:
        counts[row["status"]] += 1
    print(
        f"stress report: {counts['pass']} pass, {counts['xfail']} xfail, "
        f"{counts['xpass']} xpass, {counts['fail']} fail -> {out}"
    )
    for row in rows:
        if row["status"] in {"fail", "xfail", "xpass"}:
            print(f"{row['status'].upper()} [{row['stage']}] {row['id']}: {row['detail']}")
    return 1 if counts["fail"] or counts["xpass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
