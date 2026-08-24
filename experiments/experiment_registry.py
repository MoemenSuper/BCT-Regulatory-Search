"""Schema-checked append-only registry for controlled RAG experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_REQUIRED_FIELDS = (
    "experiment_id",
    "timestamp",
    "hypothesis",
    "changed_variable",
    "configuration",
    "model_identifiers",
    "providers",
    "code_commit",
    "dataset_hashes",
    "development_metrics",
    "validation_metrics",
    "language_metrics",
    "repairs",
    "regressions",
    "failure_distribution",
    "latency",
    "approximate_cost",
    "conclusion",
    "decision",
)


def _validate(entry: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"Experiment entry is missing required fields: {', '.join(missing)}")
    if entry["decision"] not in {"KEEP", "REJECT"}:
        raise ValueError("Experiment decision must be KEEP or REJECT")
    languages = entry["language_metrics"]
    if (
        not isinstance(languages, dict)
        or not {"fr", "ar"}.issubset(languages)
        or not languages["fr"]
        or not languages["ar"]
    ):
        raise ValueError("Experiment entry must include French and Arabic language metrics")
    hashes = entry["dataset_hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Experiment entry must include at least one dataset hash")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9A-Fa-f]{64}", value) is None
        for value in hashes.values()
    ):
        raise ValueError("Every dataset hash must be a complete SHA-256 hexadecimal digest")


def _read_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid registry JSON on line {number}") from error
    return entries


def append_experiment(path: Path, entry: dict[str, Any]) -> None:
    """Validate and atomically append one unique experiment entry."""
    _validate(entry)
    entries = _read_entries(path)
    if any(existing.get("experiment_id") == entry["experiment_id"] for existing in entries):
        raise ValueError(f"Experiment ID already exists: {entry['experiment_id']}")
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in entries),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--entry", type=Path, required=True)
    args = parser.parse_args()
    append_experiment(
        args.registry,
        json.loads(args.entry.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    main()
