"""Failure-stage vocabulary shared by the stress suite."""
from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    RETRIEVAL = "Retrieval"
    COVERAGE = "Coverage"
    TEMPORAL = "Temporal"
    SELECTION = "Selection"
    GENERATION = "Generation"
    VALIDATION = "Validation"
    CITATION = "Citation"


def failure_message(stage: Stage, case_id: str, detail: str) -> str:
    return f"Failure stage = {stage.value}\ncase={case_id}\n{detail}"


def require(condition: bool, *, stage: Stage, case_id: str, detail: str) -> None:
    if not condition:
        raise AssertionError(failure_message(stage, case_id, detail))
