"""Suite B smoke: schema, size, and category balance for the human benchmark."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REQUIRED_CATEGORIES = {
    "lookup",
    "scenario",
    "it_depends",
    "comparative",
    "current_state",
    "multi_doc",
    "ambiguous",
    "long_scenario",
    "conversation",
}
REQUIRED_SHAPES = {
    "lookup",
    "yes_no",
    "it_depends",
    "clarify",
    "comparative",
    "multi_doc",
}

DATA = Path(__file__).resolve().parent / "human_questions.jsonl"


def _load() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_suite_b_has_at_least_100_items():
    items = _load()
    assert len(items) >= 100, f"Suite B too small: {len(items)}"


def test_suite_b_schema_and_ids_unique():
    items = _load()
    ids = []
    for item in items:
        assert "id" in item and "category" in item and "source_hint" in item
        assert "turns" in item and isinstance(item["turns"], list) and item["turns"]
        assert all(isinstance(t, str) and t.strip() for t in item["turns"])
        assert item["source_hint"] in {"none", "explicit"}
        assert item["category"] in REQUIRED_CATEGORIES
        exp = item["expected"]
        assert exp["answer_shape"] in REQUIRED_SHAPES
        assert isinstance(exp.get("instruments"), list)
        ids.append(item["id"])
    assert len(ids) == len(set(ids))


def test_suite_b_mix_no_source_and_explicit():
    items = _load()
    hints = Counter(i["source_hint"] for i in items)
    # Human desks usually omit the circular number — none should dominate.
    assert hints["none"] >= hints["explicit"] * 3
    assert hints["explicit"] >= 8


def test_suite_b_has_all_design_categories_and_conversations():
    items = _load()
    cats = {i["category"] for i in items}
    assert REQUIRED_CATEGORIES <= cats
    assert sum(1 for i in items if len(i["turns"]) > 1) >= 8
    assert sum(1 for i in items if i["expected"]["answer_shape"] in {"it_depends", "clarify"}) >= 15
    assert sum(1 for i in items if i["category"] in {"multi_doc", "comparative", "current_state"}) >= 20


def test_suite_b_is_multilingual():
    items = _load()
    langs = {i.get("lang", "fr") for i in items}
    assert {"fr", "en", "ar"} <= langs


def test_suite_b_arabic_parity_covers_categories_and_explicit_hints():
    """Arabic must not be a thin afterthought relative to French Suite B design."""
    items = _load()
    arabic = [i for i in items if i.get("lang") == "ar"]
    assert len(arabic) >= 30, f"Arabic Suite B too thin: {len(arabic)}"
    cats = {i["category"] for i in arabic}
    assert REQUIRED_CATEGORIES <= cats, f"Arabic missing categories: {REQUIRED_CATEGORIES - cats}"
    assert sum(1 for i in arabic if i["source_hint"] == "explicit") >= 6
    assert sum(1 for i in arabic if len(i["turns"]) > 1) >= 3
    assert sum(
        1
        for i in arabic
        if i["category"] in {"multi_doc", "comparative", "current_state"}
    ) >= 6
