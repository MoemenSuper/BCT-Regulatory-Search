# Benchmark suites

Two suites, different jobs.

| Suite | Path | Purpose |
| --- | --- | --- |
| **A — Engineering / adversarial** | [`../stress/`](../stress/) | “Did we break something?” Precise, repetitive regression. Stage-labeled (`Failure stage = X`). |
| **B — Human benchmark** | [`suite_b/`](suite_b/) | “Would a banker, compliance officer, lawyer, or auditor ask this?” Natural language, scenarios, conversations. |

## Suite A

Keep running:

```text
cd backend
python -m pytest tests/stress/ -q
python -m tests.stress.report
```

Coverage matrix: [`../stress/COVERAGE.md`](../stress/COVERAGE.md).

## Suite B

Data: [`suite_b/human_questions.jsonl`](suite_b/human_questions.jsonl)

Each line is one item:

- `category`: `lookup` | `scenario` | `it_depends` | `comparative` | `current_state` | `multi_doc` | `ambiguous` | `long_scenario` | `conversation`
- `source_hint`: `none` (user did **not** hand the circular number) or `explicit`
- `lang`: `fr` | `en` | `ar` — Arabic is first-class: same categories, explicit-source anchors, and conversations as French
- `turns`: one or more user messages (conversations = multi-turn)
- `expected.answer_shape`: `lookup` | `yes_no` | `it_depends` | `clarify` | `comparative` | `multi_doc`
- `expected.instruments`: corpus PDF basenames that should be in play
- `expected.must_touch`: concepts a good answer should engage

Smoke:

```text
python -m pytest tests/benchmark/suite_b/test_suite_b_smoke.py -q
```

Live RAG scoring of Suite B needs a non-empty retrieval index; until then the JSONL is the gold set for human review and future eval harnesses.

### Design rules (Suite B)

1. Prefer **no-source** questions — retrieval must find the law without the answer key.
2. Include **explicit-source** questions — still needed for named-instrument anchors.
3. Include **it_depends** / **clarify** — missing conditions and underspecified asks.
4. Include **multi_doc** / **comparative** / **current_state** — amendment chains and “what applies today”.
5. Include **conversation** sequences — follow-ups like “Et pour 180 jours ?”.
6. Mix **FR / EN / AR** — real BCT desks are multilingual.

Grounded in corpus circulars (esp. 2025-13 export settlement, 2026-04 non-priority import financing, 2020-02 prior regime) plus public reporting on those texts (Juridoc, Tustex, WMC, African Manager).
