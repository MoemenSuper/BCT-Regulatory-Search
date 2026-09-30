# Benchmarks

Unit tests (`backend/tests/test_*.py`) protect the hard rules: grounding gates, ingestion
guarantees, auth, trusted paths. Answer quality is measured here, on real PDFs, not with
synthetic cases.

| Set | Path | Purpose |
| --- | --- | --- |
| Evaluation queries (807) | kept outside the repo (`evaluation_queries_enriched.json`) | Retrieval Page@k and answers over the regulatory corpus; 45 questions target instruments later amended (score separately) |
| Human-style targeted (60) | [`suite_stats/human_style_questions.json`](suite_stats/human_style_questions.json) | How a BCT tester actually asks: tables, charts, statistics PDFs, circular tables, FR/EN/AR; `accept` lists every valid value |
| Statistics (41) and circular tables (19) | [`suite_stats/`](suite_stats/) | Same facts phrased precisely, with page and verified literal |
| Retrieval pages (100) | [`suite_retrieval/`](suite_retrieval/) | Is an answer page in the top 5? 58 human-style questions (gold pages from `suite_stats`) + 42 real-user questions (FR/AR/EN) with verified pages; 34 held out (`split`), never used to tune. `run.py --assets <root> --wordings question\|translations\|all` |
| Suite B (human scenarios) | [`suite_b/human_questions.jsonl`](suite_b/human_questions.jsonl) | Scenario and conversation questions; gold set for review |

Rules:

1. Do not tune code or prompts against these questions until the change is written as a
   general rule; keep a held-out slice for final scoring.
2. Treat any answer whose diagnostics show a provider error (HTTP 413/429) as a failed
   measurement, not a refusal by the gates.
3. Numbers in statistics can legitimately appear in several publications; score the value,
   not only the page.
