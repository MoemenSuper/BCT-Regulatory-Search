# Enriched GraphRAG end-to-end evaluation v1

## Decision

Measure answer, citation, grounding-contract, retrieval, and graph behavior on
the exact audited 50-case GraphRAG slice derived from the missing 807-record
user evaluation file.

## Frozen input

- Audited GraphRAG slice:
  `experiments/stress_suites/graph_temporal_evaluation_v1.json`
- SHA-256: `F77A6B0F5727CB2FD8820A787AA4BBC2906A8E3FE012A872553729065A6E167E`
- Cases: 50 relevant questions, 44 French and 6 Arabic
- The original 807-record source hash was
  `6DACE8538C9D5F47EFCA5E038DC4853B66A283803C1B0CF06A8B11DBD8DC99EB`,
  but that untracked file is no longer present. This run must not claim to
  evaluate the unavailable 757 remaining records.

## Exact runtime configuration

1. Frozen StructuredDocument native representation: 4,611 chunks.
2. Native dense top 20 plus BM25 top 15.
3. Arabic-only additive OCR dense top 5 plus BM25 top 5: 196 chunks.
4. Exact candidate union.
5. Runtime-observable document-identity prefix.
6. `BAAI/bge-reranker-v2-m3` and stable source/page diversity.
7. Top-five ordinary pages as gold-blind seeds to the read-only Neo4j runtime.
8. Verified graph evidence union, then the same reranker and diversity policy.
9. `openai/gpt-oss-120b`, structured claim-linked answer prompt v5, low
   reasoning, temperature 0.

This is the strongest currently integrated local GraphRAG path. It is not the
separate Voyage Context-4 plus Voyage reranker arm (configuration K), which has
not been integrated into this runtime.

## Isolation and safety

- Runtime receives only ID, query, language, and category. Expected answers,
  sources, pages, and quotes are used only after every runtime output is frozen.
- Neo4j is read-only and its complete snapshot must remain unchanged.
- Source PDFs, production Chroma, the empty worktree Chroma, provisional
  validation, and final holdout remain untouched.
- Hosted calls are limited to public BCT evidence and cached by case, model,
  prompt, suite hash, and exact user payload.
- Temporal ambiguity fails closed without an answer-model call.

## Reported outcomes

- answered versus safe abstention;
- complete required source/page retrieval at five;
- complete required source/page citation;
- citation-to-evidence and claim-to-evidence integrity;
- graph expansion/abstention counts and graph evidence contribution;
- numeric/identifier diagnostics, usage, latency, and provider failures;
- case-level semantic review against the frozen answer and evidence.

No result from this development slice authorizes production promotion.
