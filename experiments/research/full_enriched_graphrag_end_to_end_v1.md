# Full enriched GraphRAG end-to-end evaluation v1

## Objective

Run the complete restored `evaluation_queries_enriched.json` through the
currently integrated retrieval, read-only Neo4j GraphRAG, and structured answer
path. Report retrieval, answer-status, citation-contract, graph, provider,
latency, and usage outcomes without tuning from gold fields.

## Frozen input

- User-owned source file:
  `C:/Users/Moemen Super/BCT-Regulatory-Search/evaluation_queries_enriched.json`
- SHA-256:
  `6DACE8538C9D5F47EFCA5E038DC4853B66A283803C1B0CF06A8B11DBD8DC99EB`
- Cases: 807 total; 779 relevant and 28 negative.
- Languages: 456 French and 351 Arabic.
- Slices: 689 retrieval regression, 50 graph temporal, 40 user simulation,
  and 28 robustness.

The source file remains untracked and read-only. Runtime receives only case ID,
query, language, and category. Expected answers, sources, pages, behavior, and
quotes are used only after all responses are frozen.

## Runtime configuration

1. Frozen StructuredDocument native representation: 4,611 chunks.
2. Native dense top 20 plus BM25 top 15.
3. Arabic-only additive OCR dense top 5 plus BM25 top 5: 196 chunks.
4. Exact candidate union.
5. Runtime-observable document-identity prefix.
6. `BAAI/bge-reranker-v2-m3` and stable source/page diversity.
7. Top-five ordinary pages seed the read-only Neo4j runtime.
8. Verified graph evidence is unioned and reranked with the same BGE policy.
9. `openai/gpt-oss-120b`, structured claim-linked answer prompt v5, low
   reasoning, temperature 0.

This is the strongest fully integrated local GraphRAG path. It is not the
separate Voyage Context-4 plus Voyage reranker configuration K.

## Safety and completion

- Public BCT pages only; hosted responses are cached by suite hash, model,
  prompt, case, and exact payload.
- Neo4j must remain byte-for-byte unchanged before and after the run.
- Production Chroma, PDFs, provisional validation, and final holdout remain
  untouched.
- A completed run requires 807 frozen runtime records, 807 scored records, no
  provider failure, a matching graph snapshot, valid JSON artifacts, and a
  clean test pass.
- Retrieval and structural citation metrics do not establish semantic answer
  correctness. The prior 50-case graph slice retains its separate primary-agent
  semantic review; no claim of manual review will be made for all 807 cases.

No result authorizes production promotion.
