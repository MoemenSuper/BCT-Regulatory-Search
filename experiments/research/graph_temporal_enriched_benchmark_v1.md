# Graph-temporal enriched benchmark v1

## Decision to make

Measure what the current BCT retrieval pipeline and the current temporal graph can
actually support on the newly supplied graph-specific questions. This is a
readiness and retrieval experiment, not a claim that GraphRAG answer generation
already exists.

## Frozen input and selection

- User-supplied file: `evaluation_queries_enriched.json`
- SHA-256: `6DACE8538C9D5F47EFCA5E038DC4853B66A283803C1B0CF06A8B11DBD8DC99EB`
- Total records: 807
- Selection: every record where `requires_graph` is `true` and
  `evaluation_slice` is `graph_temporal`
- Selected records: 50 (44 French, 6 Arabic)
- All 50 selected records are relevant and are new IDs relative to the frozen
  697-question retrieval suite.

`requires_graph` and `evaluation_slice` are used only to select the evaluation
slice. They are not available to a runtime retriever or answer policy.

## Pre-scoring gold audit and corrections

All expected PDF filenames were resolved uniquely in the local corpus. The audit
checked 88 supplied evidence snippets over 35 source pages. Seventy-four snippets
had at least 95% contiguous token coverage in native extraction. Arabic extraction
corruption accounted for ten low-alignment snippets and was visually checked on
the three affected pages. Three French low-alignment pages were also visually
checked.

Two records contain a confirmed page-label defect in the supplied gold:

1. `graph_fr_2019_07_multi_office_02`: the old Article 3 one-person/one-office
   rule is on `Cir_2018_07_fr.pdf` page 3, not page 2.
2. `graph_fr_2019_07_start_deadline_03`: the old three-month activation rule is
   on `Cir_2018_07_fr.pdf` page 3, not page 2.

The original file remains unchanged. The experiment must emit a derived frozen
50-case slice and apply only these two corrections before scoring. The correction
receipt must preserve the original hash, record IDs, before/after page labels, and
exact replacement quotes extracted from PDF page 3.

## Frozen control

Run the selected 50 queries through the existing development retrieval pipeline:

1. structured native dense top 20;
2. structured native BM25 top 15;
3. additive Arabic OCR dense top 5 plus BM25 top 5 for Arabic queries;
4. exact candidate union;
5. `BAAI/bge-reranker-v2-m3`;
6. the retained runtime-observable document-identity reranker prefix;
7. stable source/page diversity.

No answer-model or hosted call is allowed. Existing frozen local Chroma and OCR
representations may be reused read-only.

## Metrics

Report separately for overall, French, and Arabic:

- primary exact page at 1 and 5, using `expected_source` and `expected_page`;
- complete required-source coverage at 5, 10, and 20;
- complete required source/page-pair coverage at 5, 10, and 20;
- mean required source/page-pair recall at 5, 10, and 20;
- failures split into missing from candidate union versus reranked below cutoff;
- local retrieval/reranking latency.

Because some questions require up to four instruments, primary Page@5 alone is
not a sufficient GraphRAG metric.

## Current graph readiness audit

Measure the current graph implementation without changing it:

- local source-edition filenames present in the graph fixture;
- selected cases with every required source represented in the graph;
- selected cases with at least one required source represented in the graph;
- whether a runtime graph retriever/orchestrator exists;
- whether the graph exposes an answer-context assembly seam.

Gold source fields may be used only after graph construction for coverage scoring.
They may not create graph nodes, relationships, routes, or runtime query results.

## Decision rule

Return `GRAPH_ARM_READY` only if all are true:

1. every selected case has all required source instruments represented in the
   graph;
2. a gold-blind runtime graph retrieval/orchestration path exists;
3. the graph can assemble source/page-linked evidence for answer generation;
4. all benchmark corrections and input hashes are frozen and auditable.

Otherwise return `GRAPH_ARM_NOT_READY` and report the exact missing coverage.
The non-graph control remains a diagnostic baseline and cannot make the graph arm
pass.

## Scope boundaries

- Do not modify production code, source PDFs, Chroma, or the user-supplied file.
- Do not use expected answers, expected sources, expected pages, or evidence quotes
  in runtime retrieval or graph construction.
- Do not open provisional validation or final holdout artifacts.
- Do not claim retrieval metrics measure answer correctness.
- Do not expand the graph in this experiment; first measure the current system.

