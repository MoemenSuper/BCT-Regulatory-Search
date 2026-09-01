# Runtime GraphRAG 50-case evaluation v1

## Question

How does the currently implemented GraphRAG runtime behave on the 50 frozen
`requires_graph=true`, `evaluation_slice=graph_temporal` questions from
`evaluation_queries_enriched.json`?

## Frozen inputs

- Original enriched evaluation SHA-256:
  `6DACE8538C9D5F47EFCA5E038DC4853B66A283803C1B0CF06A8B11DBD8DC99EB`
- Audited derived 50-case slice SHA-256:
  `F77A6B0F5727CB2FD8820A787AA4BBC2906A8E3FE012A872553729065A6E167E`
- Ordinary-retrieval control result SHA-256:
  `57E605B3582A4D09164D538291F5F314E3EDA8E9F99C492B48578F6CF409EA86`
- Runtime commit: `9a372febc4354e53c070f82958cb85eb74999516`
- Fixed current date for deterministic current-rule routing: `2026-09-01`

The original enriched evaluation file must not be modified. The derived slice
contains the two already-audited page-label corrections documented by the
earlier benchmark.

## Gold isolation

Before any graph call, construct a runtime-only record containing exactly:

- case ID;
- query;
- language;
- category;
- the prior control's top-five source/page results as retrieval seeds.

Expected sources, pages, quotes, answers, graph paths, and relationship types
must not be passed to the router, the graph retriever, or Neo4j. Runtime outputs
are collected for every case before scoring begins.

## What is measured

1. Relationship and temporal routing counts.
2. Graph status, temporal status, failure reason, returned evidence count, and
   graph-only latency per question.
3. Required source/page coverage of returned graph evidence, scored only after
   all runtime outputs are frozen.
4. Whether graph evidence adds a missing required source/page to the ordinary
   top-five seeds.
5. Read-only safety by comparing the complete Neo4j snapshot before and after.

## Boundaries

- The persistent Neo4j database is read-only for this evaluation.
- No hosted model or answer-generation call is made.
- This evaluates runtime routing and evidence retrieval, not final answer
  correctness.
- The ordinary retrieval seeds are reused from the completed frozen control;
  embeddings and reranking are not rerun.
- A conservative abstention is counted separately from an incorrect answer.

## Success interpretation

This is diagnostic, not a promotion gate. A useful result must distinguish:

- questions routed to GraphRAG;
- questions left on ordinary retrieval;
- routed questions with verified evidence;
- safe temporal abstentions;
- missing graph inventory or unsupported wording;
- any database mutation, which is an automatic failure.
