# Minimal runtime GraphRAG v1

Status: predeclared before implementation

## Objective

Connect the existing local Neo4j graph to the retained Chroma plus BM25 plus BGE
RAG path without rebuilding or replacing the existing PDF extraction, OCR, VLM,
chunking, embedding, or reranking stages.

The already-kept per-document reference-ingestion seam remains responsible for
detecting, validating, and writing new verified citation relationships. This
slice implements the missing query-time use of relationships that are already
present in Neo4j.

## Public seams under test

1. `Neo4jRegulatoryGraph.relationship_evidence` accepts ordinary-retrieval seed
   filenames and returns bounded, exact-page evidence for verified reference or
   change relationships connected to their instruments.
2. `RelationshipGraphRetriever.retrieve` invokes Neo4j only for deterministic
   French, Arabic, or English relationship intent, converts graph evidence into
   source/page-linked retrieval documents, and returns an observable trace.
3. `conversation.chat` keeps the current retrieval and reranking path, uses its
   initial top results as graph seeds, reranks the union only when verified graph
   evidence exists, and otherwise preserves baseline behavior.
4. The FastAPI lifespan owns one optional local Neo4j driver and closes it on
   shutdown. If the local graph is unavailable, `/chat` remains usable through
   the ordinary RAG path.

## Scope

- reuse the existing Chroma, BM25, reranker, answer generator, and extracted
  document metadata;
- link ordinary candidates to graph instruments through `SourceEdition.filename`;
- traverse only the existing allowlisted `DECLARES_REFERENCE`,
  `DECLARES_CHANGE`, `TARGETS`, `HAS_PROVISION`, `EVIDENCED_BY`, `ON_PAGE`,
  `HAS_PAGE`, and `HAS_EDITION` relationship families;
- require `VERIFIED` graph facts;
- include incoming and outgoing relationships;
- return the evidence quote, filename, page number, relationship type,
  direction, and explainable instrument path;
- cap seed documents and returned evidence;
- make no graph mutation at query time.

## Explicit exclusions

- no OCR, VLM, extraction, chunking, embedding, or vector-store redesign;
- no SQL job ledger, upload API, background worker, or orchestration framework;
- no generic entity extraction or unrestricted graph traversal;
- no automatic promotion of candidate or `NEEDS_REVIEW` legal relationships;
- no LLM call for relationship routing;
- no production checkout, production Chroma, source PDF, benchmark gold, or
  persistent Neo4j mutation;
- no claim of answer-quality improvement before a separate frozen evaluation.

## Gate

KEEP only if relationship queries retrieve bounded verified graph evidence from
ordinary source seeds, non-relationship queries make no graph call, graph
unavailability preserves the existing RAG result, graph evidence retains exact
source/page/path provenance, focused and repository tests pass, disposable live
Neo4j behavior passes, and the persistent Checkpoint B graph snapshot remains
unchanged.
