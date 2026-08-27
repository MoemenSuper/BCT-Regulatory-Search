# Temporal regulatory graph tracer v1

## Decision to make

Determine whether the verified graph core can persist and retrieve a small real
multi-change circular in Neo4j without duplicate facts, fabricated predecessor
text, or date-boundary errors.

This is a storage and deterministic-query tracer bullet. It does not test graph
retrieval, embeddings, answer generation, or GraphRAG orchestration.

## Frozen starting point

- Branch: `experiment/temporal-provision-graphrag-20260827`
- Base commit: `588481911e3bb489f06e841532ede5ed5f134bf6`
- Neo4j server image: `neo4j:2026.07.1`
- Python driver: `neo4j==6.2.0`
- Source PDF: `documents/Circulaires et notes 2016/Cir_2016_03_fr.pdf`
- Source SHA-256: `E463736EBB98BE5DBC6E02635E1D635734B9CDF4ACFD598F4F1BA8351DC43078`
- Source page count: 14

The source was visually checked against rendered PDF pages before this
predeclaration. The older target circular PDFs could not be found in the local
corpus using their cited identities, so those instruments are represented as
`EXTERNAL_STUB`. No missing predecessor text or historical version may be
invented.

## Frozen tracer facts

The fixture contains exactly these three independently evidenced replacement
events declared by BCT Circular 2016-03:

1. Source Article 2 replaces Circular 91-24 Article 4, effective 2016-12-30.
   Evidence and replacement text are on PDF page 2.
2. Source Article 5 replaces Circular 93-08 Annex 13, effective 2016-08-08.
   The replacement declaration is on PDF page 3. The annex content is not
   transcribed in this tracer; the introduced version records only the verified
   replacement declaration and therefore is not eligible as full provision
   text for answer generation.
3. Source Article 6 replaces Circular 91-24 Article 16, effective 2016-08-08.
   Evidence and replacement text are on PDF page 3.

Article 7 on PDF page 4 establishes 8 August 2016 as the general effective date
and 30 December 2016 as the exception for Articles 2 and 3. Each event must link
to both its operative evidence and the effective-date evidence needed to
justify its temporal boundary.

## Interfaces under test

- `Neo4jGraphWriter.write_bundle(bundle)` installs the approved schema and
  writes nodes and allowlisted relationships with static parameterized Cypher.
- Repeating the same write is idempotent.
- `Neo4jRegulatoryGraph.resolve_provision_as_of(provision_uid, as_of_date)`
  returns at most one verified version using half-open date intervals.
- `Neo4jRegulatoryGraph.lineage(provision_uid)` returns the declaring
  instrument, event action, effective date, introduced version, and exact
  source/page evidence. It explicitly marks the predecessor lineage incomplete
  when the replaced source version is absent from the corpus.

## Predeclared checks

1. Focused unit tests fail before implementation and pass afterward.
2. A disposable live Neo4j write creates the expected entities and edges.
3. Repeating the identical write leaves all node and relationship counts
   unchanged.
4. Circular 91-24 Article 4 resolves to no verified in-corpus version on
   2016-12-29 and to the introduced version on 2016-12-30 and later.
5. Circular 91-24 Article 16 resolves to no verified in-corpus version on
   2016-08-07 and to the introduced version on 2016-08-08 and later.
6. Lineage for the three targets reports exactly one replacement event each,
   the correct source filename and PDF pages, and incomplete predecessor
   coverage rather than a fabricated old version.
7. Every persisted event is VERIFIED, evidence-backed, and passes the existing
   activation/reference invariants.
8. The focused and repository-wide test suites pass and `git diff --check`
   reports no errors.

## Scope boundaries

- No production files, PDFs, Chroma collections, or indexes are modified.
- No LLM, VLM, embedding, or reranker call is made.
- No development-gold field affects graph construction.
- Provisional validation and final holdout remain unopened.
- Success authorizes only the next graph retrieval experiment; it does not
  authorize production integration.
