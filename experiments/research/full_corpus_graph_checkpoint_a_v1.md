# Full-corpus graph checkpoint A v1

Status: `KEEP_CHECKPOINT_A`

## Decision

Keep the persistent Neo4j infrastructure, complete declared relationship
writer, and deterministic graph-audit seam as the foundation for Checkpoint B.
This decision does not claim a complete corpus graph or GraphRAG readiness.

## Frozen starting point

- branch: `experiment/temporal-provision-graphrag-20260827`
- handoff base: `ac35f89cf4fdcb44ea86d1d5f669dde75746a7b3`
- corpus: 439 PDFs under the production `documents` directory, read-only
- persistent root:
  `C:\Users\Moemen Super\BCT-Regulatory-Search-local-data\neo4j\temporal-regulatory-v1`
- Neo4j: `neo4j:2026.07.1` pinned by image digest
- database: `neo4j`
- hosted calls: zero
- evaluation gold used at runtime: false

## Checkpoint A seams

1. `RegulatoryGraphBundle` validates node references, verified temporal
   intervals, deterministic chunk positions, and same-provision supersession.
2. `Neo4jGraphWriter.write_bundle` supports every relationship family already
   declared in `regulatory_graph/schema.py`, including chunks, nested
   provisions, current/superseded versions, targets, and exact evidence links.
3. `Neo4jRegulatoryGraph.snapshot` produces a deterministic content hash across
   database row ordering.
4. `infrastructure/neo4j/compose.yaml` pins the image, binds HTTP/Bolt to
   loopback, requires an explicit persistent root, and bind-mounts data, logs,
   import, and backups outside Git.
5. The disposable live-test guard rejects the persistent Bolt port and still
   requires explicit confirmation, loopback, a non-default port, and an empty
   starting database.

## Physical persistence proof

The exact source fixture was verified against
`Cir_2016_03_fr.pdf` before writing. Two unchanged writes produced 20 nodes, 29
relationships, and the same graph content SHA-256:
`FC40CAF1E7D96612785F9CAA72F899F91D6019AB6A0C35038C57825BEAFF3E95`.

The Neo4j container was fully stopped and started from the same bind-mounted
directories. After restart, the content hash and counts were identical; Article
4 resolved on 2016-12-30 but not 2016-12-29, and its lineage still returned the
exact event with evidence pages 2 and 4 and incomplete predecessor coverage.

Neo4j Community rejected the first attempted online `STOP DATABASE` dump without
creating an archive. The supported offline path then fully stopped the server,
ran the pinned image once against the validated data and backup mounts, created
`neo4j.dump`, restarted the service, and reproduced the same graph hash.

## Disposable cleanup isolation proof

An unmounted disposable container ran at `bolt://localhost:27687`. All 14 live
tracer tests passed, the fixture-only cleanup left zero nodes, and the container
was removed. The persistent service remained on port 17687 with 20 nodes and is
now explicitly rejected by the disposable URI guard.

## Verification

- focused checkpoint tests: 32 passed, 2 optional skips
- repository suite: 207 passed, 2 optional skips
- disposable live tracer suite: 14 passed, zero nodes after cleanup
- Compose configuration: valid with the resolved persistent root
- `git diff --check`: passed
- persistent container health: healthy
- provisional answer validation: unopened
- final holdout: unopened

## Gate

Checkpoint A passes. Checkpoint B remains required to inventory and ingest all
439 PDFs, every page, and selected retrieval chunks. The persistent database
currently contains only the verified 20-node tracer, so it is not a complete
corpus graph and the graph arm remains not ready for GraphRAG claims.
