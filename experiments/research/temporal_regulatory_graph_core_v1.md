# Temporal regulatory graph core v1

Status: predeclared before implementation

## Objective

Implement the first bounded GraphRAG vertical slice: an auditable domain contract
for provision-level legal changes and an idempotent Neo4j schema installer. This
checkpoint makes no retrieval, answer-generation, PDF-ingestion, embedding, or
production changes.

The graph is a regulatory-state layer, not an automatically generated source of
legal truth. Only verified, temporally resolved change events may be used by a
later materialized-current-state projection.

## Frozen starting point

- branch base: `8e1b14ae3f9c438f5094e4aae84f468a0320a3ab`
- source branch: `experiment/structured-ingestion-benchmark-20260823`
- isolated branch: `experiment/temporal-provision-graphrag-20260827`
- existing retrieval and final-answer artifacts remain untouched

## In scope

- closed vocabularies for instrument kind, provision type, legal action, target
  scope, version state, and verification state;
- immutable contracts for `Instrument`, `SourceEdition`, `Provision`,
  `ProvisionVersion`, `TargetSpan`, `EvidenceSpan`, and `ChangeEvent`;
- validation of half-open temporal intervals and verified-event evidence/target
  requirements;
- an explicit distinction between a verified legal effect and an effect whose
  effective date is resolved enough to materialize;
- deterministic, idempotent Neo4j uniqueness constraints and range indexes;
- tests through the public model and schema-installation interfaces.

## Out of scope

- LLM extraction;
- automatic legal-effect activation;
- semantic `Rule`, `ActorClass`, `Topic`, `Condition`, or `Constraint` nodes;
- graph retrieval, graph traversal, conversational state, or answer generation;
- Neo4j vector/full-text migration;
- production/main changes;
- provisional validation or final holdout access.

## Gate

KEEP this core checkpoint only if:

1. invalid or incomplete verified events fail locally before graph writes;
2. unresolved effective triggers cannot report temporal materialization readiness;
3. provision-version intervals enforce `[valid_from, valid_to)` ordering;
4. every core node label receives an idempotent UID uniqueness constraint;
5. the installer is testable through an injected Neo4j-driver-compatible seam;
6. focused tests and the existing repository test suite pass;
7. production and the pre-existing experimental worktree remain untouched.

Passing this gate authorizes only the next experimental slice. It does not
qualify the graph for retrieval, answers, validation, or production use.
