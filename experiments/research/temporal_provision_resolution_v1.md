# Temporal provision resolution v1

Status: predeclared before implementation

## Objective

Answer bounded questions about the rule applicable now or on an exact date by
selecting a verified provision version, not by assuming that the newest PDF
controls the whole subject. Reuse the existing ordinary retrieval, persistent
Neo4j graph, temporal models, verified relationship ingestion, and answer
generator without changing PDF extraction, OCR, VLM, chunking, embedding, or
vector-store ingestion.

## Public seams under test

1. `Neo4jRegulatoryGraph.affected_provisions` maps bounded ordinary-retrieval
   filename/page seeds to verified provision-level change targets.
2. `RelationshipGraphRetriever.retrieve` detects conservative French, Arabic,
   and English temporal-rule intent, selects one exact affected provision,
   calls the existing as-of and lineage queries, and returns a verified temporal
   context or an explicit incomplete state.
3. `conversation.chat` makes verified temporal context mandatory for a temporal
   answer and returns a deterministic abstention when the graph is absent,
   unavailable, ambiguous, or incomplete.

## Resolution contract

- use today's date only for explicit current/in-force wording;
- accept an explicit ISO `YYYY-MM-DD` or Tunisian-style `DD/MM/YYYY` date;
- treat an apparent historical request without an exact supported date as
  ambiguous and abstain;
- use ordinary top-five source filenames and one-based pages as bounded graph
  seeds;
- consider only `VERIFIED` change events and provision versions;
- select one candidate when the graph returns exactly one, or when the query
  explicitly names exactly one candidate provision label;
- require exactly one half-open provision version at the as-of date;
- require every applicable replacement event to have a resolved effective date,
  a retired predecessor version, and an introduced successor version;
- require the resolved version to match the last applicable introduced version;
- preserve the exact evidence filename, page, quote, effective interval, version
  UID, provision UID, and lineage event UIDs in the answer context;
- never mutate Neo4j at query time.

## Explicit exclusions

- no automatic verification or promotion of the 6,255 semantic candidates;
- no reconstruction of unobserved predecessor text;
- no generic legal consolidation engine or unrestricted graph traversal;
- no implicit year-end interpretation for a year-only question;
- no claim of corpus-wide coverage or answer-quality improvement;
- no production checkout, source PDF, Chroma, evaluation gold, or persistent
  Neo4j mutation.

## Gate

KEEP only if a complete provision replacement resolves to the verified version
valid at the exact date, a partial or ambiguous target cannot select an
unrelated provision, incomplete predecessor/effective/version lineage abstains,
ordinary non-temporal behavior remains unchanged, focused and repository tests
pass, disposable live behavior passes, both code reviews pass, and the
persistent graph fingerprint remains unchanged.
