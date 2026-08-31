# Uploaded PDF graph enrichment v1

Status: predeclared before implementation

## Objective

Connect the existing per-document extraction and Neo4j components through one
small callable operation for a newly uploaded, already-extracted PDF. The
operation writes its supplied structural graph bundle, detects the established
French and Arabic relationship signals, persists only reference relationships
whose existing source and target verification checks pass, and returns the
remaining signals for review.

This is not a new upload API, extraction pipeline, legal consolidation engine,
or background job system.

## Public seam under test

`process_uploaded_pdf_graph` accepts exactly one validated source edition, its
already-extracted pages, the immutable source PDF, a verified instrument
catalog, optional reviewed citation identities, and an existing graph bundle
writer. It returns one receipt plus the deterministic candidates and promotion
decisions produced during the call.

## Contract

- reuse existing OCR, VLM, extraction, structural graph, relationship patterns,
  verification models, and Neo4j writer;
- process one PDF per call so a multi-PDF upload can call the operation once for
  each independent document;
- always write the supplied valid structural bundle through the existing
  idempotent writer;
- extract explicit BCT instrument-reference candidates and the existing legal
  action/provision signals from the same ordered page text;
- accept reviews only for candidate UIDs produced from the current immutable
  PDF and extraction artifact;
- persist only references that pass the existing hash, rendered-page, target
  identity, and catalog checks;
- return unreviewed or failed references and all unpromoted legal-effect signals
  as `NEEDS_REVIEW`; they must not become answer-bearing graph facts;
- preserve verified change events already supplied in the bundle without
  inventing missing targets, dates, predecessor text, or provision versions.

## Explicit exclusions

- no changes to PDF loading, OCR, VLM, chunking, embeddings, Chroma, or retrieval;
- no upload endpoint, SQL ledger, retry worker, queue, or batch transaction;
- no automatic promotion of `REPLACE`, `MODIFY`, `ABROGATE`, `ADD`, or related
  legal effects from keywords alone;
- no persistent Neo4j mutation during development verification;
- no evaluation-gold or hosted-model calls.

## Gate

KEEP only if one call writes one valid document bundle, returns deterministic
reference and legal-action candidates, persists a reviewed exact citation, does
not persist unreviewed or failed candidates, rejects stale review IDs, remains
idempotent through the existing writer, and passes focused and repository tests.
