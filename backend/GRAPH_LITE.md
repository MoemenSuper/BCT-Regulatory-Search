# BCT Graph Lite — automatic relationship mode

Graph Lite is a small Neo4j sidecar for relationships between BCT circulars,
notes and explicitly named provisions. It does **not** replace the existing
`cloud`, `local_hybrid` or `local` retrieval profiles.

## Automatic trust policy

There is no manual approval queue in this version.

Neo4j's Graph Lite uses Groq (`ChatGroq`) to extract only four relation
types: `CITES`, `AMENDS`, `REPLACES` and `ABROGATES`. The LLM result is never
written directly to the live graph. A deterministic Python gate first requires:

1. the relationship source to be the PDF currently being ingested;
2. the target circular/note to resolve to a real instrument in the trusted BCT
   corpus catalog;
3. the proposed evidence quotation to be recoverable verbatim from the exact
   supplied physical page text;
4. that quotation to explicitly name the target instrument;
5. `AMENDS`, `REPLACES` and `ABROGATES` quotations to contain matching legal
   action wording;
6. provision-level edges to identify provisions on both endpoints rather than
   mixing an instrument with a provision.

If every check passes, the edge is stored immediately as `VERIFIED` with
`verification_method=AUTO_DETERMINISTIC_V1`. If any check fails, the edge is
discarded. There is no manual review queue in normal operation.

This is intentionally less conservative than the previous manual-review build.
A valid quote and matching action word still do not mathematically prove the
LLM interpreted the legal sentence correctly. The exact evidence file, physical
page and quotation are retained on every live edge so the answer contract can
show the source of the relationship.

LLM-proposed effective dates are stored only as diagnostic metadata and are not
exposed to the live answer model. Graph Lite still does not claim complete
provision-level temporal consolidation.

## Install

```powershell
pip install -r requirements-graph.txt
```

Graph extraction uses the same Groq stack as the answer model (`langchain-groq`).
Builder imports remain isolated from normal API startup.

## Start local Neo4j

```powershell
$env:BCT_NEO4J_PASSWORD = "choose-a-strong-local-password"
docker compose -f docker-compose.graph.yml up -d
```

The provided compose file binds Neo4j to loopback for the laptop prototype.

## Environment

```text
BCT_ENABLE_GRAPH=1
BCT_NEO4J_URI=bolt://127.0.0.1:7687
BCT_NEO4J_USERNAME=neo4j
BCT_NEO4J_PASSWORD=...
BCT_NEO4J_DATABASE=neo4j
BCT_GRAPH_BUILDER_MODEL=openai/gpt-oss-120b
BCT_GRAPH_BUILDER_MAX_TOKENS=4096
```

The extractor uses the existing `GROQ_API_KEY` through `ChatGroq`. Live graph
retrieval itself makes no LLM call.

## Ingest one extracted page

```powershell
python graph_lite.py ingest-page `
  --source "Cir_2025_17_fr.pdf" `
  --page 3 `
  --text-file ".\page3.txt" `
  --catalog-dir ".\documents"
```

Pages without both a relationship-looking phrase and BCT instrument identity
are skipped before an LLM request.

The command prints the relationships that passed the automatic evidence gate.
Those relationships are already live; no approval command follows.

## Batch future ingestion

Your existing ingestion pipeline can emit:

```json
{"source":"Cir_2025_17_fr.pdf","page":3,"text":"...physical page text..."}
```

Then run:

```powershell
python graph_lite.py ingest-jsonl --input extracted_pages.jsonl --catalog-dir documents
```

`GraphLiteBuilder.ingest_page(...)` is also available directly from Python so a
future ingestion worker can update Neo4j immediately after extracting a page.

## Bootstrap the frozen corpus

```powershell
python graph_lite.py bootstrap-assets `
  --native-chunks "C:\path\to\runtime-assets\native.jsonl" `
  --catalog-dir ".\documents"
```

`--limit 20` and `--source Cir_2025_17_fr.pdf` are still useful for a cheap trial,
but they are no longer required for human review. The runtime chunk file itself
can also supply the trusted catalog when `--catalog-dir` is omitted.

## Inspect live relationships (optional)

Inspection is diagnostic only; it is **not** an approval workflow:

```powershell
python graph_lite.py list-relationships
```

## Runtime behavior

With `BCT_ENABLE_GRAPH=1`, relationship/currentness questions use an explicit
instrument identity from the question when available plus ordinary RAG results
as graph seeds. The live graph contains only automatically verified edges.

Graph evidence has a separate bounded context budget: up to **2 graph evidence
pages + the normal top 5 RAG pages**. This avoids forcing graph evidence to fight
for the original five slots, which was one failure mode in the older experiment.
Traversal is bounded to at most two hops.

- `cloud`: Voyage retrieval/rerank + Groq answer
- `local_hybrid`: local E5/BGE retrieval + Groq answer
- `local`: local E5/BGE retrieval + Ollama answer

Ordinary questions do not query Neo4j. Graph failure degrades to ordinary RAG.
`CITES` edges are excluded for currentness/temporal questions.

## Remaining limitations

- Relationship extraction is automatically trusted after deterministic evidence
  checks; there is no human adjudication.
- Exact quotation validation does not guarantee perfect legal interpretation.
- The graph does not infer a replacement merely from dates or newer filenames.
- A whole-instrument `REPLACES` edge does not prove which provision controls on
  a specific date.
- Proposed effective dates are not used in answers.
- There is no complete temporal lineage/consolidation engine yet.
- No graph embeddings or Neo4j vector search are used.
