# RAG technique prioritization for BCT Regulatory Search

Date: 2026-08-21

## Bottom line

There is no single revolutionary RAG technique that is likely to produce a large, safe gain on this corpus by itself. For BCT, the strongest next bet is **not agentic RAG, Self-RAG, RAPTOR, or a knowledge graph**. It is a retrieval-focused sequence:

1. **Preserve the real structure of the PDFs and add a visual/page fallback for tables, columns, and damaged extraction.**
2. **Test a stronger multilingual candidate retriever that does not compress every chunk into one vector**—most practically BGE-M3's dense, sparse, and multi-vector modes or a multilingual ColBERT model.
3. **Then fine-tune the winning retriever on clean BCT query–evidence pairs with same-document hard negatives**, while keeping the existing 697 cases completely held out.

This ordering follows the actual BCT failure shape. The current pipeline splits extracted pages every 1,000 characters with 200-character overlap, embeds each chunk with `multilingual-e5-small`, performs dense top-20 plus whitespace-tokenized BM25 top-15, reranks the union with BGE, and sends five chunks to the answer model. Its exact-page top-5 is 86.07% and exact-page top-20 is 92.45%. The prior Contextual Retrieval experiment improved top-20 but left top-5 essentially unchanged and introduced generated-label risk. The remaining problem is therefore less "the LLM needs more autonomy" and more "the right page must survive extraction, representation, and first-stage candidate generation."

The expected rankings below are BCT-specific judgments, not promises. Published gains come from different corpora and metrics and must not be transferred directly to BCT.

## BCT-specific ranking

| Rank | Technique | Likely value for BCT | Main metric it can improve | Cost / risk | Recommendation |
| ---: | --- | --- | --- | --- | --- |
| 1 | Structure-, table-, and layout-aware ingestion plus page-image fallback | High | Exact-page recall and table/OCR questions | Medium implementation cost; low factual risk if original page remains evidence | Test first |
| 2 | Multilingual late-interaction / multi-vector retrieval | Medium-high | Candidate recall and ranking of buried exact details | Larger index and more engineering | Test immediately after or alongside rank 1 |
| 3 | Fine-tuned multilingual embeddings with hard negatives | High potential, medium confidence | Domain-specific page retrieval | Requires new clean training labels; severe leakage risk | Do after selecting a stronger base retriever |
| 4 | Deterministic hierarchical child-to-parent/neighbor retrieval | Medium-high for answer completeness; medium for exact-page recall | Complete clauses, definitions, exceptions, and section context | Low hallucination risk; moderate parsing work | Strong practical addition |
| 5 | Controlled query expansion / legal query rewriting | Medium on short or vocabulary-mismatched questions | Recall for synonyms, abbreviations, French/Arabic variants | Query drift and invented legal terms | Use selectively, not on every query |
| 6 | Multi-query RAG / question decomposition | Medium for genuinely multi-part questions; low-medium otherwise | Multi-hop evidence coverage | More retrieval/reranking cost and candidate noise | Route only complex queries to it |
| 7 | Late chunking | Medium-low to medium | Chunks that depend on surrounding context | Requires a compatible long-context embedding stack | Small controlled experiment, not first |
| 8 | Knowledge graph / GraphRAG | Low for ordinary exact-page lookup; high for relationship questions | Amendments, supersession, references, entities, timelines | High ontology and verification cost | Add later for a distinct product capability |
| 9 | Self-reflective / corrective RAG | Low retrieval gain; medium safety potential | Answer support, citation quality, abstention | Extra calls or specialized training; self-checks can still be wrong | Treat as a safety layer, not the next retriever |
| 10 | Higher hierarchical RAG / RAPTOR-style summary trees | Low-medium for BCT's usual questions; higher for broad synthesis | Cross-section and whole-document reasoning | Generated-summary provenance and indexing cost | Prefer deterministic legal hierarchy first |
| 11 | Agentic RAG | No intrinsic gain; depends entirely on its tools and routing policy | Complex workflow orchestration | Highest complexity, latency, and nondeterminism | Last, after each underlying tool is proven |

Hybrid search is not in the ranking because it is already the measured BCT baseline. The useful question is which new retrieval channel or preprocessing method adds complementary evidence to it.

## Why the top three fit this corpus

### 1. Structure, tables, OCR, and page images

This is the highest-confidence recommendation because it addresses information loss before retrieval begins. BCT's current `RecursiveCharacterTextSplitter` knows nothing about titles, articles, numbered clauses, tables, headers, footnotes, columns, or reading order. If a number is detached from its row heading, an exception is split from its governing clause, or Arabic text is extracted in the wrong order, no embedding or agent can reliably reconstruct the authoritative evidence.

Recent evidence is unusually close to BCT's domain. A 2026 study of Portuguese administrative PDFs compared 19 PDF-to-RAG configurations and found that metadata enrichment and hierarchy-aware chunking contributed more to accuracy than the converter choice alone; its best automated setup used Docling with hierarchical splitting and image descriptions. The authors also reported that an exploratory GraphRAG configuration underperformed basic RAG. This is one small corpus with LLM-based answer scoring, so its percentages are not transferable, but the failure mode is directly relevant ([paper](https://arxiv.org/abs/2604.04948)). IBM's Docling paper describes layout analysis, reading-order reconstruction, and table-structure recognition rather than plain text extraction ([Docling paper](https://arxiv.org/abs/2408.09869)).

For pages that remain visually difficult, ColPali retrieves directly over page images with multi-vector late interaction and reported stronger results than text-parsing pipelines on visually rich document retrieval. A French adaptation improved Recall@1 by 6.63 points on TabFQuAD, which makes a French page-image fallback plausible, although Arabic/RTL performance and 8 GB GPU feasibility remain unproven ([ColPali paper](https://arxiv.org/abs/2407.01449)).

The safe BCT design is not "replace all text retrieval with screenshots." It is:

- retain immutable page text and page images;
- create chunks on document structure—title, section, article, paragraph, list, table—not character count alone;
- repeat verified headings and identifiers as metadata, not generated prose;
- serialize tables with their headers attached to each row or cell group;
- use page-image retrieval as a complementary channel for visually complex or low-confidence pages;
- cite and display the original PDF page, regardless of which channel found it.

### 2. Multilingual late interaction / multi-vector retrieval

The user's observation that the documents repeat many general words while one important phrase is buried inside a chunk is precisely where a single-vector dense embedding can struggle. A single vector compresses the whole chunk; a highly discriminative code, amount, exception, or institutional phrase can be diluted by the rest of the regulatory language.

ColBERT retains token-level document vectors and scores fine-grained query-token/document-token matches at query time. ColBERTv2 reported strong in-domain and out-of-domain retrieval while reducing late-interaction storage by 6–10 times relative to the original design ([ColBERTv2](https://arxiv.org/abs/2112.01488)). Jina-ColBERT-v2 applies this architecture to multilingual retrieval, though its aggregate multilingual results are not proof of Arabic/French regulatory performance ([Jina-ColBERT-v2](https://arxiv.org/abs/2408.16672)).

BGE-M3 is the most compact BCT experiment because one multilingual model exposes dense, learned-sparse, and multi-vector retrieval, supports more than 100 languages, and accepts inputs up to 8,192 tokens ([BGE-M3 paper](https://arxiv.org/abs/2402.03216)). It can test three hypotheses independently:

- whether a stronger dense model alone beats `multilingual-e5-small`;
- whether learned multilingual sparse matching beats the current `lower().split()` BM25 path;
- whether token-level multi-vector scores rescue pages whose decisive detail is buried in repetitive prose.

This is **late interaction**, not **late chunking**. Late interaction keeps multiple token vectors and compares them with the query during retrieval. Late chunking first contextualizes tokens across a long input and then pools token spans into ordinary chunk vectors.

### 3. Domain fine-tuning with difficult negatives

Fine-tuned embeddings have the largest theoretical ceiling among the techniques the user listed, but they are not the safest first experiment. BCT first needs a strong base architecture and a training set that is independent of its evaluation set.

E5 itself was trained contrastively and improved further with labeled fine-tuning ([E5 paper](https://arxiv.org/abs/2212.03533)); multilingual E5 used contrastive pretraining on one billion multilingual pairs followed by labeled fine-tuning ([multilingual E5 report](https://arxiv.org/abs/2402.05672)). Legal retrieval studies also show a meaningful gap between zero-shot and fine-tuned passage retrieval under domain shift ([legal paragraph retrieval](https://aclanthology.org/2024.lrec-main.1177/)). The 2026 LEMUR work is even closer: it fine-tuned multilingual retrievers over noisy PDF-derived legislation and reported consistent top-k gains, especially in lower-resource languages ([LEMUR](https://arxiv.org/abs/2602.09570)).

For BCT, the important ingredient is not merely more positive pairs. It is **hard negatives that share the same regulatory vocabulary but contain the wrong page, wrong year, wrong circular, neighboring amount, or related-but-inapplicable rule**. Random chunks from unrelated PDFs will be too easy and will not teach the model the distinctions that currently cause regressions.

The existing 697-query benchmark must remain untouched as the test set. Training directly on it and then reporting its score would invalidate the only broad regression detector. Create new training queries or establish document/version-separated train, development, and locked test partitions first. Also preserve Arabic and French balance and inspect mined hard negatives for false negatives.

## Assessment of every requested technique

### Self-reflective RAG

The published Self-RAG method is not simply asking the existing answer model to "check its work." It trains a language model to emit special reflection tokens, decide when to retrieve, assess passage relevance, and critique its own output. The paper reports improvements in factuality and citation accuracy across open-domain tasks ([Self-RAG](https://arxiv.org/abs/2310.11511)). That evidence concerns the generator and adaptive retrieval behavior, not a guaranteed improvement to BCT's exact-page candidate recall.

For a regulatory system, a second unconstrained LLM opinion is not a safety proof. A more defensible safety layer is claim-level support checking against the cited original text, retrieval sufficiency thresholds, explicit document-status checks, and calibrated abstention. Research on risk-controlled RAG explicitly evaluates answers with an abstain option ([RC-RAG](https://aclanthology.org/2024.findings-emnlp.133/)). This should be developed in parallel with retrieval work, but measured on answer correctness, unsupported claims, citation entailment, and the eight negative cases—not counted as a retrieval improvement.

Corrective RAG similarly uses a learned retrieval evaluator and, when retrieval is weak, may search the web ([CRAG](https://arxiv.org/abs/2401.15884)). Web fallback is inappropriate for a closed authoritative BCT corpus unless external official sources are explicitly approved, versioned, and cited. The useful idea to borrow is the fail-closed retrieval-quality gate, not the web-search workflow.

### Higher hierarchical RAG

This label can refer to two substantially different designs:

1. **Deterministic legal hierarchy:** retrieve a small article/paragraph child, then attach its verified title, parent article, and nearby clauses. This is recommended. Dense hierarchical retrieval research represents documents as title trees and retrieves at document and passage levels ([DHR](https://aclanthology.org/2021.findings-emnlp.19/)). Legal research has also found gains from incorporating statute hierarchy into contrastive retrieval ([legal hierarchy retrieval](https://arxiv.org/abs/2203.02259)).
2. **Generated hierarchy such as RAPTOR:** cluster chunks, recursively summarize them with an LLM, and retrieve at multiple abstraction levels. RAPTOR's strongest results concern complex, multi-step reasoning over long texts, including a 20-point absolute QuALITY gain when paired with GPT-4 ([RAPTOR](https://arxiv.org/abs/2401.18059)). That does not predict a similar gain on exact circular/page questions.

The first design preserves authority and traceability. The second creates a generated-summary layer with the same basic provenance risk already observed in BCT's Contextual Retrieval experiment. RAPTOR is worth testing only if a separately labeled set of cross-section or whole-document questions establishes a real product need.

### Late chunking

Late chunking processes a long document through a long-context embedding model before pooling the token spans that correspond to individual chunks. It lets a chunk representation inherit surrounding context without generating new words, and the paper reports gains across multiple retrieval tasks without additional training ([late chunking](https://arxiv.org/abs/2409.04701)).

It is safer than LLM-generated contextual prefixes, but it is not a drop-in toggle for BCT's current short-context `multilingual-e5-small`. Long BCT PDFs also require windowing or a documented long-document variant. The paper's gains are strongest where small chunks lose necessary context; ordinary chunking can remain competitive for larger chunks. Test late chunking only after structure-aware boundaries exist, otherwise it contextualizes an arbitrary 1,000-character segmentation rather than fixing it.

### Multi-query RAG

Multi-query retrieval asks an LLM for several paraphrases or facets, retrieves for each, and fuses the ranked lists. It can help ambiguous queries and multi-hop questions, but it also multiplies latency and candidate noise. A RAG-Fusion study itself notes that insufficiently relevant generated queries can make answers stray off topic ([RAG-Fusion](https://arxiv.org/abs/2402.03367)). A 2026 production-style study found that fusion increased raw recall but its gains were largely neutralized after reranking and truncation; some configurations lowered top-k accuracy ([industry retrieval-fusion study](https://arxiv.org/abs/2603.02153)).

That warning maps closely to BCT: it already has a strong BGE reranker and only five final chunks. More queries may enlarge the union without changing which five survive. Use two or three bounded variants only for a detected ambiguous, multilingual, or multi-part query, retain the exact original query in every fusion, and prohibit changing numbers, dates, circular identifiers, negation, or legal actors.

For genuinely compound questions, question decomposition is more purposeful than generic paraphrasing: retrieve each sub-question, merge the candidates, and rerank them. A 2025 study reports strong gains on multi-hop benchmarks with this pattern ([question decomposition](https://arxiv.org/abs/2507.00355)). BCT needs a labeled multi-document/multi-clause subset before deciding how often this path should run.

### Query expansion

Query expansion adds likely terms or a pseudo-document to the query. Query2doc reported 3–15% BM25 gains on MS MARCO and TREC DL and smaller gains for already-strong dense retrievers ([Query2doc](https://aclanthology.org/2023.emnlp-main.585/)). HyDE also reports strong zero-shot retrieval by embedding an LLM-generated hypothetical document, while explicitly acknowledging that the hypothetical document can contain false details ([HyDE](https://arxiv.org/abs/2212.10496)).

The safe BCT version is narrower: expand verified abbreviations, official institution names, Arabic/French transliterations, morphology, and controlled terminology from a curated lexicon; or generate candidates and discard any expansion that changes a number, date, code, negation, named instrument, or institution. Generated expansion is a retrieval hint only and must never appear as evidence. Because BCT already has BM25 plus dense retrieval, expected gains are moderate and concentrated in short or vocabulary-mismatched queries.

### Knowledge graphs

Knowledge graphs are a strategically valuable BCT feature, but they solve a different class of questions. Microsoft's GraphRAG work targets global corpus-wide sensemaking by extracting entities and relationships, grouping them into communities, and generating community summaries. Its reported improvements concern comprehensiveness and diversity for global questions, not exact provision/page retrieval ([Microsoft GraphRAG paper](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/)). Microsoft also emphasizes that suitability depends on whether those benefits outweigh expensive index construction ([official project discussion](https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/)).

BCT should eventually build a **regulatory provenance graph**, not a generic LLM-generated topic graph. High-value typed edges include `amends`, `repeals`, `supersedes`, `implements`, `refers_to`, `issued_by`, `effective_from`, and `applies_to`. Every node and edge should point to a verified source page and retain uncertainty/review state. Use the graph to select related instruments and status context, then return to hybrid text/page retrieval for evidence. This can materially improve amendment, lineage, and cross-document questions even if it does not improve ordinary top-5 page lookup.

### Agentic RAG that includes hybrid search

"Agentic RAG" is an orchestration pattern, not a retriever. An agent can choose hybrid search, graph traversal, query decomposition, document-status lookup, or abstention, but it inherits the quality of those tools and adds routing errors. Adaptive-RAG research supports selecting different strategies according to question complexity rather than applying an expensive workflow to every query ([Adaptive-RAG](https://arxiv.org/abs/2403.14403)). Active retrieval work similarly shows value for repeated retrieval during long-form generation, a different workload from most precise BCT lookups ([FLARE](https://arxiv.org/abs/2305.06983)).

BCT already has a limited router that rewrites conversational follow-ups before hybrid retrieval. Expanding this into an autonomous loop before the individual tools are benchmarked would make regressions harder to attribute. The right order is: prove each retrieval channel, define deterministic routing conditions, cap tool calls and latency, log every decision, and fail closed when evidence is insufficient. Agentic orchestration comes last.

## Additional techniques worth prioritizing

### Learned multilingual sparse retrieval

The current BM25 implementation lowercases and splits on whitespace. It does not normalize punctuation, Arabic forms, clitics, or French morphology, and it cannot learn useful term expansion. Before adding an LLM, test better deterministic normalization/tokenization and BGE-M3's multilingual sparse head. SPLADE is the established learned-sparse alternative and reported more than 9% nDCG@10 improvement on TREC DL 2019 ([SPLADE v2](https://arxiv.org/abs/2109.10086)), but its evidence is heavily English and a separate SPLADE deployment is less attractive than a BGE-M3 ablation for this small bilingual corpus.

This change is especially plausible because exact circular numbers, dates, amounts, and legal phrases benefit from lexical matching while semantically similar repeated prose should not dominate. Preserve raw exact tokens as a separate feature; stemming must not merge identifiers or values.

### Evidence-window retrieval

Retrieve and score the smallest authoritative unit, but provide the answer model with a deterministic evidence window: section heading, hit paragraph or table row, and bounded neighboring clauses. This "small-to-big" pattern avoids asking one embedding to be both a precise search unit and a complete answer context. Research on long legal documents has found that targeted sequential or semantically selected neighboring context can outperform heavier hierarchical models ([legal context-retrieval study](https://aclanthology.org/2025.argmining-1.15/)).

Crucially, retrieval success should still be scored against the exact child/page. Expanding the context after retrieval must not disguise a miss or weaken page-level citations.

### A visual fallback, not a visual-only system

Table, stamp, signature, multi-column, scanned, and Arabic-layout pages should be tagged during ingestion. For those pages, fuse text retrieval with page-image retrieval. This directly attacks extraction failures without imposing multimodal cost on every clean text page. Any extracted answer still requires visual page verification and a precise source location.

## Retrieval improvement versus answer safety

These are separate objectives and should have separate release gates.

| Retrieval evaluation | Answer-safety evaluation |
| --- | --- |
| Exact page/source top-1, top-5, top-20 | Correctness against the evidence quote |
| Arabic and French reported separately | Every material claim entailed by cited original text |
| Tables/OCR/long-doc/ordinary strata | Exact source and page attribution |
| Repairs and regressions versus current hybrid | Correct abstention on missing or insufficient evidence |
| Candidate recall before BGE and survival after BGE | Document status, amendment, and supersession handling |
| Latency, index size, GPU/RAM cost | No generated metadata, graph edge, or query expansion treated as evidence |

Self-reflection can improve the right column while doing little for the left. A new embedding or late-interaction retriever can improve the left while making the right worse if it introduces irrelevant-but-plausible context. Neither should be called a RAG improvement unless end-to-end tests pass.

## Recommended experiment sequence

### Experiment 1: failure attribution before another full rebuild

Take all current exact-page top-5 failures and classify them, with a blinded sample of successes, into:

- extraction/OCR/reading-order loss;
- table or layout loss;
- bad chunk boundary or missing heading;
- candidate-generation miss;
- correct candidate retrieved but removed by BGE;
- correct top-five evidence but answer/citation failure;
- ambiguous or defective evaluation label.

This is inexpensive and determines how much headroom each technique can actually address. If many correct pages are absent from extracted text, fine-tuning embeddings cannot be the first fix. If correct chunks are already in the union but BGE drops them, changing query expansion or first-stage retrieval may have little effect.

### Experiment 2: structure-aware and multi-vector retrieval bake-off

Build separate disposable indexes and keep the same 697 queries, candidate budget, BGE reranker, and final top-five:

1. current hybrid baseline;
2. structure-aware text chunks with deterministic headings/table serialization;
3. stronger dense-only BGE-M3;
4. BGE-M3 dense plus sparse;
5. BGE-M3 dense plus sparse plus multi-vector, if hardware permits;
6. structure-aware retrieval plus a visual fallback for tagged difficult pages.

This isolates where the gain comes from. Do not change chunking, embedding model, candidate counts, and reranker simultaneously in the promoted result without the ablations.

### Experiment 3: domain fine-tuning

Only if Experiment 2 selects a stronger base retriever:

- create new training queries from held-out source pages;
- include Arabic, French, and realistic cross-language phrasing;
- mine same-document and same-topic hard negatives;
- manually remove false negatives;
- split by document family/version to prevent near-duplicate leakage;
- freeze the 697-query suite as the final test;
- require language-specific improvements and manually review all new regressions.

### Parallel safety track

Independently test retrieval sufficiency, claim-to-evidence verification, document status, and abstention. A technique does not earn production promotion from recall alone. The eight negative cases are too few for a high-trust regulatory release gate, so add more adversarial unanswerable questions, wrong-premise questions, superseded-rule questions, and questions whose answer exists only in an OCR/table failure.

## Final recommendation

If only one technique can be tested next, test **structure-aware ingestion/chunking plus a multilingual multi-vector candidate retriever** as a controlled bake-off. If only one model family can be tested, use **BGE-M3** because its dense, sparse, and multi-vector outputs allow three useful ablations from one multilingual model. If the result is strong, fine-tune that retriever with BCT-specific hard negatives.

Do not invest first in an agent, a recursive summary tree, or generic self-reflection. Those methods add intelligence after the most important failure may already have happened: the authoritative clause, table row, or page never entered the candidate set in a usable form.
