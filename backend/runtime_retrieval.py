"""Local retrieval (default for local_hybrid and local): e5 + Chroma, BM25, BGE reranker.

The cloud profile's Voyage backend lives in cloud/voyage_retrieval.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from bm25 import retrieve_bm25
from retrieval_selection import (
    diversify_ranked_pages,
    build_identity_reranker_documents,
    expand_answer_pages,
    page_chunks,
    parse_query_identity,
    prefer_historical_hits,
    prefer_named_instrument_hits,
    query_instrument_refs,
    source_matches_identity,
    is_arabic_query,
)
from ingestion.quality import arabic_character_ratio
from reranker import score_documents
from vector_store import retrieve_relevant_chunks
from langchain_core.documents import Document


def _doc_key(document):
    return (
        document.page_content,
        document.metadata.get("source"),
        document.metadata.get("page"),
    )


def dedupe(*groups):
    """Deduplicate documents across retrieval lanes by content + source + page."""
    combined = []
    seen = set()
    for documents in groups:
        for document in documents:
            key = _doc_key(document)
            if key not in seen:
                seen.add(key)
                combined.append(document)
    return combined


def _is_arabic_text(text):
    # Same threshold extraction uses to call a page Arabic.
    return arabic_character_ratio(text) >= 0.20


def script_scores(score, query, other_queries, documents):
    """Each chunk's reranker score against one wording written in its own script.

    score(wording, documents) returns one score per document. A French chunk is compared with the
    first Latin-script wording (the question, or its French version), an Arabic chunk with the
    first Arabic one. The other wordings still find candidates (each search lane), but scoring
    every chunk against every same-script wording too cost 2.3x the reranker work for the same
    result (retrieval benchmark: 82 % top-5 both ways; 7.7k instead of 18k pairs). The reranker
    is 97 % of an answer's time on a CPU server.
    """
    wordings = [query, *other_queries]
    scores = [None] * len(documents)
    for arabic in (False, True):
        indices = [i for i, document in enumerate(documents) if _is_arabic_text(document.page_content) == arabic]
        if not indices:
            continue
        wording = next((w for w in wordings if is_arabic_query(w) == arabic), query)
        for i, value in zip(indices, score(wording, [documents[i] for i in indices])):
            scores[i] = float(value)
    return scores


def promote_agreed_hit(ranked, agreed, slots=5):
    """Keep in the top `slots` a chunk that both searches rank in their first three for one wording.

    The reranker judges a chunk by how closely it matches the question's words, so it can pass
    over a rule written differently: asked for the "ratio de solvabilité minimum", it prefers the
    dividend circulars that repeat those words over "un ratio de solvabilité qui ne peut pas être
    inférieur à 10 %". When the meaning search and the keyword search both rank that rule among
    their first three (the answer sketch is worded like it), it takes the last of the top places.
    At most one chunk moves; the rest of the order is the reranker's.
    """
    top_pages = {_page(document) for document, _score in ranked[:slots]}
    for chunk in agreed:
        if _page(chunk) in top_pages:
            continue
        position = next((i for i, (document, _score) in enumerate(ranked) if _doc_key(document) == _doc_key(chunk)), None)
        if position is None:
            continue
        moved = ranked[position]
        rest = ranked[:position] + ranked[position + 1:]
        return rest[:slots - 1] + [moved] + rest[slots - 1:]
    return ranked


def _page(document):
    return document.metadata.get("source"), document.metadata.get("page")


def _identity_diversified_rank(query, documents, scores):
    scores = [float(score) for score in scores]
    if len(scores) != len(documents):
        raise ValueError("Reranker returned an invalid score count")
    ranked = diversify_ranked_pages(
        sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
    )
    # Named before historical: "Avant 2025-13" must demote the named cutoff last.
    ranked = prefer_named_instrument_hits(ranked, query_instrument_refs(query))
    return prefer_historical_hits(ranked, query)


class LocalRetrievalBackend:
    def __init__(
        self,
        vector_store,
        reranker,
        bm25,
        bm25_documents,
        *,
        ocr_vector_store=None,
        ocr_bm25=None,
        ocr_documents=None,
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.bm25 = bm25
        self.bm25_documents = bm25_documents
        self.ocr_vector_store = ocr_vector_store
        self.ocr_bm25 = ocr_bm25
        self.ocr_documents = list(ocr_documents or [])
        self._pages = page_chunks([*bm25_documents, *self.ocr_documents])

    def expand_pages(self, ranked):
        """Full retrieved page text, plus bounded same-PDF neighbour pages."""
        return expand_answer_pages(ranked, self._pages)

    def retrieve(self, query, other_queries=(), instruments=()):
        """Rank chunks for a question.

        other_queries: other wordings of the same question (the user's own words, its French and
        Arabic versions). Each one finds its own candidates, so an English question also reaches a
        French circular.
        instruments: extra instruments to search inside, like the ones the question names (for
        "avant la circulaire X", the texts X replaced).
        """
        groups = [
            retrieve_relevant_chunks(query, self.vector_store),
            retrieve_bm25(query, self.bm25, self.bm25_documents),
        ]
        lanes = ["dense", "bm25"]
        agreed = []  # chunks both searches rank in their first three for one wording (see promote_agreed_hit)
        if is_arabic_query(query) and self.ocr_vector_store is not None:
            groups.append(retrieve_relevant_chunks(query, self.ocr_vector_store, k=5))
            groups.append(retrieve_bm25(query, self.ocr_bm25, self.ocr_documents, k=5))
            lanes += ["ocr_dense", "ocr_bm25"]
        # The other wordings bring fewer candidates each: they add what the main query misses.
        for version in other_queries:
            groups.append(retrieve_relevant_chunks(version, self.vector_store, k=10))
            groups.append(retrieve_bm25(version, self.bm25, self.bm25_documents, k=8))
            lanes += ["dense (other wording)", "bm25 (other wording)"]
            dense_top = {_doc_key(document) for document in groups[-2][:3]}
            agreed += [document for document in groups[-1][:3] if _doc_key(document) in dense_top]
            if is_arabic_query(version) and self.ocr_vector_store is not None:
                groups.append(retrieve_relevant_chunks(version, self.ocr_vector_store, k=5))
                groups.append(retrieve_bm25(version, self.ocr_bm25, self.ocr_documents, k=5))
                lanes += ["ocr_dense (other wording)", "ocr_bm25 (other wording)"]
        identity_refs = query_instrument_refs(query) + list(instruments)
        for identity in identity_refs:
            # ponytail: first 40 chunks of each named instrument; the reranker picks among them
            matched_docs = [
                document
                for document in self.bm25_documents
                if source_matches_identity(str(document.metadata.get("source", "")), identity)
            ]
            if matched_docs:
                groups.append(matched_docs[:40])
                lanes.append(f"instrument {identity['year']}-{identity['number']}")
        import chat_tracing

        chat_tracing.event("retrieval-lanes", output=[
            {"lane": name, "hits": chat_tracing.brief(group, limit=8)} for name, group in zip(lanes, groups)
        ], metadata={"identity_refs": [str(r) for r in identity_refs], "pool": len(dedupe(*groups)),
                     "other_queries": list(other_queries)})
        return promote_agreed_hit(self.rank(query, dedupe(*groups), other_queries), agreed)

    def rank(self, query, documents, other_queries=()):
        reranker_documents = build_identity_reranker_documents(
            documents,
            parse_query_identity(query),
        )
        # An Arabic question and a French page are compared through the French wording.
        scores = script_scores(
            lambda wording, chunks: score_documents(self.reranker, wording, chunks),
            query, other_queries, reranker_documents,
        )
        return _identity_diversified_rank(query, documents, scores)


def _read_chunks(path: Path) -> list[Document]:
    """Load JSONL chunks. An empty file is a valid empty corpus (bootstrap / FR-only OCR)."""
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            documents.append(
                Document(
                    page_content=value["page_content"],
                    metadata=value["metadata"],
                )
            )
    return documents


def document_binding(documents: list[Document]) -> str:
    """Bind ordered text and citation metadata, not just embedding inputs."""
    digest = hashlib.sha256()
    for document in documents:
        record = {"page_content": document.page_content, "metadata": document.metadata}
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def create_local_backend():
    from embedding import create_embedding_model
    from vector_store import load_vector_store
    from reranker import create_reranker
    from bm25 import load_documents_from_chroma, create_bm25

    embedding_model = create_embedding_model()
    vector_store = load_vector_store(embedding_model)
    reranker = create_reranker()
    documents = load_documents_from_chroma(vector_store)
    bm25 = create_bm25(documents)

    ocr_vector_store = None
    ocr_documents = []
    ocr_bm25 = None
    if os.environ.get("BCT_OCR_CHROMA_DB"):
        ocr_vector_store = load_vector_store(
            embedding_model,
            persist_directory=os.environ["BCT_OCR_CHROMA_DB"],
            collection_name=os.environ.get("BCT_OCR_CHROMA_COLLECTION", "bct_arabic_ocr_secondary_v1"),
        )
        ocr_documents = load_documents_from_chroma(ocr_vector_store)
        ocr_bm25 = create_bm25(ocr_documents)
    if not documents:
        # Valid before the first ingest, but a wrong --assets root looks identical.
        print(
            f"WARNING: local corpus is empty (collection {os.environ.get('BCT_CHROMA_COLLECTION')!r} "
            f"in {os.environ.get('BCT_CHROMA_DB')!r}); every question will find no evidence.",
            flush=True,
        )
    backend = LocalRetrievalBackend(
        vector_store,
        reranker,
        bm25,
        documents,
        ocr_vector_store=ocr_vector_store,
        ocr_bm25=ocr_bm25,
        ocr_documents=ocr_documents,
    )
    from jsonl_supersession import maybe_wrap_backend

    native = os.environ.get("BCT_NATIVE_CHUNKS_PATH")
    return maybe_wrap_backend(backend, Path(native).resolve().parent if native else None)
