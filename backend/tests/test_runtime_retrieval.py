from langchain_core.documents import Document
import hashlib
import json
import numpy as np
import pytest

from rag import runtime_retrieval
from cloud import voyage_retrieval
from cloud.voyage_client import VoyageRuntimeClient
from cloud.voyage_retrieval import VoyageRetrievalBackend, load_voyage_backend
from rag.runtime_retrieval import LocalRetrievalBackend


def _doc(text, source, page):
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_local_backend_uses_dense_and_bm25_candidates_before_reranking(monkeypatch):
    dense = _doc("dense evidence", "Cir_A_fr.pdf", 0)
    sparse = _doc("sparse evidence", "Cir_B_fr.pdf", 1)
    calls = {}

    monkeypatch.setattr(
        runtime_retrieval,
        "retrieve_relevant_chunks",
        lambda query, store, k=20: (
            calls.update(dense=(query, store)) or [dense]
        ),
    )
    monkeypatch.setattr(
        runtime_retrieval,
        "retrieve_bm25",
        lambda query, bm25, documents, k=15: (
            calls.update(sparse=(query, bm25, documents)) or [sparse]
        ),
    )
    monkeypatch.setattr(
        runtime_retrieval,
        "score_documents",
        lambda reranker, query, documents: [
            0.9 if document is sparse else 0.2 for document in documents
        ],
    )

    backend = LocalRetrievalBackend(
        vector_store="store",
        reranker="reranker",
        bm25="bm25",
        bm25_documents=[dense, sparse],
    )
    ranked = backend.retrieve("deadline")

    assert [document for document, _score in ranked] == [sparse, dense]
    assert calls["dense"] == ("deadline", "store")
    assert calls["sparse"] == ("deadline", "bm25", [dense, sparse])


def test_local_backend_can_rerank_graph_documents_with_the_same_model(monkeypatch):
    first = _doc("first", "Cir_A_fr.pdf", 0)
    second = _doc("second", "Cir_B_fr.pdf", 1)
    monkeypatch.setattr(
        runtime_retrieval,
        "score_documents",
        lambda _reranker, _query, documents: list(range(len(documents))),
    )

    backend = LocalRetrievalBackend(object(), object(), object(), [])

    assert [item[0] for item in backend.rank("query", [first, second])] == [second, first]


def test_local_backend_keeps_only_the_best_chunk_from_each_source_page(monkeypatch):
    best = _doc("best", "Cir_A_fr.pdf", 0)
    duplicate = _doc("duplicate", "Cir_A_fr.pdf", 0)
    other = _doc("other", "Cir_B_fr.pdf", 0)
    monkeypatch.setattr(
        runtime_retrieval,
        "score_documents",
        lambda _reranker, _query, documents: [0.9, 0.8, 0.7][: len(documents)],
    )
    backend = LocalRetrievalBackend(object(), object(), object(), [])

    ranked = backend.rank("Circular 2020-03", [best, duplicate, other])

    assert [item[0] for item in ranked] == [best, other]


def test_local_backend_uses_identity_for_scoring_but_returns_original_documents(
    monkeypatch,
):
    original = _doc("answer text", "Cir_2020_03_fr.pdf", 0)
    scored_texts = []

    def fake_score(_reranker, _query, documents):
        scored_texts.extend(document.page_content for document in documents)
        return [0.5] * len(documents)

    monkeypatch.setattr(runtime_retrieval, "score_documents", fake_score)
    backend = LocalRetrievalBackend(object(), object(), object(), [])

    ranked = backend.rank("Cir_2020_03_fr.pdf", [original])

    assert scored_texts[0].startswith("[document kind=cir; year=2020; number=3;")
    assert ranked == [(original, 0.5)]


def test_local_backend_adds_bounded_ocr_candidates_only_for_arabic(monkeypatch):
    native = _doc("native", "Cir_A_ar.pdf", 0)
    ocr = _doc("ocr", "Cir_B_ar.pdf", 1)
    dense_calls = []
    sparse_calls = []
    monkeypatch.setattr(
        runtime_retrieval,
        "retrieve_relevant_chunks",
        lambda _query, store, k=20: (
            dense_calls.append((store, k)) or ([ocr] if store == "ocr-store" else [native])
        ),
    )
    monkeypatch.setattr(
        runtime_retrieval,
        "retrieve_bm25",
        lambda _query, bm25, _documents, k=15: (
            sparse_calls.append((bm25, k)) or []
        ),
    )
    monkeypatch.setattr(
        runtime_retrieval,
        "score_documents",
        lambda _reranker, _query, documents: [1.0] * len(documents),
    )
    backend = LocalRetrievalBackend(
        "native-store",
        object(),
        "native-bm25",
        [native],
        ocr_vector_store="ocr-store",
        ocr_bm25="ocr-bm25",
        ocr_documents=[ocr],
    )

    arabic = backend.retrieve("ما هو الأجل؟")
    french = backend.retrieve("Quel est le délai ?")

    assert [item[0] for item in arabic] == [native, ocr]
    assert [item[0] for item in french] == [native]
    assert ("ocr-store", 10) in dense_calls  # 10 asked, 5 kept after the per-page cap
    assert ("ocr-bm25", 10) in sparse_calls  # 10 asked, 5 kept after the per-page cap


def test_voyage_backend_fuses_arabic_ocr_and_returns_one_chunk_per_page():
    native_a = _doc("native A", "Cir_A_ar.pdf", 0)
    native_a_duplicate = _doc("native A continuation", "Cir_A_ar.pdf", 0)
    native_b = _doc("native B", "Cir_B_ar.pdf", 1)
    ocr = _doc("OCR target", "Cir_C_ar.pdf", 2)

    class FakeClient:
        def __init__(self):
            self.queries = []
            self.reranks = []

        def embed_query(self, query):
            self.queries.append(query)
            return np.asarray([1.0, 0.0], dtype=np.float32)

        def rerank(self, query, texts):
            self.reranks.append((query, texts))
            return [0.95 if "OCR target" in text else 0.8 - index * 0.1 for index, text in enumerate(texts)]

    client = FakeClient()
    backend = VoyageRetrievalBackend(
        native_documents=[native_a, native_a_duplicate, native_b],
        native_vectors=np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32),
        ocr_documents=[ocr],
        ocr_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        client=client,
    )

    ranked = backend.retrieve("ما هو الأجل؟")

    assert client.queries == ["ما هو الأجل؟"]
    assert ranked[0][0] is ocr
    assert len({(item[0].metadata["source"], item[0].metadata["page"]) for item in ranked}) == len(ranked)


def test_voyage_backend_does_not_add_arabic_ocr_for_a_french_query():
    native = _doc("preuve", "Cir_A_fr.pdf", 0)
    ocr = _doc("OCR", "Cir_B_ar.pdf", 1)

    class FakeClient:
        def embed_query(self, _query):
            return np.asarray([1.0, 0.0], dtype=np.float32)

        def rerank(self, _query, texts):
            return [1.0] * len(texts)

    backend = VoyageRetrievalBackend(
        native_documents=[native],
        native_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        ocr_documents=[ocr],
        ocr_vectors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        client=FakeClient(),
    )

    ranked = backend.retrieve("Quel est le délai ?")

    assert [item[0] for item in ranked] == [native]


def _sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_bound_index(root, representation, documents, vectors):
    indexes = root / "indexes"
    indexes.mkdir(parents=True, exist_ok=True)
    index = indexes / f"runtime_{representation}_document_context_TEST.npy"
    np.save(index, vectors)
    manifest = {
        "model": "voyage-context-4",
        "provider": "voyage",
        "task": "document",
        "representation": representation,
        "contextual": True,
        "dimension": int(vectors.shape[1]),
        "texts": [
            hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
            for document in documents
        ],
        "array_sha256": _sha_file(index),
        "documents_sha256": runtime_retrieval.document_binding(documents),
        "shape": list(vectors.shape),
        "dtype": "float32",
    }
    index.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_chunks(path, documents):
    path.write_text(
        "".join(
            json.dumps(
                {"page_content": document.page_content, "metadata": document.metadata}
            )
            + "\n"
            for document in documents
        ),
        encoding="utf-8",
    )


def test_voyage_backend_loads_only_hash_bound_persisted_document_indexes(tmp_path):
    native = [_doc("native", "Cir_A_fr.pdf", 0)]
    ocr = [_doc("ocr", "Cir_A_ar.pdf", 0)]
    native_chunks = tmp_path / "native.jsonl"
    ocr_chunks = tmp_path / "ocr.jsonl"
    _write_chunks(native_chunks, native)
    _write_chunks(ocr_chunks, ocr)
    _write_bound_index(
        tmp_path,
        "native",
        native,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    _write_bound_index(
        tmp_path,
        "arabic_ocr_secondary",
        ocr,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    backend = load_voyage_backend(
        provider_root=tmp_path,
        native_chunks=native_chunks,
        ocr_chunks=ocr_chunks,
        client=object(),
    )

    assert [document.page_content for document in backend.native_documents] == ["native"]
    assert [document.page_content for document in backend.ocr_documents] == ["ocr"]


def test_voyage_backend_rejects_an_index_bound_to_different_chunk_text(tmp_path):
    native = [_doc("native", "Cir_A_fr.pdf", 0)]
    ocr = [_doc("ocr", "Cir_A_ar.pdf", 0)]
    native_chunks = tmp_path / "native.jsonl"
    ocr_chunks = tmp_path / "ocr.jsonl"
    _write_chunks(native_chunks, native)
    _write_chunks(ocr_chunks, ocr)
    _write_bound_index(
        tmp_path,
        "native",
        [_doc("different", "Cir_A_fr.pdf", 0)],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    _write_bound_index(
        tmp_path,
        "arabic_ocr_secondary",
        ocr,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="No bound voyage index"):
        load_voyage_backend(
            provider_root=tmp_path,
            native_chunks=native_chunks,
            ocr_chunks=ocr_chunks,
            client=object(),
        )


@pytest.mark.parametrize("field,value", [("source", "Other.pdf"), ("page", 100),
    ("temporal_resolution", "VERIFIED")])
def test_index_rejects_changed_citation_or_temporal_metadata(tmp_path, field, value):
    documents = [_doc("unchanged text", "Cir_2020_03_fr.pdf", 0)]
    _write_bound_index(tmp_path, "native", documents, np.asarray([[1, 0]], dtype=np.float32))
    documents[0].metadata[field] = value
    with pytest.raises(ValueError, match="metadata binding mismatch"):
        voyage_retrieval._load_bound_index(tmp_path, "native", documents)


def test_index_rejects_legacy_manifest_without_metadata_binding(tmp_path):
    documents = [_doc("text", "Cir_2020_03_fr.pdf", 0)]
    _write_bound_index(tmp_path, "native", documents, np.asarray([[1, 0]], dtype=np.float32))
    path = next((tmp_path / "indexes").glob("*.json"))
    value = json.loads(path.read_text())
    del value["documents_sha256"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="metadata binding mismatch"):
        voyage_retrieval._load_bound_index(tmp_path, "native", documents)


def test_voyage_runtime_client_rotates_once_and_reuses_its_exact_cache(
    monkeypatch, tmp_path
):
    calls = []

    class Response:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body

        def json(self):
            return self._body

    def post(_url, *, headers, **_kwargs):
        calls.append(headers["Authorization"])
        if len(calls) == 1:
            return Response(429, {})
        return Response(
            200,
            {
                "data": [
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 1, "relevance_score": 0.3},
                ]
            },
        )

    monkeypatch.setenv("VOYAGE_API_KEY_TERTIARY", "first")
    monkeypatch.setenv("VOYAGE_API_KEY_QUATERNARY", "second")
    monkeypatch.delenv("VOYAGE_API_KEY_SECONDARY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    for index in range(2, 6):
        monkeypatch.delenv(f"VOYAGE_API_KEY_{index}", raising=False)
    client = VoyageRuntimeClient(tmp_path, request_post=post)

    assert client.rerank("query", ["one", "two"]) == [0.8, 0.3]
    assert client.rerank("query", ["one", "two"]) == [0.8, 0.3]
    assert calls == ["Bearer first", "Bearer second"]


def test_voyage_runtime_client_includes_numbered_overflow_key(monkeypatch, tmp_path):
    calls = []

    class Response:
        def __init__(self, status_code, body):
            self.status_code = status_code
            self._body = body

        def json(self):
            return self._body

    def post(_url, *, headers, **_kwargs):
        calls.append(headers["Authorization"])
        if len(calls) == 1:
            return Response(429, {})
        return Response(
            200,
            {
                "data": [
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 1, "relevance_score": 0.3},
                ]
            },
        )

    for name in (
        "VOYAGE_API_KEY_TERTIARY",
        "VOYAGE_API_KEY_QUATERNARY",
        "VOYAGE_API_KEY_SECONDARY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VOYAGE_API_KEY", "primary")
    monkeypatch.setenv("VOYAGE_API_KEY_2", "overflow")
    client = VoyageRuntimeClient(tmp_path, request_post=post)

    assert client.rerank("query", ["one", "two"]) == [0.8, 0.3]
    assert calls == ["Bearer primary", "Bearer overflow"]


def test_voyage_runtime_client_cools_down_and_retries_after_all_keys_rate_limit(
    monkeypatch, tmp_path
):
    calls = []
    sleeps = []

    class Response:
        def __init__(self, status_code, body=None):
            self.status_code = status_code
            self._body = body or {}

        def json(self):
            return self._body

    def post(_url, *, headers, **_kwargs):
        calls.append(headers["Authorization"])
        if len(calls) <= 2:
            return Response(429)
        return Response(
            200,
            {
                "data": [
                    {"index": 0, "relevance_score": 0.7},
                    {"index": 1, "relevance_score": 0.2},
                ]
            },
        )

    monkeypatch.setenv("VOYAGE_API_KEY_TERTIARY", "first")
    monkeypatch.setenv("VOYAGE_API_KEY_QUATERNARY", "second")
    monkeypatch.delenv("VOYAGE_API_KEY_SECONDARY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    for index in range(2, 6):
        monkeypatch.delenv(f"VOYAGE_API_KEY_{index}", raising=False)
    monkeypatch.setenv("BCT_VOYAGE_RETRY_SLEEP_SECONDS", "0")
    monkeypatch.setattr(
        "cloud.voyage_client.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )
    client = VoyageRuntimeClient(tmp_path, request_post=post)

    assert client.rerank("query", ["one", "two"]) == [0.7, 0.2]
    assert sleeps == [0.0]
    # First sweep burns both keys; second sweep resumes from the next cursor.
    assert calls == ["Bearer first", "Bearer second", "Bearer second"]


def test_selon_circulaire_2025_13_named_instrument_outranks_newer_mention():
    """Explicit instrument names are hard retrieval anchors before semantic/latest preference."""
    from pathlib import Path

    from rag.retrieval_selection import (
        explicit_instrument_identity,
        prefer_named_instrument_hits,
    )
    from rag.runtime_retrieval import _identity_diversified_rank

    query = (
        "Selon la circulaire 2025-13, quelles sont les règles d'exportation ?"
    )
    identity = explicit_instrument_identity(query)
    assert identity is not None
    assert (identity["kind"], identity["year"], identity["number"]) == ("cir", 2025, 13)

    named = _doc(
        "Dispositions relatives à l'exportation de produits.",
        "Cir_2025_13_fr.pdf",
        2,
    )
    mention = _doc(
        "Vu la circulaire n° 2025-13 du 27 octobre 2025, citée dans le présent texte.",
        "Cir_2026_04_fr.pdf",
        1,
    )
    # Newer document that only mentions 2025-13 scores higher semantically.
    preferred = prefer_named_instrument_hits(
        [(mention, 0.99), (named, 0.40)],
        identity,
    )
    assert Path(str(preferred[0][0].metadata["source"])).name == "Cir_2025_13_fr.pdf"

    reranked = _identity_diversified_rank(query, [mention, named], [0.99, 0.40])
    assert Path(str(reranked[0][0].metadata["source"])).name == "Cir_2025_13_fr.pdf"


def test_avant_named_instrument_demotes_cutoff_after_named_promotion():
    """'Avant 2025-13' must not keep the named later circular on top."""
    from pathlib import Path

    from rag.runtime_retrieval import _identity_diversified_rank

    query = (
        "Avant 2025-13, a partir de combien de jours fallait-il des conditions "
        "pour une vente libre entre 61 et 360 jours ?"
    )
    older = _doc(
        "Article 10 nouveau: délais de règlement allant jusqu'à 60 jours. "
        "Les ventes de 61 à 360 jours sont effectuées librement sous conditions.",
        "Cir_2020_02_fr.pdf",
        2,
    )
    cutoff = _doc(
        "Article 10 nouveau: délais de règlement allant jusqu'à 120 jours. "
        "Les ventes de 121 à 360 jours sont effectuées librement sous conditions.",
        "Cir_2025_13_fr.pdf",
        2,
    )
    # Named promotion would otherwise put 2025-13 first; historical must win last.
    reranked = _identity_diversified_rank(query, [cutoff, older], [0.99, 0.40])
    assert Path(str(reranked[0][0].metadata["source"])).name == "Cir_2020_02_fr.pdf"


def test_grandfathering_avant_keeps_named_circular():
    """Commitments before 2026-04 ask for that circular — do not demote it."""
    from pathlib import Path

    from rag.retrieval_selection import is_historical_cutoff_query
    from rag.runtime_retrieval import _identity_diversified_rank

    query = (
        "Pour un engagement de financement existant avant la circulaire 2026-04, "
        "quelle disposition s'applique si l'exécution était déjà entamée ?"
    )
    assert is_historical_cutoff_query(query) is False
    older = _doc(
        "Liste des produits non prioritaires. Dépôt de 100%.",
        "Cir_2017_09_fr.pdf",
        1,
    )
    named = _doc(
        "Sont exclues les importations dont l'exécution de l'engagement de financement "
        "a été effectivement entamée avant l'entrée en vigueur de la présente circulaire.",
        "Cir_2026_04_fr.pdf",
        3,
    )
    reranked = _identity_diversified_rank(query, [older, named], [0.95, 0.50])
    assert Path(str(reranked[0][0].metadata["source"])).name == "Cir_2026_04_fr.pdf"


def test_selon_bare_year_number_is_explicit_instrument_identity():
    from rag.retrieval_selection import explicit_instrument_identity

    identity = explicit_instrument_identity(
        "Selon 2025-13, une vente a 150 jours avec police d'assurance-credit "
        "a l'export est-elle libre ?"
    )
    assert identity is not None
    assert (identity["kind"], identity["year"], identity["number"]) == ("cir", 2025, 13)

    identity_2026 = explicit_instrument_identity(
        "Selon 2026-04, une industrielle sans fiche technique "
        "beneficie-t-elle de l'exclusion ?"
    )
    assert identity_2026 is not None
    assert (identity_2026["kind"], identity_2026["year"], identity_2026["number"]) == (
        "cir",
        2026,
        4,
    )


def test_arabic_instrument_identity_forms_are_generic():
    """Arabic cite forms must parse any instrument — not a single circular's answer key."""
    from rag.retrieval_selection import explicit_instrument_identity

    classic = explicit_instrument_identity("ما هو موضوع المنشور عدد 6 لسنة 2026؟")
    assert classic is not None
    assert (classic["kind"], classic["year"], classic["number"]) == ("cir", 2026, 6)

    year_number = explicit_instrument_identity("وفقا للمنشور 2020-03، ما هو موضوعه؟")
    assert year_number is not None
    assert (year_number["kind"], year_number["year"], year_number["number"]) == (
        "cir",
        2020,
        3,
    )

    bare = explicit_instrument_identity("حسب 2016-01، ما هو التوقيت؟")
    assert bare is not None
    assert (bare["kind"], bare["year"], bare["number"]) == ("cir", 2016, 1)

    note = explicit_instrument_identity("حسب المذكرة 2024-163، ما السقف؟")
    assert note is not None
    assert (note["kind"], note["year"], note["number"]) == ("note", 2024, 163)


def test_empty_collection_has_no_bm25_index_and_retrieves_nothing():
    """A French-only corpus has an empty Arabic OCR collection; chat must not crash on it."""
    from rag.bm25 import create_bm25, retrieve_bm25

    assert create_bm25([]) is None
    assert retrieve_bm25("taux directeur", None, []) == []


def test_keyword_search_matches_the_way_people_type():
    from rag.bm25 import create_bm25, retrieve_bm25, tokenize

    # accents, plural, elision, digits glued to letters; Arabic harakat, hamza and taa marbuta
    assert tokenize("Délais de l'agrément (Cir2024-03)") == tokenize("delai de agrements cir 2024 03")
    assert tokenize("إعادةُ التمويل") == tokenize("اعاده التمويل")
    docs = [Document(page_content=t, metadata={}) for t in (
        "Les délais de paiement des chèques.", "Le taux directeur reste inchangé.", "Autre texte.")]
    assert retrieve_bm25("delai paiement cheque", create_bm25(docs), docs, k=1)[0] is docs[0]


def test_historical_cutoff_needs_avant_before_an_instrument_reference():
    from rag.retrieval_selection import is_historical_cutoff_query

    assert is_historical_cutoff_query("Avant la circulaire 2025-13, quel était le délai ?")
    assert not is_historical_cutoff_query("Faut-il un accord avant de payer le fournisseur selon 2025-13 ?")
    assert not is_historical_cutoff_query("Engagements pris avant le 26 mars 2026 : que dit 2026-04 ?")


def test_batch_sizes_shrink_with_weaker_hardware(monkeypatch):
    from rag import hardware

    monkeypatch.setattr(hardware, "torch_device", lambda: "cuda")
    monkeypatch.setattr(hardware, "gpu_memory_gb", lambda: 6.0)
    assert hardware.batch_size(64) == 32
    monkeypatch.setattr(hardware, "torch_device", lambda: "cpu")
    assert hardware.batch_size(64) == 8


def test_other_wordings_add_candidates_and_each_chunk_keeps_its_best_score(monkeypatch):
    french_page = _doc("allocation de séjour pour études : 4.000 D par mois", "Cir_2025_10_fr.pdf", 4)
    arabic_note = _doc("مذكرة حول الدراسة", "Note_2019_05_ar.pdf", 1)
    by_query = {"ما هو المبلغ الأقصى لمنحة الدراسة؟": arabic_note, "montant maximum de l'allocation études": french_page}
    monkeypatch.setattr(runtime_retrieval, "retrieve_relevant_chunks", lambda query, store, k=20: [by_query[query]])
    monkeypatch.setattr(runtime_retrieval, "retrieve_bm25", lambda query, bm25, documents, k=15: [])
    # The reranker judges the French page badly against the Arabic wording, well against the French one.
    scores = {("ما هو المبلغ الأقصى لمنحة الدراسة؟", "Cir_2025_10_fr.pdf"): 0.01,
              ("montant maximum de l'allocation études", "Cir_2025_10_fr.pdf"): 0.95}
    monkeypatch.setattr(runtime_retrieval, "score_documents", lambda reranker, query, documents: [
        scores.get((query, document.metadata["source"]), 0.3) for document in documents])

    backend = LocalRetrievalBackend(vector_store="store", reranker="reranker", bm25="bm25",
                                    bm25_documents=[french_page, arabic_note])
    ranked = backend.retrieve("ما هو المبلغ الأقصى لمنحة الدراسة؟",
                              other_queries=["montant maximum de l'allocation études"])

    assert [(document.metadata["source"], score) for document, score in ranked] == [
        ("Cir_2025_10_fr.pdf", 0.95), ("Note_2019_05_ar.pdf", 0.3)]


def test_a_chunk_both_searches_agree_on_keeps_a_top_five_place():
    from rag.runtime_retrieval import promote_agreed_hit

    ranked = [(_doc(f"page {n}", f"Cir_{n}_fr.pdf", 1), 1.0 - n / 10) for n in range(8)]
    rule = ranked[7][0]  # the reranker put it last
    promoted = promote_agreed_hit(ranked, [rule])
    assert [document for document, _score in promoted[:5]][-1] is rule
    assert [document for document, _score in promoted[:4]] == [document for document, _score in ranked[:4]]
    # Already in the top five: nothing moves.
    assert promote_agreed_hit(ranked, [ranked[1][0]]) == ranked
