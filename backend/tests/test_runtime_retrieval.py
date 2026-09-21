from langchain_core.documents import Document
import hashlib
import json
import numpy as np
import pytest

import runtime_retrieval
from runtime_retrieval import (
    LocalRetrievalBackend,
    VoyageRetrievalBackend,
    VoyageRuntimeClient,
    load_voyage_backend,
)


def _doc(text, source, page):
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_local_backend_uses_dense_and_bm25_candidates_before_reranking(monkeypatch):
    dense = _doc("dense evidence", "Cir_A_fr.pdf", 0)
    sparse = _doc("sparse evidence", "Cir_B_fr.pdf", 1)
    calls = {}

    monkeypatch.setattr(
        runtime_retrieval,
        "retrieve_relevant_chunks",
        lambda query, store: (
            calls.update(dense=(query, store)) or [dense]
        ),
    )
    monkeypatch.setattr(
        runtime_retrieval,
        "retrieve_bm25",
        lambda query, bm25, documents: (
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
    assert ("ocr-store", 5) in dense_calls
    assert ("ocr-bm25", 5) in sparse_calls


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
        runtime_retrieval._load_bound_index(tmp_path, "native", documents)


def test_index_rejects_legacy_manifest_without_metadata_binding(tmp_path):
    documents = [_doc("text", "Cir_2020_03_fr.pdf", 0)]
    _write_bound_index(tmp_path, "native", documents, np.asarray([[1, 0]], dtype=np.float32))
    path = next((tmp_path / "indexes").glob("*.json"))
    value = json.loads(path.read_text())
    del value["documents_sha256"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="metadata binding mismatch"):
        runtime_retrieval._load_bound_index(tmp_path, "native", documents)


def test_google_provider_cannot_load_voyage_indexes(tmp_path):
    documents = [_doc("shared text", "Cir_2020_03_fr.pdf", 0)]
    _write_bound_index(
        tmp_path,
        "native",
        documents,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    google = runtime_retrieval.CLOUD_EMBED_SPECS["google"]
    with pytest.raises(ValueError, match="not interchangeable"):
        runtime_retrieval._load_bound_index(tmp_path, "native", documents, google)


def test_cloud_embed_spec_defaults_to_voyage(monkeypatch):
    monkeypatch.delenv("BCT_CLOUD_RETRIEVAL_PROVIDER", raising=False)
    assert runtime_retrieval.cloud_embed_spec().key == "voyage"
    monkeypatch.setenv("BCT_CLOUD_RETRIEVAL_PROVIDER", "google")
    assert runtime_retrieval.cloud_embed_spec().provider == "google"
    monkeypatch.setenv("BCT_CLOUD_RETRIEVAL_PROVIDER", "nope")
    with pytest.raises(ValueError, match="Unknown BCT_CLOUD_RETRIEVAL_PROVIDER"):
        runtime_retrieval.cloud_embed_spec()


def test_google_runtime_client_rerank_uses_vertex_payload(monkeypatch, tmp_path):
    calls = []

    class Response:
        status_code = 200

        def json(self):
            return {
                "records": [
                    {"id": "1", "score": 0.2},
                    {"id": "0", "score": 0.9},
                ]
            }

        @property
        def text(self):
            return ""

    def post(url, *, headers, json, **_kwargs):
        calls.append((url, headers["Authorization"], json))
        return Response()

    monkeypatch.setenv("BCT_GCP_PROJECT", "demo-project")
    client = runtime_retrieval.GoogleRuntimeClient(tmp_path, request_post=post)
    monkeypatch.setattr(client, "_rank_access_token", lambda: "token-xyz")

    assert client.rerank("q", ["a", "b"]) == [0.9, 0.2]
    assert "demo-project" in calls[0][0]
    assert calls[0][1] == "Bearer token-xyz"
    assert calls[0][2]["records"][0]["content"] == "a"
    # cache hit
    assert client.rerank("q", ["a", "b"]) == [0.9, 0.2]
    assert len(calls) == 1


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
    client = VoyageRuntimeClient(tmp_path, request_post=post)

    assert client.rerank("query", ["one", "two"]) == [0.8, 0.3]
    assert client.rerank("query", ["one", "two"]) == [0.8, 0.3]
    assert calls == ["Bearer first", "Bearer second"]


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
    monkeypatch.setenv("BCT_VOYAGE_RETRY_SLEEP_SECONDS", "0")
    monkeypatch.setattr(
        "runtime_retrieval.time.sleep",
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

    from retrieval_selection import (
        explicit_instrument_identity,
        prefer_named_instrument_hits,
    )
    from runtime_retrieval import _identity_diversified_rank

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

    from runtime_retrieval import _identity_diversified_rank

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

    from retrieval_selection import is_historical_cutoff_query
    from runtime_retrieval import _identity_diversified_rank

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
    from retrieval_selection import explicit_instrument_identity

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
    from retrieval_selection import explicit_instrument_identity, query_regulatory_regime

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

    # Domain lexicons are Arabic vocabulary, not instrument pins.
    assert (
        query_regulatory_regime("ما شروط تمويل توريد المنتجات غير ذات الأولوية؟")
        == "nonpriority_import"
    )
    assert (
        query_regulatory_regime("شركة صناعية تستورد منتجا غير ذي أولوية")
        == "nonpriority_import"
    )
    assert (
        query_regulatory_regime("مدة الدفع القصوى لتصدير بضاعة دون شروط خاصة")
        == "export_settlement"
    )
    assert (
        query_regulatory_regime(
            "بيع لمدة 250 يوما بسفتجة مظهرة من بنك غير مقيم. هل تلزم رخصة؟"
        )
        == "export_settlement"
    )


def test_export_settlement_query_prefers_settlement_doc_over_nonpriority_mention():
    """Domain prefer uses page language, not instrument filenames."""
    from pathlib import Path

    from retrieval_selection import prefer_regime_hits, query_regulatory_regime
    from runtime_retrieval import _identity_diversified_rank

    query = (
        "Quel est actuellement le delai de reglement maximum autorise "
        "pour une vente a l'export sans condition speciale ?"
    )
    assert query_regulatory_regime(query) == "export_settlement"

    settlement = _doc(
        "Les ventes a l'exportation des marchandises sont reglees librement "
        "et sans autorisation lorsque le delai de reglement n'excede pas 120 jours.",
        "Any_Settlement_Instrument_fr.pdf",
        1,
    )
    nonpriority = _doc(
        "Objet : Conditions de financement de l'importation de produits non "
        "prioritaires. Les importateurs constituent des depots en numeraire "
        "sur leurs fonds propres a 100%.",
        "Any_Nonpriority_Instrument_fr.pdf",
        1,
    )
    preferred = prefer_regime_hits(
        [(nonpriority, 0.99), (settlement, 0.35)],
        query,
    )
    assert Path(str(preferred[0][0].metadata["source"])).name == (
        "Any_Settlement_Instrument_fr.pdf"
    )

    reranked = _identity_diversified_rank(
        query, [nonpriority, settlement], [0.99, 0.35]
    )
    assert Path(str(reranked[0][0].metadata["source"])).name == (
        "Any_Settlement_Instrument_fr.pdf"
    )


def test_nonpriority_import_query_prefers_nonpriority_language_over_settlement():
    from pathlib import Path

    from retrieval_selection import query_regulatory_regime
    from runtime_retrieval import _identity_diversified_rank

    query = (
        "Quelles sont les regles pour le financement des importations "
        "de produits non prioritaires ?"
    )
    assert query_regulatory_regime(query) == "nonpriority_import"

    settlement = _doc(
        "Les ventes a l'exportation des marchandises sont reglees librement "
        "et sans autorisation lorsque le delai de reglement n'excede pas 120 jours.",
        "Any_Settlement_Instrument_fr.pdf",
        2,
    )
    nonpriority = _doc(
        "Les importateurs de produits non prioritaires doivent constituer "
        "des depots en numeraire sur leurs fonds propres a hauteur de 100%.",
        "Any_Nonpriority_Instrument_fr.pdf",
        2,
    )
    reranked = _identity_diversified_rank(
        query, [settlement, nonpriority], [0.90, 0.40]
    )
    assert Path(str(reranked[0][0].metadata["source"])).name == (
        "Any_Nonpriority_Instrument_fr.pdf"
    )


def test_mixed_regime_query_does_not_force_domain_reorder():
    from retrieval_selection import prefer_regime_hits, query_regulatory_regime

    query = (
        "Un intermediaire doit appliquer les delais d'export et les "
        "restrictions de financement non prioritaire."
    )
    assert query_regulatory_regime(query) is None
    settlement = _doc(
        "120 jours librement et sans autorisation.",
        "Any_Settlement_Instrument_fr.pdf",
        2,
    )
    nonpriority = _doc(
        "produits non prioritaires fonds propres 100%",
        "Any_Nonpriority_Instrument_fr.pdf",
        2,
    )
    ranked = [(nonpriority, 0.9), (settlement, 0.8)]
    assert prefer_regime_hits(ranked, query) == ranked


def test_nonpriority_prefer_newer_strong_hit_over_mid_era_list():
    from pathlib import Path

    from retrieval_selection import prefer_regime_hits

    query = (
        "Quelles operations sont exclues du champ de l'article premier "
        "relatif aux concours pour produits non prioritaires ?"
    )
    older = _doc(
        "La liste relative aux produits non prioritaires est abrogee et remplacee.",
        "Cir_2018_01_fr.pdf",
        2,
    )
    newer = _doc(
        "Sont exclues des dispositions de l'article premier les importations "
        "realisees dans le cadre de marches publics conclus au profit de l'Etat.",
        "Cir_2026_04_fr.pdf",
        2,
    )
    preferred = prefer_regime_hits([(older, 0.95), (newer, 0.40)], query)
    assert Path(str(preferred[0][0].metadata["source"])).name == "Cir_2026_04_fr.pdf"

