"""Retrieval backends used by the selectable interactive runtime profiles."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import requests

from bm25 import create_bm25, retrieve_bm25
from retrieval_selection import (
    diversify_ranked_pages,
    build_identity_reranker_documents,
    expand_ranked_pages,
    page_chunks,
    parse_query_identity,
    parse_source_identity,
    is_arabic_query,
)
from reranker import score_documents
from vector_store import retrieve_relevant_chunks
from langchain_core.documents import Document


VOYAGE_KEY_NAMES = (
    "VOYAGE_API_KEY_TERTIARY",
    "VOYAGE_API_KEY_QUATERNARY",
    "VOYAGE_API_KEY_SECONDARY",
    "VOYAGE_API_KEY",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CloudEmbedSpec:
    """One cloud embed+rerank stack. Indexes are never shared across specs."""

    key: str
    provider: str
    model: str
    dimension: int
    contextual: bool


CLOUD_EMBED_SPECS = {
    "voyage": CloudEmbedSpec(
        key="voyage",
        provider="voyage",
        model="voyage-context-4",
        dimension=1024,
        contextual=True,
    ),
    # ponytail: gemini-embedding-001 (task_type) over embedding-2 prompt prefixes
    "google": CloudEmbedSpec(
        key="google",
        provider="google",
        model="gemini-embedding-001",
        dimension=768,
        contextual=False,
    ),
}


def cloud_embed_spec(value: str | None = None) -> CloudEmbedSpec:
    key = (value or os.environ.get("BCT_CLOUD_RETRIEVAL_PROVIDER") or "voyage").strip().casefold()
    try:
        return CLOUD_EMBED_SPECS[key]
    except KeyError as error:
        choices = ", ".join(sorted(CLOUD_EMBED_SPECS))
        raise ValueError(
            f"Unknown BCT_CLOUD_RETRIEVAL_PROVIDER {key!r}; choose one of: {choices}"
        ) from error


@dataclass
class CloudRetrievalUsage:
    """Live cloud embed/rerank tokens for one chat turn (cache hits are not counted)."""

    embed_tokens: int = 0
    rerank_tokens: int = 0


_cloud_retrieval_usage: ContextVar[CloudRetrievalUsage | None] = ContextVar(
    "cloud_retrieval_usage", default=None
)


@contextmanager
def track_cloud_retrieval_usage():
    bucket = CloudRetrievalUsage()
    token = _cloud_retrieval_usage.set(bucket)
    try:
        yield bucket
    finally:
        _cloud_retrieval_usage.reset(token)


def _record_voyage_usage(endpoint: str, body: dict) -> None:
    bucket = _cloud_retrieval_usage.get()
    if bucket is None or not isinstance(body, dict):
        return
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return
    tokens = int(usage.get("total_tokens") or 0)
    if tokens <= 0:
        return
    if "embed" in endpoint:
        bucket.embed_tokens += tokens
    elif endpoint == "rerank":
        bucket.rerank_tokens += tokens


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


def _identity_diversified_rank(query, documents, scores):
    scores = [float(score) for score in scores]
    if len(scores) != len(documents):
        raise ValueError("Reranker returned an invalid score count")
    return diversify_ranked_pages(
        sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
    )


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
        """Full retrieved page text for the answer layer; ranking is untouched."""
        return expand_ranked_pages(ranked, self._pages)

    def retrieve(self, query):
        dense = retrieve_relevant_chunks(query, self.vector_store)
        sparse = retrieve_bm25(query, self.bm25, self.bm25_documents)
        groups = [dense, sparse]
        if is_arabic_query(query) and self.ocr_vector_store is not None:
            groups.extend(
                (
                    retrieve_relevant_chunks(query, self.ocr_vector_store, k=5),
                    retrieve_bm25(
                        query,
                        self.ocr_bm25,
                        self.ocr_documents,
                        k=5,
                    ),
                )
            )
        return self.rank(query, dedupe(*groups))

    def rank(self, query, documents):
        reranker_documents = build_identity_reranker_documents(
            documents,
            parse_query_identity(query),
        )
        scores = score_documents(self.reranker, query, reranker_documents)
        return _identity_diversified_rank(query, documents, scores)


def _provider_candidates(
    documents,
    vectors,
    query_vector,
    bm25,
    query,
    *,
    dense_k,
    bm25_k,
):
    similarities = vectors @ query_vector
    dense_indices = np.argsort(-similarities)[:dense_k]
    return dedupe(
        [documents[int(index)] for index in dense_indices],
        retrieve_bm25(query, bm25, documents, k=bm25_k),
    )


class VoyageRetrievalBackend:
    """Use persisted Context-4 document vectors and live query/rerank calls."""

    def __init__(
        self,
        *,
        native_documents,
        native_vectors,
        ocr_documents,
        ocr_vectors,
        client,
    ):
        self.native_documents = list(native_documents)
        self.native_vectors = np.asarray(native_vectors, dtype=np.float32)
        self.ocr_documents = list(ocr_documents)
        self.ocr_vectors = np.asarray(ocr_vectors, dtype=np.float32)
        self.client = client
        self.native_bm25 = create_bm25(self.native_documents)
        self.ocr_bm25 = create_bm25(self.ocr_documents)
        self._pages = page_chunks([*self.native_documents, *self.ocr_documents])

    def expand_pages(self, ranked):
        """Full retrieved page text for the answer layer; ranking is untouched."""
        return expand_ranked_pages(ranked, self._pages)

    def retrieve(self, query):
        query_vector = np.asarray(self.client.embed_query(query), dtype=np.float32)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        groups = [
            _provider_candidates(
                self.native_documents,
                self.native_vectors,
                query_vector,
                self.native_bm25,
                query,
                dense_k=20,
                bm25_k=15,
            )
        ]
        if is_arabic_query(query):
            groups.append(
                _provider_candidates(
                    self.ocr_documents,
                    self.ocr_vectors,
                    query_vector,
                    self.ocr_bm25,
                    query,
                    dense_k=5,
                    bm25_k=5,
                )
            )
        identity = parse_query_identity(query)
        if identity and identity.get("kind") and identity.get("number") is not None:
            indices = []
            for index, document in enumerate(self.native_documents):
                source_identity = parse_source_identity(str(document.metadata.get("source", "")))
                if source_identity and all(source_identity[key] == identity[key]
                                           for key in ("kind", "year", "number")):
                    indices.append(index)
            if indices:
                matching = [self.native_documents[index] for index in indices]
                groups.append(_provider_candidates(
                    matching, self.native_vectors[indices], query_vector,
                    create_bm25(matching), query,
                    dense_k=20, bm25_k=15,
                ))
        documents = dedupe(*groups)
        reranker_documents = build_identity_reranker_documents(documents, identity)
        scores = self.client.rerank(
            query,
            [document.page_content for document in reranker_documents],
        )
        return _identity_diversified_rank(query, documents, scores)


def _read_chunks(path: Path) -> list[Document]:
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
    if not documents:
        raise ValueError(f"Chunk file is empty: {path}")
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


def _load_bound_index(
    provider_root: Path,
    representation: str,
    documents: list[Document],
    spec: CloudEmbedSpec | None = None,
) -> np.ndarray:
    spec = spec or CLOUD_EMBED_SPECS["voyage"]
    text_hashes = [
        hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
        for document in documents
    ]
    matches = []
    for manifest_path in (provider_root / "indexes").glob("*.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("provider") == spec.provider
            and manifest.get("model") == spec.model
            and manifest.get("task") == "document"
            and manifest.get("representation") == representation
            and manifest.get("contextual") is spec.contextual
            and manifest.get("texts") == text_hashes
        ):
            matches.append((manifest_path, manifest))
    if len(matches) != 1:
        raise ValueError(
            f"No bound {spec.provider} index for {representation}; "
            f"found {len(matches)} matches (provider indexes are not interchangeable)"
        )
    manifest_path, manifest = matches[0]
    if manifest.get("documents_sha256") != document_binding(documents):
        raise ValueError(
            f"{spec.provider} document metadata binding mismatch: {manifest_path}"
        )
    index_path = manifest_path.with_suffix(".npy")
    index_bytes = index_path.read_bytes()
    if hashlib.sha256(index_bytes).hexdigest().upper() != str(
        manifest.get("array_sha256", "")
    ).upper():
        raise ValueError(f"{spec.provider} index hash mismatch: {index_path}")
    vectors = np.load(io.BytesIO(index_bytes), allow_pickle=False)
    expected_shape = [len(documents), int(manifest["dimension"])]
    if (
        list(vectors.shape) != expected_shape
        or list(vectors.shape) != manifest.get("shape")
        or vectors.dtype != np.float32
        or manifest.get("dtype") != "float32"
        or not np.isfinite(vectors).all()
        or np.any(np.linalg.norm(vectors, axis=1) <= 0)
    ):
        raise ValueError(f"{spec.provider} index shape or dtype mismatch: {index_path}")
    return vectors


def load_voyage_backend(
    *,
    provider_root,
    native_chunks,
    ocr_chunks,
    client,
    spec: CloudEmbedSpec | None = None,
) -> VoyageRetrievalBackend:
    spec = spec or getattr(client, "spec", None) or CLOUD_EMBED_SPECS["voyage"]
    provider_root = Path(provider_root)
    native_documents = _read_chunks(Path(native_chunks))
    ocr_documents = _read_chunks(Path(ocr_chunks))
    return VoyageRetrievalBackend(
        native_documents=native_documents,
        native_vectors=_load_bound_index(
            provider_root, "native", native_documents, spec
        ),
        ocr_documents=ocr_documents,
        ocr_vectors=_load_bound_index(
            provider_root, "arabic_ocr_secondary", ocr_documents, spec
        ),
        client=client,
    )


def _batches(
    texts,
    max_bytes,
    *,
    max_items=None,
    max_item_bytes=None,
    item_too_large_error="Text exceeds the request budget",
):
    groups = []
    batch = []
    size = 0
    for text in texts:
        text_size = len(text.encode("utf-8"))
        if max_item_bytes is not None and text_size > max_item_bytes:
            raise ValueError(item_too_large_error)
        if batch and (
            size + text_size > max_bytes
            or (max_items is not None and len(batch) >= max_items)
        ):
            groups.append(batch)
            batch, size = [], 0
        batch.append(text)
        size += text_size
    if batch:
        groups.append(batch)
    return groups


class VoyageRuntimeClient:
    """Bounded, cache-first Voyage client for interactive queries."""

    spec = CLOUD_EMBED_SPECS["voyage"]

    def __init__(self, cache_dir, request_post=requests.post):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_post = request_post
        self._credential_cursor = 0
        self._lock = Lock()

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    def _credentials(self):
        credentials = [
            os.environ[name]
            for name in VOYAGE_KEY_NAMES
            if os.environ.get(name)
        ]
        if not credentials:
            raise RuntimeError("No Voyage API key is configured")
        return credentials

    def _post(self, endpoint, payload, parse):
        encoded = json.dumps(
            {"endpoint": endpoint, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cache_path = self.cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.json"
        if cache_path.exists():
            return parse(json.loads(cache_path.read_text(encoding="utf-8")))

        credentials = self._credentials()
        with self._lock:
            start = self._credential_cursor % len(credentials)
            self._credential_cursor += 1
        statuses = []
        for offset in range(len(credentials)):
            secret = credentials[(start + offset) % len(credentials)]
            try:
                response = self.request_post(
                    f"https://api.voyageai.com/v1/{endpoint}",
                    headers={
                        "Authorization": f"Bearer {secret}",
                        "Content-Type": "application/json",
                        "Connection": "close",
                    },
                    json=payload,
                    timeout=(10, 30),
                )
            except requests.RequestException as error:
                statuses.append(type(error).__name__)
                continue
            statuses.append(str(response.status_code))
            if 200 <= response.status_code < 300:
                body = response.json()
                _record_voyage_usage(endpoint, body)
                parsed = parse(body)
                # Each request owns its temporary file, including concurrent
                # requests for the same cache key.
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                            dir=self.cache_dir, suffix=".tmp", delete=False) as handle:
                        temporary = Path(handle.name)
                        json.dump(body, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    os.replace(temporary, cache_path)
                except OSError:
                    # The cache is optional. Windows may deny replacement while
                    # another request holds the same destination; the validated
                    # network response is still usable.
                    logger.warning("Could not persist Voyage response cache.")
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                return parsed
            if response.status_code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"Voyage {endpoint} failed with HTTP {response.status_code}"
                )
        raise RuntimeError(
            f"Voyage {endpoint} is unavailable after trying configured keys "
            f"(statuses: {', '.join(statuses)})"
        )

    def embed_query(self, query):
        return self._post(
            "contextualizedembeddings",
            {
                "inputs": [query],
                "model": "voyage-context-4",
                "input_type": "query",
                "output_dimension": self.dimension,
                "output_dtype": "float",
                "enable_auto_chunking": False,
            },
            self._parse_query_vector,
        )

    def embed_document_chunks(self, texts):
        """Contextualize pre-chunked document text for ingestion.

        Requests are bounded so one unusually long circular does not exceed the
        contextual-embedding input limit. Each batch remains page/document-local
        context rather than embedding chunks independently.
        """
        texts = [str(text) for text in texts if str(text).strip()]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        groups = _batches(
            texts,
            60_000,
            max_items=24,
            max_item_bytes=48_000,
            item_too_large_error="A document chunk exceeds the Voyage ingestion budget",
        )

        vectors = []
        for group in groups:
            expected = len(group)
            batch = self._post(
                "contextualizedembeddings",
                {
                    "inputs": [group],
                    "model": "voyage-context-4",
                    "input_type": "document",
                    "output_dimension": self.dimension,
                    "output_dtype": "float",
                    "enable_auto_chunking": False,
                },
                lambda body, expected=expected: self._parse_document_vectors(body, expected),
            )
            vectors.extend(batch)
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def _flatten_embeddings(body):
        return [
            item["embedding"]
            for group in body.get("data", [])
            for item in group.get("data", [group])
        ]

    def _unit(self, vector, *, label="embedding"):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if (
            vector.shape != (self.dimension,)
            or not np.isfinite(vector).all()
            or not np.isfinite(norm)
            or norm <= 0
        ):
            raise ValueError(f"Voyage returned a nonfinite, zero, or malformed {label}")
        return vector / norm

    def _parse_document_vectors(self, body, expected):
        vectors = self._flatten_embeddings(body)
        if len(vectors) != expected:
            raise ValueError("Voyage returned an invalid document embedding count")
        return [self._unit(value, label="document embedding") for value in vectors]

    def _parse_query_vector(self, body):
        vectors = self._flatten_embeddings(body)
        if len(vectors) != 1:
            raise ValueError("Voyage returned an invalid query embedding")
        return self._unit(vectors[0], label="query embedding")

    def rerank(self, query, texts):
        if not texts:
            return []
        batches = _batches(
            texts,
            12_000,
            max_item_bytes=12_000,
            item_too_large_error="A reranker candidate exceeds the Voyage request budget",
        )

        all_scores = []
        for batch in batches:
            scores = self._post(
                "rerank",
                {
                    "model": "rerank-2.5",
                    "query": query,
                    "documents": batch,
                    "top_k": len(batch),
                    "truncation": False,
                },
                lambda body: self._parse_scores(body, len(batch)),
            )
            all_scores.extend(scores)
        return all_scores

    @staticmethod
    def _parse_scores(body, count):
        scores = [None] * count
        for item in body.get("data", []):
            index = item.get("index")
            if type(index) is not int or not 0 <= index < count or scores[index] is not None:
                raise ValueError("Voyage returned an invalid or duplicate reranker index")
            value = item["relevance_score"]
            if isinstance(value, bool):
                raise ValueError("Voyage returned an invalid reranker score")
            score = float(value)
            if not np.isfinite(score):
                raise ValueError("Voyage returned a nonfinite reranker score")
            scores[index] = score
        if any(score is None for score in scores):
            raise ValueError("Voyage returned a partial reranker response")
        return scores


def _gemini_api_keys() -> list[str]:
    keys = []
    for name in ("GEMINI_API_KEY", *(f"GEMINI_API_KEY_{n}" for n in range(2, 16))):
        value = (os.environ.get(name) or "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def _record_google_embed_usage(response) -> None:
    bucket = _cloud_retrieval_usage.get()
    if bucket is None:
        return
    usage = getattr(response, "usage_metadata", None) or getattr(response, "metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata") or response.get("usage")
    if usage is None:
        return
    if isinstance(usage, dict):
        tokens = int(usage.get("total_token_count") or usage.get("total_tokens") or 0)
    else:
        tokens = int(getattr(usage, "total_token_count", 0) or getattr(usage, "total_tokens", 0) or 0)
    if tokens > 0:
        bucket.embed_tokens += tokens


class GoogleRuntimeClient:
    """Gemini embed + Vertex Ranking. Indexes must not mix with Voyage."""

    spec = CLOUD_EMBED_SPECS["google"]
    _RANK_MODEL = "semantic-ranker-default@latest"

    def __init__(self, cache_dir, *, request_post=requests.post):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_post = request_post
        self._lock = Lock()
        self._genai_client = None
        self._key_cursor = 0

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    def _genai(self):
        if self._genai_client is not None:
            return self._genai_client
        try:
            from google import genai
        except ImportError as error:
            raise RuntimeError(
                "Google cloud retrieval requires google-genai (see requirements-ingestion.txt)"
            ) from error
        keys = _gemini_api_keys()
        if not keys:
            raise RuntimeError("GEMINI_API_KEY is required for BCT_CLOUD_RETRIEVAL_PROVIDER=google")
        self._genai_client = genai.Client(api_key=keys[0])
        self._keys = keys
        return self._genai_client

    def _cache_path(self, payload: dict) -> Path:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.json"

    def _unit(self, vector, *, label="embedding"):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if (
            vector.shape != (self.dimension,)
            or not np.isfinite(vector).all()
            or not np.isfinite(norm)
            or norm <= 0
        ):
            raise ValueError(f"Gemini returned a nonfinite, zero, or malformed {label}")
        return vector / norm

    def _embed_one(self, text: str, *, task_type: str):
        from google.genai import types

        payload = {
            "provider": "google",
            "model": self.spec.model,
            "task_type": task_type,
            "dimension": self.dimension,
            "text": text,
        }
        cache_path = self._cache_path(payload)
        if cache_path.exists():
            return self._unit(json.loads(cache_path.read_text(encoding="utf-8"))["embedding"])

        client = self._genai()
        keys = getattr(self, "_keys", _gemini_api_keys())
        last_error: Exception | None = None
        with self._lock:
            start = self._key_cursor % len(keys)
            self._key_cursor += 1
        for offset in range(len(keys)):
            key = keys[(start + offset) % len(keys)]
            try:
                from google import genai

                client = genai.Client(api_key=key)
                self._genai_client = client
                response = client.models.embed_content(
                    model=self.spec.model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimension,
                    ),
                )
                _record_google_embed_usage(response)
                embedding = response.embeddings[0].values
                vector = self._unit(embedding)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=self.cache_dir,
                        suffix=".tmp",
                        delete=False,
                    ) as handle:
                        temporary = Path(handle.name)
                        json.dump({"embedding": vector.tolist()}, handle, allow_nan=False)
                    os.replace(temporary, cache_path)
                except OSError:
                    logger.warning("Could not persist Gemini embedding cache.")
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                return vector
            except Exception as error:  # noqa: BLE001 - rotate on quota / transient
                last_error = error
                text_error = str(error).casefold()
                if any(token in text_error for token in ("429", "resource_exhausted", "quota", "rate")):
                    continue
                raise
        raise RuntimeError(f"Gemini embed unavailable after trying configured keys: {last_error}")

    def embed_query(self, query):
        return self._embed_one(str(query), task_type="RETRIEVAL_QUERY")

    def embed_document_chunks(self, texts):
        texts = [str(text) for text in texts if str(text).strip()]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.asarray(
            [self._embed_one(text, task_type="RETRIEVAL_DOCUMENT") for text in texts],
            dtype=np.float32,
        )

    def _gcp_project(self) -> str:
        project = (
            os.environ.get("BCT_GCP_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or ""
        ).strip()
        if not project:
            raise RuntimeError(
                "BCT_GCP_PROJECT (or GOOGLE_CLOUD_PROJECT) is required for Vertex Ranking"
            )
        return project

    def _rank_access_token(self) -> str:
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as error:
            raise RuntimeError(
                "Vertex Ranking requires google-auth (installed with google-genai)"
            ) from error
        credentials, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        if not credentials.token:
            raise RuntimeError("Could not refresh Google Cloud credentials for Vertex Ranking")
        return credentials.token

    def rerank(self, query, texts):
        if not texts:
            return []
        # Vertex Ranking: max ~200 records/request; keep batches small.
        batches = _batches(
            [str(text) for text in texts],
            180_000,
            max_items=100,
            max_item_bytes=24_000,
            item_too_large_error="A reranker candidate exceeds the Vertex Ranking budget",
        )
        project = self._gcp_project()
        url = (
            f"https://discoveryengine.googleapis.com/v1/projects/{project}"
            f"/locations/global/rankingConfigs/default_ranking_config:rank"
        )
        all_scores = []
        for batch in batches:
            payload = {
                "model": self._RANK_MODEL,
                "query": query,
                "topN": len(batch),
                "records": [
                    {"id": str(index), "content": text}
                    for index, text in enumerate(batch)
                ],
            }
            cache_payload = {"endpoint": "vertex-rank", "payload": payload}
            cache_path = self._cache_path(cache_payload)
            if cache_path.exists():
                body = json.loads(cache_path.read_text(encoding="utf-8"))
                all_scores.extend(self._parse_rank_scores(body, len(batch)))
                continue
            token = self._rank_access_token()
            response = self.request_post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Connection": "close",
                },
                json=payload,
                timeout=(10, 60),
            )
            if not (200 <= response.status_code < 300):
                raise RuntimeError(
                    f"Vertex Ranking failed with HTTP {response.status_code}: {response.text[:300]}"
                )
            body = response.json()
            scores = self._parse_rank_scores(body, len(batch))
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.cache_dir,
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    json.dump(body, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
                os.replace(temporary, cache_path)
            except OSError:
                logger.warning("Could not persist Vertex Ranking cache.")
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            all_scores.extend(scores)
        return all_scores

    @staticmethod
    def _parse_rank_scores(body, count):
        scores = [None] * count
        for item in body.get("records", []):
            raw_id = item.get("id")
            try:
                index = int(raw_id)
            except (TypeError, ValueError) as error:
                raise ValueError("Vertex Ranking returned an invalid record id") from error
            if not 0 <= index < count or scores[index] is not None:
                raise ValueError("Vertex Ranking returned an invalid or duplicate record id")
            value = item.get("score")
            if isinstance(value, bool) or value is None:
                raise ValueError("Vertex Ranking returned an invalid score")
            score = float(value)
            if not np.isfinite(score):
                raise ValueError("Vertex Ranking returned a nonfinite score")
            scores[index] = score
        if any(score is None for score in scores):
            raise ValueError("Vertex Ranking returned a partial response")
        return scores


def create_cloud_runtime_client(cache_root: str | Path, spec: CloudEmbedSpec | None = None):
    spec = spec or cloud_embed_spec()
    cache_root = Path(cache_root)
    if spec.key == "voyage":
        return VoyageRuntimeClient(
            os.environ.get("BCT_VOYAGE_RUNTIME_CACHE", str(cache_root / "voyage-cache"))
        )
    return GoogleRuntimeClient(
        os.environ.get("BCT_GOOGLE_RUNTIME_CACHE", str(cache_root / "google-cache"))
    )


def create_voyage_backend_from_environment():
    names = {
        "BCT_VOYAGE_PROVIDER_ROOT": os.environ.get("BCT_VOYAGE_PROVIDER_ROOT"),
        "BCT_NATIVE_CHUNKS_PATH": os.environ.get("BCT_NATIVE_CHUNKS_PATH"),
        "BCT_OCR_CHUNKS_PATH": os.environ.get("BCT_OCR_CHUNKS_PATH"),
    }
    missing = [name for name, value in names.items() if not value]
    if missing:
        raise RuntimeError(
            "Cloud profile requires: " + ", ".join(missing)
        )
    provider_root = Path(names["BCT_VOYAGE_PROVIDER_ROOT"])
    spec = cloud_embed_spec()
    client = create_cloud_runtime_client(provider_root, spec)
    return load_voyage_backend(
        provider_root=provider_root,
        native_chunks=names["BCT_NATIVE_CHUNKS_PATH"],
        ocr_chunks=names["BCT_OCR_CHUNKS_PATH"],
        client=client,
        spec=spec,
    )
