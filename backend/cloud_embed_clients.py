"""Cloud embedding and rerank HTTP clients (Voyage + Google)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Named slots first (production). Numbered VOYAGE_API_KEY_<n> are temporary
# overflow for eval quota — drop those env vars when daily limits are fine.
VOYAGE_KEY_NAMES = (
    "VOYAGE_API_KEY_TERTIARY",
    "VOYAGE_API_KEY_QUATERNARY",
    "VOYAGE_API_KEY_SECONDARY",
    "VOYAGE_API_KEY",
)


def _voyage_credential_names() -> tuple[str, ...]:
    names = list(VOYAGE_KEY_NAMES)
    numbered: list[tuple[int, str]] = []
    for name in os.environ:
        if not name.startswith("VOYAGE_API_KEY_"):
            continue
        suffix = name.removeprefix("VOYAGE_API_KEY_")
        if suffix.isdigit():
            numbered.append((int(suffix), name))
    for _, name in sorted(numbered):
        if name not in names:
            names.append(name)
    return tuple(names)

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
    # gemini-embedding-2: multimodal (text+image+PDF). task_type is 001-only; use prompt prefixes.
    "google": CloudEmbedSpec(
        key="google",
        provider="google",
        model="gemini-embedding-2",
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
        seen: set[str] = set()
        credentials: list[str] = []
        for name in _voyage_credential_names():
            secret = (os.environ.get(name) or "").strip()
            if not secret or secret in seen:
                continue
            seen.add(secret)
            credentials.append(secret)
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
        last_statuses: list[str] = []
        # Bulk ingest often burns Voyage free/paid RPM; cool down and retry more than once.
        max_sweeps = int(os.environ.get("BCT_VOYAGE_RETRY_SWEEPS", "4"))
        base_sleep = float(os.environ.get("BCT_VOYAGE_RETRY_SLEEP_SECONDS", "20"))
        for sweep in range(max(1, max_sweeps)):
            with self._lock:
                start = self._credential_cursor % len(credentials)
                self._credential_cursor += 1
            statuses: list[str] = []
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
                        timeout=(10, 60),
                    )
                except requests.RequestException as error:
                    statuses.append(type(error).__name__)
                    continue
                statuses.append(str(response.status_code))
                if 200 <= response.status_code < 300:
                    body = response.json()
                    _record_voyage_usage(endpoint, body)
                    parsed = parse(body)
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                dir=self.cache_dir, suffix=".tmp", delete=False) as handle:
                            temporary = Path(handle.name)
                            json.dump(body, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
                        os.replace(temporary, cache_path)
                    except OSError:
                        logger.warning("Could not persist Voyage response cache.")
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
                    return parsed
                if response.status_code not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(
                        f"Voyage {endpoint} failed with HTTP {response.status_code}"
                    )
            last_statuses = statuses
            retryable = statuses and all(
                status in {"429", "500", "502", "503", "504"} or status.endswith("Error")
                for status in statuses
            )
            if sweep + 1 < max_sweeps and retryable:
                time.sleep(base_sleep * (sweep + 1))
                continue
            break
        raise RuntimeError(
            f"Voyage {endpoint} is unavailable after trying configured keys "
            f"(statuses: {', '.join(last_statuses)})"
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

    def embed_document_chunks(self, texts, **_kwargs):
        """Contextualize pre-chunked document text for ingestion.

        Requests are bounded so one unusually long circular does not exceed the
        contextual-embedding input limit. Each batch remains page/document-local
        context rather than embedding chunks independently.
        Voyage ignores Google multimodal kwargs (images/titles).
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

    @staticmethod
    def _format_query(text: str) -> str:
        return f"task: question answering | query: {text}"

    @staticmethod
    def _format_document(text: str, *, title: str = "none") -> str:
        safe_title = (title or "none").replace("\n", " ").strip() or "none"
        return f"title: {safe_title} | text: {text}"

    def _embed_contents(self, contents, *, cache_payload: dict):
        """Embed text and/or interleaved image parts (gemini-embedding-2)."""
        from google.genai import types

        cache_path = self._cache_path(cache_payload)
        if cache_path.exists():
            return self._unit(json.loads(cache_path.read_text(encoding="utf-8"))["embedding"])

        keys = getattr(self, "_keys", None) or _gemini_api_keys()
        if not keys:
            raise RuntimeError("GEMINI_API_KEY is required for BCT_CLOUD_RETRIEVAL_PROVIDER=google")
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
                    contents=contents,
                    config=types.EmbedContentConfig(
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
            except Exception as error:  # noqa: BLE001 - rotate on quota / dead keys
                last_error = error
                text_error = str(error).casefold()
                if any(
                    token in text_error
                    for token in (
                        "429",
                        "resource_exhausted",
                        "quota",
                        "rate",
                        "403",
                        "permission_denied",
                        "permission denied",
                        "denied access",
                        "api key not valid",
                        "invalid api key",
                        "api_key_invalid",
                        "consumer_invalid",
                    )
                ):
                    continue
                raise
        raise RuntimeError(f"Gemini embed unavailable after trying configured keys: {last_error}")

    def _embed_one(self, text: str, *, task_type: str, image_png: bytes | None = None, title: str = "none"):
        # gemini-embedding-2 forbids task_type; prefixes live in the text part.
        if task_type == "RETRIEVAL_QUERY":
            text_part = self._format_query(text)
            role = "query"
        else:
            text_part = self._format_document(text, title=title)
            role = "document"
        cache_payload = {
            "provider": "google",
            "model": self.spec.model,
            "role": role,
            "dimension": self.dimension,
            "text": text_part,
            "image_sha256": hashlib.sha256(image_png).hexdigest() if image_png else None,
        }
        if image_png:
            from google.genai import types

            contents = [
                text_part,
                types.Part.from_bytes(data=image_png, mime_type="image/png"),
            ]
        else:
            contents = text_part
        return self._embed_contents(contents, cache_payload=cache_payload)

    def embed_query(self, query):
        return self._embed_one(str(query), task_type="RETRIEVAL_QUERY")

    def embed_document_chunks(self, texts, *, images=None, titles=None):
        texts = [str(text) for text in texts if str(text).strip()]
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        image_list = list(images) if images is not None else [None] * len(texts)
        title_list = list(titles) if titles is not None else ["none"] * len(texts)
        if len(image_list) != len(texts):
            raise ValueError("images length must match texts")
        if len(title_list) != len(texts):
            raise ValueError("titles length must match texts")
        return np.asarray(
            [
                self._embed_one(
                    text,
                    task_type="RETRIEVAL_DOCUMENT",
                    image_png=image_list[index],
                    title=str(title_list[index] or "none"),
                )
                for index, text in enumerate(texts)
            ],
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


