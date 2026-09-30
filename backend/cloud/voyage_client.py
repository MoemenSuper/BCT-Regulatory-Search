"""Cloud embedding and rerank HTTP client (Voyage)."""
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


@dataclass(frozen=True)
class CloudEmbedSpec:
    """The cloud embed+rerank stack an index is bound to (provider + model + dimension)."""

    key: str
    provider: str
    model: str
    dimension: int
    contextual: bool


VOYAGE_SPEC = CloudEmbedSpec(
    key="voyage",
    provider="voyage",
    model="voyage-context-4",
    dimension=1024,
    contextual=True,
)


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

    spec = VOYAGE_SPEC

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


def create_cloud_runtime_client(cache_root: str | Path) -> VoyageRuntimeClient:
    return VoyageRuntimeClient(
        os.environ.get("BCT_VOYAGE_RUNTIME_CACHE", str(Path(cache_root) / "voyage-cache"))
    )
