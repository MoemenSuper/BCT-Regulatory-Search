"""Cached, disposable provider retrieval matrix; never imports production retrieval."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
from langchain_core.documents import Document

from bm25 import create_bm25, retrieve_bm25
from experiments.artifacts import sha256_file, write_json_atomic as _atomic_write
from experiments.candidate_diversity_ablation import diversify_ranked_pages
from experiments.document_identity_candidate_experiment import build_identity_reranker_documents
from experiments.ocr_fusion_retrieval import _deserialize_candidates, is_arabic_query
from experiments.retrieval_ablations import _candidate_key, _merge_candidates, _page_matches
from reranker import create_reranker, score_documents

EVALUATION_SHA256 = "00964BA335B759D01BA42CED75FC6AE10F082AE4D45499426CEA69F3F1DF3CA1"
CONTROL_SHA256 = "4BCBBD54CF1C4D5E3CBE17C8ED29836482E48266BF83BD428B9898866BFCC722"
NATIVE_CACHE_SHA256 = "A061D88BEE3F4C221EFC8612AC13272C83AA1C8A2EEA475409C2DC6E0CF65B33"
NATIVE_CHUNKS_SHA256 = "1E4300E460A7FF346764633AAE8BBF6A71965B635BBECC6707B2634B79DBECCD"
OCR_CHUNKS_SHA256 = "A6F24B76DDC6E968B3B673F810C4DECB9EE12AFB3FE6E5EBF2018AE0F85E8C15"
DIMENSIONS = 1024
NATIVE_DENSE_K, NATIVE_BM25_K, OCR_DENSE_K, OCR_BM25_K = 20, 15, 5, 5
VOYAGE_SAFE_REQUEST_BYTES = 24000
VOYAGE_MIN_REQUEST_INTERVAL_SECONDS = 20.0
HTTP_TIMEOUT_SECONDS = (10, 30)
GPU_LOCK_OWNER = "owner.json"


def write_json_atomic(path: Path, value: Any) -> None:
    """Retry a Windows rename lock; benchmark checkpoints must be resumable."""
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            _atomic_write(path, value)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise last_error or RuntimeError("checkpoint write failed")


def _process_is_alive(pid: int) -> bool:
    """Return whether a same-host PID still exists without treating it as an error."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _clear_abandoned_gpu_lock(lock: Path) -> bool:
    """Reclaim only a legacy empty lock or a lock whose recorded owner has died."""
    if not lock.exists():
        return True
    owner = lock / GPU_LOCK_OWNER
    if owner.exists():
        try:
            owner_pid = int(json.loads(owner.read_text(encoding="utf-8"))["pid"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if _process_is_alive(owner_pid):
            return False
        owner.unlink()
    try:
        lock.rmdir()
    except OSError:
        return False
    return True


def _acquire_gpu_lock(lock: Path) -> None:
    """Acquire a crash-safe local-reranker lock, recovering an abandoned owner."""
    while True:
        try:
            lock.mkdir()
            (lock / GPU_LOCK_OWNER).write_text(_canonical({"pid": os.getpid()}), encoding="utf-8")
            return
        except FileExistsError:
            if not _clear_abandoned_gpu_lock(lock):
                time.sleep(30)


def _release_gpu_lock(lock: Path) -> None:
    (lock / GPU_LOCK_OWNER).unlink(missing_ok=True)
    lock.rmdir()


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_env(name: str) -> str:
    """Read only the requested secret from production .env; never print or copy it."""
    value = os.environ.get(name)
    if value:
        return value
    path = Path(r"C:\Users\Moemen Super\BCT-Regulatory-Search\.env")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        key, candidate = line.split("=", 1)
        if key.strip() == name:
            return candidate.strip().strip('"').strip("'")
    raise RuntimeError(f"Required credential {name} is absent")


def _read_chunks(path: Path) -> list[Document]:
    return [Document(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _embedding_batches(documents: list[Document], provider: str) -> list[list[Document]]:
    """Keep Voyage requests under its account-level 10K TPM cap conservatively."""
    if provider != "voyage":
        return [documents[start:start + 64] for start in range(0, len(documents), 64)]
    batches: list[list[Document]] = []
    current: list[Document] = []
    current_bytes = 0
    for document in documents:
        size = len(document.page_content.encode("utf-8"))
        if current and (len(current) >= 16 or current_bytes + size > VOYAGE_SAFE_REQUEST_BYTES):
            batches.append(current)
            current, current_bytes = [], 0
        if size > VOYAGE_SAFE_REQUEST_BYTES:
            raise ValueError("A single frozen chunk exceeds the Voyage safe request budget")
        current.append(document)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def _rerank_batches(documents: list[str], provider: str) -> list[list[str]]:
    """Keep remote Voyage reranks below the fragile full-pool request size."""
    if provider != "voyage":
        return [documents]
    maximum = VOYAGE_SAFE_REQUEST_BYTES // 2
    batches: list[list[str]] = []
    batch: list[str] = []
    size = 0
    for document in documents:
        document_size = len(document.encode("utf-8"))
        if batch and size + document_size > maximum:
            batches.append(batch)
            batch, size = [], 0
        if document_size > maximum:
            raise ValueError("A single rerank candidate exceeds the Voyage safe request budget")
        batch.append(document)
        size += document_size
    if batch:
        batches.append(batch)
    return batches


def _source(value: str) -> str:
    return Path(value.replace("\\", "/")).name.casefold()


def _identity_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    document = candidate["document"]
    return {"source": Path(str(document.metadata.get("source", ""))).name, "page": int(document.metadata.get("page", -1)), "text_sha256": _sha(document.page_content)}


def _rank(ranked: list[tuple[dict[str, Any], float]], case: dict[str, Any]) -> tuple[int | None, int | None]:
    source_rank = page_rank = None
    for position, (candidate, _score) in enumerate(ranked, 1):
        metadata = candidate["document"].metadata
        if _source(str(metadata.get("source", ""))) != _source(str(case["expected_source"])):
            continue
        source_rank = source_rank or position
        if page_rank is None and _page_matches(metadata, int(case["expected_page"])):
            page_rank = position
    return source_rank, page_rank


def _metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    relevant = [row for row in records if row["relevant"]]
    ranks = [row["result"]["page_rank"] for row in relevant]
    source_ranks = [row["result"]["source_rank"] for row in relevant]
    hit = lambda values, k: sum(value is not None and value <= k for value in values) / len(values)
    return {"n": len(relevant), "page_at_1": hit(ranks, 1), "page_at_5": hit(ranks, 5), "page_at_20": hit(ranks, 20), "source_at_1": hit(source_ranks, 1), "source_at_5": hit(source_ranks, 5), "page_mrr": sum(1 / value if value else 0 for value in ranks) / len(ranks)}


def _exact_mcnemar(records: list[dict[str, Any]], control: dict[str, dict[str, Any]]) -> dict[str, Any]:
    repaired = regressed = 0
    for row in records:
        if not row["relevant"]:
            continue
        before = control[row["id"]]["candidate_page_rank"]
        after = row["result"]["page_rank"]
        repaired += bool((before is None or before > 5) and after is not None and after <= 5)
        regressed += bool(before is not None and before <= 5 and (after is None or after > 5))
    n = repaired + regressed
    p = 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(repaired, regressed) + 1)) / 2**n)
    return {"test": "two_sided_exact_McNemar", "repairs": repaired, "regressions": regressed, "net": repaired - regressed, "p_value": p}


def _holm(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    ordered = sorted(((name, item["p_value"]) for name, item in rows.items()), key=lambda item: item[1])
    count, prior, output = len(ordered), 0.0, {}
    for index, (name, p_value) in enumerate(ordered):
        adjusted = min(1.0, max(prior, (count - index) * p_value))
        output[name], prior = adjusted, adjusted
    return output


class Cache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.counts: Counter[str] = Counter()

    def get(self, kind: str, key: str) -> dict[str, Any] | None:
        path = self.root / kind / f"{key}.json"
        if not path.exists():
            self.counts[f"{kind}_misses"] += 1
            return None
        self.counts[f"{kind}_hits"] += 1
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, kind: str, key: str, value: dict[str, Any]) -> None:
        path = self.root / kind / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, value)


class ProviderClient:
    """Small REST client with response binding and bounded transient retry."""
    def __init__(self, cache: Cache):
        self.cache, self.events, self._last_request_started, self._credential_cursor, self._credential_cooldown = cache, [], {}, Counter(), {}

    def _next_secret(self, provider: str) -> tuple[str, str]:
        # Start with the credential that passed the full frozen-pool rerank probe;
        # the scheduler still rotates fairly and cools down stalled credentials.
        names = ["VOYAGE_API_KEY_TERTIARY", "VOYAGE_API_KEY_QUATERNARY", "VOYAGE_API_KEY_SECONDARY", "VOYAGE_API_KEY"] if provider == "voyage" else ["JINA_API_KEY"]
        available: list[tuple[str, str]] = []
        for name in names:
            try:
                available.append((name, _load_env(name)))
            except RuntimeError:
                continue
        if not available:
            raise RuntimeError(f"No credential configured for {provider}")
        if provider != "voyage":
            return available[0]
        now = time.monotonic()
        ready = [entry for entry in available if self._credential_cooldown.get((provider, entry[0]), 0.0) <= now]
        if not ready:
            next_ready = min(self._credential_cooldown[(provider, name)] for name, _secret in available)
            time.sleep(next_ready - now)
            now = time.monotonic()
            ready = [entry for entry in available if self._credential_cooldown.get((provider, entry[0]), 0.0) <= now]
        for offset in range(len(available)):
            slot = (self._credential_cursor[provider] + offset) % len(available)
            entry = available[slot]
            if entry in ready:
                self._credential_cursor[provider] = slot + 1
                return entry
        raise RuntimeError("Voyage credential cooldown scheduler found no ready key")

    def _post(self, *, provider: str, url: str, secret: str | None = None, payload: dict[str, Any], cache_kind: str, binding: dict[str, Any]) -> dict[str, Any]:
        key = _sha(_canonical({"provider": provider, "url": url, "binding": binding, "payload": payload}))
        cached = self.cache.get(cache_kind, key)
        if cached is not None:
            return cached
        managed_credentials = secret is None
        credential_slot, secret = ("provided", secret) if secret is not None else self._next_secret(provider)
        transient_attempt = 0
        while True:
            transient_attempt += 1
            if provider == "voyage":
                rate_key = (provider, credential_slot)
                previous = self._last_request_started.get(rate_key)
                if previous is not None:
                    remaining = VOYAGE_MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - previous)
                    if remaining > 0:
                        time.sleep(remaining)
                self._last_request_started[rate_key] = time.monotonic()
            started = time.perf_counter()
            try:
                response = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {secret}",
                        "Content-Type": "application/json",
                        "Connection": "close",
                    },
                    json=payload,
                    timeout=HTTP_TIMEOUT_SECONDS,
                )
                elapsed = time.perf_counter() - started
                body = response.json() if response.content else {}
            except (requests.RequestException, ValueError) as error:
                response, elapsed, body = None, time.perf_counter() - started, {"error": type(error).__name__, "detail": str(error)}
            status = response.status_code if response is not None else None
            event = {"provider": provider, "credential_slot": credential_slot, "url": url, "status": status, "latency_seconds": elapsed, "attempt": transient_attempt, "request_id": (response.headers.get("x-request-id") if response is not None else None), "binding": binding, "response": body}
            self.events.append(event)
            if status is not None and 200 <= status < 300:
                self.cache.put(cache_kind, key, event)
                return event
            insufficient_balance = status == 403 and isinstance(body, dict) and body.get("code") == "AUTHZ_INSUFFICIENT_BALANCE"
            # Network exceptions have no HTTP status.  A depleted Jina key is
            # likewise resumable: the user can top it up or replace it while
            # the process waits, so neither condition should kill the worker.
            if status not in {None, 429, 500, 502, 503, 504} and not insufficient_balance:
                raise RuntimeError(_canonical({k: event[k] for k in ("provider", "status", "response")}))
            if provider == "voyage" and managed_credentials and status in {None, 429, 500, 502, 503, 504}:
                self._credential_cooldown[(provider, credential_slot)] = time.monotonic() + 60.0
                credential_slot, secret = self._next_secret(provider)
                continue
            # The benchmark is deliberately persistent: provider quotas are not a
            # partial-result signal. Honour Retry-After if supplied, otherwise
            # leave a full minute between requests under a 3 RPM account limit.
            retry_after = response.headers.get("Retry-After") if response is not None else None
            try:
                delay = max(60.0, float(retry_after)) if retry_after else 60.0
            except ValueError:
                delay = 60.0
            if insufficient_balance:
                delay = 60.0
            elif status != 429 and transient_attempt % 3 != 0:
                delay = min(60.0, 2 ** (transient_attempt % 3) + random.uniform(0, 0.25))
            time.sleep(delay)

    def embeddings(self, provider: str, model: str, texts: list[str] | list[list[str]], task: str, *, contextual: bool = False, context_identity: Any = None) -> list[list[float]]:
        if provider == "voyage":
            url = "https://api.voyageai.com/v1/contextualizedembeddings" if contextual else "https://api.voyageai.com/v1/embeddings"
            payload: dict[str, Any] = ({"inputs": (texts if task == "document" else texts), "model": model, "input_type": task, "output_dimension": DIMENSIONS, "output_dtype": "float", "enable_auto_chunking": False} if contextual else {"input": texts, "model": model, "input_type": task, "output_dimension": DIMENSIONS, "output_dtype": "float", "truncation": False})
        else:
            url = "https://api.jina.ai/v1/embeddings"
            payload = {"model": model, "input": texts, "task": f"retrieval.{ 'passage' if task == 'document' else 'query'}", "dimensions": DIMENSIONS, "embedding_type": "float", "truncate": False}
        flat_texts = [text for group in texts for text in group] if contextual and task == "document" else texts
        binding = {"model": model, "task": task, "dimensions": DIMENSIONS, "dtype": "float", "truncation": False, "text_sha256": [_sha(text) for text in flat_texts], "context": context_identity}
        event = self._post(provider=provider, url=url, payload=payload, cache_kind="embeddings", binding=binding)
        body = event["response"]
        if contextual:
            groups = body.get("data", [])
            vectors = [item["embedding"] for group in groups for item in group.get("data", [group])]
        else:
            vectors = [item["embedding"] for item in sorted(body.get("data", []), key=lambda item: item.get("index", -1))]
        if len(vectors) != len(flat_texts) or any(len(vector) != DIMENSIONS for vector in vectors):
            raise ValueError(f"Invalid {provider} embedding response binding")
        return vectors

    def rerank(self, provider: str, model: str, query: str, documents: list[str]) -> list[float]:
        url = "https://api.voyageai.com/v1/rerank" if provider == "voyage" else "https://api.jina.ai/v1/rerank"
        all_scores: list[float] = []
        for batch in _rerank_batches(documents, provider):
            payload = ({"model": model, "query": query, "documents": batch, "top_k": len(batch), "truncation": False} if provider == "voyage" else {"model": model, "query": query, "documents": batch, "top_n": len(batch), "return_documents": False, "truncate": False})
            binding = {"model": model, "query_sha256": _sha(query), "candidate_text_sha256": [_sha(text) for text in batch], "top_k": len(batch), "truncation": False}
            event = self._post(provider=provider, url=url, payload=payload, cache_kind="rerank", binding=binding)
            results = event["response"].get("data", event["response"].get("results", []))
            scores: list[float | None] = [None] * len(batch)
            for item in results:
                index = item.get("index")
                if not isinstance(index, int) or not 0 <= index < len(scores) or scores[index] is not None:
                    raise ValueError(f"Invalid {provider} reranker response index")
                scores[index] = float(item.get("relevance_score", item.get("score")))
            if any(score is None for score in scores):
                raise ValueError(f"Partial {provider} reranker response")
            all_scores.extend(float(score) for score in scores)
        return all_scores


def _dense_candidates(documents: list[Document], vectors: np.ndarray, query_vector: np.ndarray, *, representation: str, dense_k: int, bm25_k: int, query: str, bm25: Any) -> list[dict[str, Any]]:
    scores = vectors @ query_vector
    dense_indexes = np.argsort(-scores, kind="stable")[:dense_k]
    sparse = retrieve_bm25(query, bm25, documents, k=bm25_k)
    by_key: dict[str, dict[str, Any]] = {}
    for channel, items in (("dense", [documents[index] for index in dense_indexes]), ("bm25", sparse)):
        for rank, document in enumerate(items, 1):
            key = _candidate_key(document)
            target = by_key.setdefault(key, {"document": document, "representations": set(), "ranks": {}})
            target["representations"].add(representation)
            target["ranks"].setdefault(representation, {})[channel] = rank
    return list(by_key.values())


def _prefix_texts(candidates: list[dict[str, Any]], query: str) -> list[str]:
    return [document.page_content for document in build_identity_reranker_documents(candidates, __import__("experiments.document_identity_candidate_experiment", fromlist=["parse_query_identity"]).parse_query_identity(query))]


def _serialize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"document": {"page_content": item["document"].page_content, "metadata": item["document"].metadata}, "representations": sorted(item["representations"]), "ranks": item["ranks"]} for item in candidates]


def _deserialize_pool(value: Any) -> list[dict[str, Any]]:
    """Accept this runner's list cache and the older envelope cache explicitly."""
    return _deserialize_candidates(value if isinstance(value, dict) and "candidates" in value else {"candidates": value})


class Matrix:
    def __init__(self, root: Path, evaluation: Path, native_chunks: Path, ocr_chunks: Path, native_cache: Path, control: Path):
        for path, expected in ((evaluation, EVALUATION_SHA256), (native_chunks, NATIVE_CHUNKS_SHA256), (ocr_chunks, OCR_CHUNKS_SHA256), (native_cache, NATIVE_CACHE_SHA256), (control, CONTROL_SHA256)):
            if sha256_file(path) != expected:
                raise ValueError(f"Frozen input hash mismatch: {path}")
        self.root, self.cases = root, json.loads(evaluation.read_text(encoding="utf-8"))
        self.native_chunks, self.ocr_chunks = native_chunks, ocr_chunks
        self.native: list[Document] | None = None
        self.ocr: list[Document] | None = None
        self.native_bm25: Any | None = None
        self.ocr_bm25: Any | None = None
        self.native_cache = json.loads(native_cache.read_text(encoding="utf-8"))
        self.control = {row["id"]: row for row in json.loads(control.read_text(encoding="utf-8"))["records"]}
        self.cache = Cache(root / "cache")
        self.client = ProviderClient(self.cache)

    @classmethod
    def for_runtime_cases(
        cls,
        *,
        root: Path,
        cases: list[dict[str, Any]],
        native_chunks: Path,
        ocr_chunks: Path,
    ) -> "Matrix":
        """Create a provider matrix without evaluation-gold validation or scoring."""
        matrix = cls.__new__(cls)
        matrix.root, matrix.cases = root, cases
        matrix.native_chunks, matrix.ocr_chunks = native_chunks, ocr_chunks
        matrix.native, matrix.ocr = _read_chunks(native_chunks), _read_chunks(ocr_chunks)
        matrix.native_bm25, matrix.ocr_bm25 = create_bm25(matrix.native), create_bm25(matrix.ocr)
        matrix.native_cache = matrix.control = None
        matrix.cache = Cache(root / "cache")
        matrix.client = ProviderClient(matrix.cache)
        return matrix

    def _ensure_corpus(self) -> None:
        """Load corpus data only when a missing candidate pool must be rebuilt."""
        if self.native is not None:
            return
        self.native, self.ocr = _read_chunks(self.native_chunks), _read_chunks(self.ocr_chunks)
        self.native_bm25, self.ocr_bm25 = create_bm25(self.native), create_bm25(self.ocr)

    def vectors(self, name: str, provider: str, model: str, documents: list[Document], task: str, *, contextual: bool = False) -> np.ndarray:
        index = self.root / "indexes" / f"{name}_{task}_{'context' if contextual else 'plain'}.npy"
        manifest = index.with_suffix(".json")
        expected_manifest = {
            "model": model,
            "provider": provider,
            "task": task,
            "contextual": contextual,
            "dimension": DIMENSIONS,
            "texts": [_sha(document.page_content) for document in documents],
        }
        if index.exists() and manifest.exists():
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if info == expected_manifest:
                return np.load(index)
        vectors: list[list[float]] = []
        if contextual and task == "document":
            by_source: dict[str, list[Document]] = {}
            for document in documents:
                by_source.setdefault(str(document.metadata.get("source")), []).append(document)
            windows: list[tuple[str, int, list[Document]]] = []
            for source, group in by_source.items():
                group.sort(key=lambda item: (int(item.metadata.get("page", 0)), int(item.metadata.get("flat_part", 0))))
                window, chars, window_id = [], 0, 0
                for item in group:
                    if window and chars + len(item.page_content.encode("utf-8")) > VOYAGE_SAFE_REQUEST_BYTES:
                        windows.append((source, window_id, window))
                        window, chars, window_id = [], 0, window_id + 1
                    if len(item.page_content.encode("utf-8")) > VOYAGE_SAFE_REQUEST_BYTES:
                        raise ValueError(f"Context-4 pre-chunked page over deterministic window limit: {source}:{item.metadata.get('page')}")
                    window.append(item); chars += len(item.page_content.encode("utf-8"))
                if window:
                    windows.append((source, window_id, window))
            if [entry.page_content for _source_name, _window_id, window in windows for entry in window] != [document.page_content for document in documents]:
                raise ValueError("Context-4 grouping changed frozen chunk order")
            batch: list[tuple[str, int, list[Document]]] = []
            batch_bytes = 0
            for window in windows:
                window_bytes = sum(len(entry.page_content.encode("utf-8")) for entry in window[2])
                if batch and (len(batch) >= 1 or batch_bytes + window_bytes > VOYAGE_SAFE_REQUEST_BYTES):
                    vectors.extend(self.client.embeddings(provider, model, [[entry.page_content for entry in item[2]] for item in batch], task, contextual=True, context_identity=[[{"source": item[0], "window": item[1], "page": entry.metadata.get("page"), "chunk": entry.metadata.get("flat_part")} for entry in item[2]] for item in batch]))
                    batch, batch_bytes = [], 0
                batch.append(window); batch_bytes += window_bytes
            if batch:
                vectors.extend(self.client.embeddings(provider, model, [[entry.page_content for entry in item[2]] for item in batch], task, contextual=True, context_identity=[[{"source": item[0], "window": item[1], "page": entry.metadata.get("page"), "chunk": entry.metadata.get("flat_part")} for entry in item[2]] for item in batch]))
        else:
            for batch in _embedding_batches(documents, provider):
                vectors.extend(self.client.embeddings(provider, model, [document.page_content for document in batch], task, contextual=contextual))
        array = np.asarray(vectors, dtype=np.float32)
        array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
        index.parent.mkdir(parents=True, exist_ok=True); np.save(index, array)
        write_json_atomic(manifest, expected_manifest)
        return array

    def query_vectors(self, name: str, provider: str, model: str, *, contextual: bool = False) -> np.ndarray:
        documents = [Document(page_content=case["query"]) for case in self.cases]
        return self.vectors(name, provider, model, documents, "query", contextual=contextual)

    def candidates(self, name: str, provider: str, model: str, *, contextual: bool = False) -> dict[str, list[dict[str, Any]]]:
        path = self.root / "candidate_pools" / f"{name}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {case_id: _deserialize_pool(value) for case_id, value in raw.items()}
        self._ensure_corpus()
        assert self.native is not None and self.ocr is not None
        assert self.native_bm25 is not None and self.ocr_bm25 is not None
        native_vectors = self.vectors(name, provider, model, self.native, "document", contextual=contextual)
        ocr_vectors = self.vectors(name, provider, model, self.ocr, "document", contextual=contextual)
        queries = self.query_vectors(name, provider, model, contextual=contextual)
        output = {}
        for case, vector in zip(self.cases, queries):
            groups = [_dense_candidates(self.native, native_vectors, vector, representation="native", dense_k=NATIVE_DENSE_K, bm25_k=NATIVE_BM25_K, query=case["query"], bm25=self.native_bm25)]
            if is_arabic_query(case["query"]):
                groups.append(_dense_candidates(self.ocr, ocr_vectors, vector, representation="arabic_ocr_secondary", dense_k=OCR_DENSE_K, bm25_k=OCR_BM25_K, query=case["query"], bm25=self.ocr_bm25))
            output[case["id"]] = _merge_candidates(groups)
        path.parent.mkdir(parents=True, exist_ok=True); write_json_atomic(path, {key: _serialize_candidates(value) for key, value in output.items()})
        return output

    def reconstructed_control_candidates(self) -> dict[str, list[dict[str, Any]]]:
        """Control native pool is frozen; OCR additions are reconstructed from immutable OCR chunks."""
        path = self.root / "candidate_pools" / "control_reconstructed.json"
        if path.exists():
            return {case_id: _deserialize_pool(value) for case_id, value in json.loads(path.read_text(encoding="utf-8")).items()}
        # The e5 OCR query channel is the only missing material in the legacy full artifact.
        from experiments.ocr_fusion_retrieval import _load_search_representation, _retrieve
        manifest = json.loads(Path(r"C:\Users\Moemen Super\AppData\Local\Temp\bct-arabic-ocr-fusion-retrieval-20260824-v1\representation\manifest.json").read_text(encoding="utf-8"))
        rep = _load_search_representation(manifest)
        output = {}
        for case in self.cases:
            base = _deserialize_candidates(self.native_cache[case["id"]])
            output[case["id"]] = _merge_candidates((base, _retrieve(rep, case["query"], OCR_DENSE_K, OCR_BM25_K) if is_arabic_query(case["query"]) else []))
        path.parent.mkdir(parents=True, exist_ok=True); write_json_atomic(path, {key: _serialize_candidates(value) for key, value in output.items()})
        return output

    def run_reranker(self, name: str, candidates: dict[str, list[dict[str, Any]]], provider: str, model: str) -> dict[str, Any]:
        path = self.root / "results" / f"{name}.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if previous.get("status") == "complete":
            return previous
        records = previous.get("records", [])
        completed_ids = {record["id"] for record in records}
        if len(completed_ids) != len(records):
            raise ValueError(f"Duplicate checkpoint case IDs in {path}")
        local_scores: dict[str, list[float]] = {}
        if provider == "local":
            # Both provider lanes may arrive here; only one may occupy the GPU.
            lock = self.root / "bge_gpu.lock"
            _acquire_gpu_lock(lock)
            try:
                inputs = [(case, _prefix_texts(candidates[case["id"]], case["query"])) for case in self.cases if case["id"] not in completed_ids]
                pairs = [(case["query"], text) for case, texts in inputs for text in texts]
                flat_scores = create_reranker().predict(pairs, batch_size=64, show_progress_bar=False)
                position = 0
                for case, texts in inputs:
                    local_scores[case["id"]] = [float(score) for score in flat_scores[position:position + len(texts)]]
                    position += len(texts)
                if position != len(flat_scores):
                    raise ValueError("BGE batch score count does not match frozen candidate identities")
            finally:
                _release_gpu_lock(lock)
        for case in self.cases:
            if case["id"] in completed_ids:
                continue
            pool = candidates[case["id"]]
            texts = _prefix_texts(pool, case["query"])
            started = time.perf_counter()
            if provider == "local":
                scores = local_scores[case["id"]]
            else:
                scores = self.client.rerank(provider, model, case["query"], texts)
            ranked = diversify_ranked_pages(sorted(zip(pool, scores), key=lambda item: item[1], reverse=True))
            source_rank = page_rank = None
            if case["relevant"]:
                source_rank, page_rank = _rank(ranked, case)
            control = self.control[case["id"]]
            records.append({"id": case["id"], "language": case["language"], "relevant": case["relevant"], "category": case["category"], "result": {"source_rank": source_rank, "page_rank": page_rank, "top5": [{**_identity_signature(candidate), "reranker_score": float(score), "representations": sorted(candidate["representations"]), "ranks": candidate["ranks"]} for candidate, score in ranked[:5]]}, "candidate_pool": _serialize_candidates(pool), "candidate_page_recall": (any(_source(str(candidate["document"].metadata.get("source", ""))) == _source(case["expected_source"]) and _page_matches(candidate["document"].metadata, int(case["expected_page"])) for candidate in pool) if case["relevant"] else None), "latency_seconds": time.perf_counter() - started, "control_page_rank": control["candidate_page_rank"]})
            if len(records) % 10 == 0:
                write_json_atomic(path, {"status": "running", "records": records})
        metrics = {"overall": _metrics(records), "fr": _metrics([row for row in records if row["language"] == "fr"]), "ar": _metrics([row for row in records if row["language"] == "ar"])}
        paired = _exact_mcnemar(records, self.control)
        result = {"status": "complete", "name": name, "embedding": name.split("+")[0], "reranker": model, "metrics": metrics, "paired_page5_vs_control": paired, "candidate_pool_page_recall": sum(bool(row["candidate_page_recall"]) for row in records if row["relevant"]) / 689, "repairs": [row["id"] for row in records if row["relevant"] and (row["control_page_rank"] is None or row["control_page_rank"] > 5) and row["result"]["page_rank"] is not None and row["result"]["page_rank"] <= 5], "regressions": [row["id"] for row in records if row["relevant"] and row["control_page_rank"] is not None and row["control_page_rank"] <= 5 and (row["result"]["page_rank"] is None or row["result"]["page_rank"] > 5)], "mean_latency_seconds": statistics.mean(row["latency_seconds"] for row in records), "records": records}
        write_json_atomic(path, result); return result


def _verify_control(matrix: Matrix, candidates: dict[str, list[dict[str, Any]]]) -> None:
    output = matrix.run_reranker("control_reconstruction", candidates, "local", "BAAI/bge-reranker-v2-m3")
    mismatches = []
    for row in output["records"]:
        expected = matrix.control[row["id"]]["candidate_top5"]
        actual = [{"source": item["source"], "page": item["page"]} for item in row["result"]["top5"]]
        expected_pair = [{"source": item["source"], "page": item["page"]} for item in expected]
        if actual != expected_pair:
            mismatches.append(row["id"])
    if mismatches:
        raise RuntimeError(f"Frozen control candidate reconstruction mismatch for {len(mismatches)} cases: {mismatches[:10]}")


def _has_complete_result(path: Path) -> bool:
    """Avoid loading a large completed result merely to decide whether to resume it."""
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        return '"status": "complete"' in handle.read(1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("probe", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=Path("evaluation_queries.json"))
    parser.add_argument("--native-chunks", type=Path, required=True)
    parser.add_argument("--ocr-chunks", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--lane", choices=("all", "voyage", "jina"), default="all")
    parser.add_argument("--skip-context-bge", action="store_true")
    args = parser.parse_args(); matrix = Matrix(args.root, args.evaluation, args.native_chunks, args.ocr_chunks, args.native_cache, args.control)
    if args.command == "probe":
        # One cached public-text request per provider/model verifies response task/model bindings before full quota.
        matrix.client.embeddings("voyage", "voyage-4-large", ["probe"], "query")
        matrix.client.embeddings("jina", "jina-embeddings-v5-text-small", ["probe"], "query")
        matrix.client.rerank("voyage", "rerank-2.5", "probe", ["probe"])
        matrix.client.rerank("jina", "jina-reranker-v3.5", "probe", ["probe"])
        write_json_atomic(args.root / "probe.json", {"status": "complete", "events": matrix.client.events, "cache": matrix.cache.counts})
        return
    control_pool = matrix.reconstructed_control_candidates()
    control_result = args.root / "results" / "control_reconstruction.json"
    if args.lane == "all" or not _has_complete_result(control_result):
        _verify_control(matrix, control_pool)
    runs: dict[str, Any] = {}
    if args.lane in {"all", "voyage"}:
        b_result = args.root / "results" / "voyage-4-large+BGE.json"
        runs["B"] = {"status": "complete", "name": "voyage-4-large+BGE", "resumed": True} if _has_complete_result(b_result) else matrix.run_reranker("voyage-4-large+BGE", matrix.candidates("voyage-4-large", "voyage", "voyage-4-large"), "local", "BAAI/bge-reranker-v2-m3")
        runs["E"] = matrix.run_reranker("control+rerank-2.5", control_pool, "voyage", "rerank-2.5")
        if args.lane == "voyage":
            del control_pool
            gc.collect()
        b_pool = matrix.candidates("voyage-4-large", "voyage", "voyage-4-large")
        runs["G"] = matrix.run_reranker("voyage-4-large+rerank-2.5", b_pool, "voyage", "rerank-2.5")
        del b_pool
        gc.collect()
        context_rerank_result = args.root / "results" / "voyage-context-4+rerank-2.5.json"
        context_pool: dict[str, list[dict[str, Any]]] | None = None
        if _has_complete_result(context_rerank_result):
            runs["K"] = {"status": "complete", "name": "voyage-context-4+rerank-2.5", "resumed": True}
        else:
            context_pool = matrix.candidates("voyage-context-4", "voyage", "voyage-context-4", contextual=True)
            runs["K"] = matrix.run_reranker("voyage-context-4+rerank-2.5", context_pool, "voyage", "rerank-2.5")
        if args.skip_context_bge:
            runs["I"] = {"status": "excluded", "reason": "user_requested_skip_context_bge"}
        else:
            if context_pool is None:
                context_pool = matrix.candidates("voyage-context-4", "voyage", "voyage-context-4", contextual=True)
            runs["I"] = matrix.run_reranker("voyage-context-4+BGE", context_pool, "local", "BAAI/bge-reranker-v2-m3")
    if args.lane in {"all", "jina"}:
        c_pool = matrix.candidates("jina-embeddings-v5-text-small", "jina", "jina-embeddings-v5-text-small")
        runs |= {"C": matrix.run_reranker("jina-embeddings-v5-text-small+BGE", c_pool, "local", "BAAI/bge-reranker-v2-m3"), "D": matrix.run_reranker("control+jina-reranker-v3.5", control_pool, "jina", "jina-reranker-v3.5"), "H": matrix.run_reranker("jina-embeddings-v5-text-small+jina-reranker-v3.5", c_pool, "jina", "jina-reranker-v3.5")}
        b_path = args.root / "candidate_pools" / "voyage-4-large.json"
        i_path = args.root / "candidate_pools" / "voyage-context-4.json"
        if b_path.exists():
            b_pool = matrix.candidates("voyage-4-large", "voyage", "voyage-4-large")
            runs["F"] = matrix.run_reranker("voyage-4-large+jina-reranker-v3.5", b_pool, "jina", "jina-reranker-v3.5")
        if i_path.exists():
            i_pool = matrix.candidates("voyage-context-4", "voyage", "voyage-context-4", contextual=True)
            runs["J"] = matrix.run_reranker("voyage-context-4+jina-reranker-v3.5", i_pool, "jina", "jina-reranker-v3.5")
    write_json_atomic(args.root / f"lane_{args.lane}_status.json", {"status": "complete", "lane": args.lane, "runs": runs, "provider_events": matrix.client.events, "cache": dict(matrix.cache.counts)})


if __name__ == "__main__":
    main()
