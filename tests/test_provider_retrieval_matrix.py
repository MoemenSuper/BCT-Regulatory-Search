from __future__ import annotations

import pytest
import requests

from langchain_core.documents import Document
from experiments.provider_retrieval_matrix import ProviderClient, VOYAGE_SAFE_REQUEST_BYTES, _acquire_gpu_lock, _deserialize_pool, _embedding_batches, _exact_mcnemar, _holm, _release_gpu_lock, _rerank_batches


class _Cache:
    def get(self, *_args):
        return None

    def put(self, *_args):
        return None


def test_reranker_rejects_partial_or_duplicate_provider_indexes(monkeypatch):
    client = ProviderClient(_Cache())
    monkeypatch.setattr("experiments.provider_retrieval_matrix._load_env", lambda _name: "test")
    monkeypatch.setattr(client, "_post", lambda **_kwargs: {"response": {"data": [{"index": 0, "relevance_score": 1.0}]}})
    with pytest.raises(ValueError, match="Partial"):
        client.rerank("jina", "jina-reranker-v3.5", "q", ["a", "b"])


def test_reranker_preserves_submitted_index_mapping(monkeypatch):
    client = ProviderClient(_Cache())
    monkeypatch.setattr("experiments.provider_retrieval_matrix._load_env", lambda _name: "test")
    monkeypatch.setattr(client, "_post", lambda **_kwargs: {"response": {"data": [{"index": 1, "relevance_score": 0.2}, {"index": 0, "relevance_score": 0.9}]}})
    assert client.rerank("jina", "jina-reranker-v3.5", "q", ["first", "second"]) == [0.9, 0.2]


def test_voyage_rerank_batches_preserve_candidate_order_under_the_safe_budget():
    documents = ["a" * 7000, "b" * 7000, "c" * 7000]
    batches = _rerank_batches(documents, "voyage")
    assert batches == [[documents[0]], [documents[1]], [documents[2]]]
    assert [text for batch in batches for text in batch] == documents
    assert all(sum(len(text.encode("utf-8")) for text in batch) <= VOYAGE_SAFE_REQUEST_BYTES // 2 for batch in batches)


def test_voyage_standard_embedding_uses_singular_input_field(monkeypatch):
    client = ProviderClient(_Cache())
    monkeypatch.setattr("experiments.provider_retrieval_matrix._load_env", lambda _name: "test")
    captured = {}
    def post(**kwargs):
        captured.update(kwargs)
        return {"response": {"data": [{"index": 0, "embedding": [0.0] * 1024}]}}
    monkeypatch.setattr(client, "_post", post)
    assert len(client.embeddings("voyage", "voyage-4-large", ["q"], "query")[0]) == 1024
    assert captured["payload"]["input"] == ["q"]
    assert "inputs" not in captured["payload"]


def test_context_query_preserves_flat_query_input_and_nested_response(monkeypatch):
    client = ProviderClient(_Cache())
    monkeypatch.setattr("experiments.provider_retrieval_matrix._load_env", lambda _name: "test")
    captured = {}
    def post(**kwargs):
        captured.update(kwargs)
        return {"response": {"data": [{"index": 0, "data": [{"index": 0, "embedding": [0.0] * 1024}]}]}}
    monkeypatch.setattr(client, "_post", post)
    assert len(client.embeddings("voyage", "voyage-context-4", ["q"], "query", contextual=True)[0]) == 1024
    assert captured["payload"]["inputs"] == ["q"]


def test_candidate_pool_cache_accepts_its_own_list_format():
    raw = [{"document": {"page_content": "text", "metadata": {"source": "x.pdf", "page": 1}}, "representations": ["native"], "ranks": {"native": {"dense": 1}}}]
    assert _deserialize_pool(raw)[0]["document"].page_content == "text"


def test_checkpoint_write_retries_a_transient_windows_lock(monkeypatch, tmp_path):
    import experiments.provider_retrieval_matrix as matrix
    calls = {"count": 0}
    def flaky(_path, _value):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("locked")
    monkeypatch.setattr(matrix, "_atomic_write", flaky)
    monkeypatch.setattr(matrix.time, "sleep", lambda _seconds: None)
    matrix.write_json_atomic(tmp_path / "checkpoint.json", {"ok": True})
    assert calls["count"] == 3


def test_gpu_lock_recovers_an_empty_lock_left_by_a_crashed_legacy_worker(tmp_path):
    lock = tmp_path / "bge_gpu.lock"
    lock.mkdir()
    _acquire_gpu_lock(lock)
    assert (lock / "owner.json").exists()
    _release_gpu_lock(lock)
    assert not lock.exists()


def test_provider_429_waits_and_retries_without_abandoning_the_request(monkeypatch):
    client = ProviderClient(_Cache())
    responses = iter([
        type("Response", (), {"status_code": 429, "content": b"x", "headers": {}, "json": lambda self: {"detail": "rate limited"}})(),
        type("Response", (), {"status_code": 200, "content": b"x", "headers": {}, "json": lambda self: {"data": []}})(),
    ])
    monkeypatch.setattr("experiments.provider_retrieval_matrix.requests.post", lambda *_args, **_kwargs: next(responses))
    waits = []
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.sleep", waits.append)
    result = client._post(provider="test", url="https://example.invalid", secret="secret", payload={}, cache_kind="test", binding={})
    assert result["status"] == 200
    assert waits == [60.0]


def test_transient_network_error_retries_in_the_worker(monkeypatch):
    client = ProviderClient(_Cache())
    responses = iter([
        requests.exceptions.SSLError("connection closed"),
        type("Response", (), {"status_code": 200, "content": b"x", "headers": {}, "json": lambda self: {"data": []}})(),
    ])
    def post(*_args, **_kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr("experiments.provider_retrieval_matrix.requests.post", post)
    monkeypatch.setattr("experiments.provider_retrieval_matrix.random.uniform", lambda *_args: 0.0)
    monotonic = iter([0.0, 2.0, 20.0])
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.monotonic", lambda: next(monotonic))
    waits = []
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.sleep", waits.append)
    assert client._post(provider="voyage", url="https://example.invalid", secret="secret", payload={}, cache_kind="test", binding={})["status"] == 200
    assert waits == [2.0, 18.0]


def test_jina_insufficient_balance_waits_and_retries_in_the_worker(monkeypatch):
    client = ProviderClient(_Cache())
    responses = iter([
        type("Response", (), {"status_code": 403, "content": b"x", "headers": {}, "json": lambda self: {"code": "AUTHZ_INSUFFICIENT_BALANCE"}})(),
        type("Response", (), {"status_code": 200, "content": b"x", "headers": {}, "json": lambda self: {"data": []}})(),
    ])
    monkeypatch.setattr("experiments.provider_retrieval_matrix.requests.post", lambda *_args, **_kwargs: next(responses))
    waits = []
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.sleep", waits.append)
    assert client._post(provider="jina", url="https://example.invalid", secret="secret", payload={}, cache_kind="test", binding={})["status"] == 200
    assert waits == [60.0]


def test_voyage_requests_are_paced_to_the_reduced_three_rpm_limit(monkeypatch):
    client = ProviderClient(_Cache())
    monkeypatch.setattr("experiments.provider_retrieval_matrix.requests.post", lambda *_args, **_kwargs: type("Response", (), {"status_code": 200, "content": b"x", "headers": {}, "json": lambda self: {"data": []}})())
    monotonic = iter([0.0, 5.0, 5.0])
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.monotonic", lambda: next(monotonic))
    waits = []
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.sleep", waits.append)
    client._post(provider="voyage", url="https://example.invalid/a", secret="secret", payload={}, cache_kind="test", binding={})
    client._post(provider="voyage", url="https://example.invalid/b", secret="secret", payload={}, cache_kind="test", binding={})
    assert waits == [15.0]


def test_voyage_rotates_independent_configured_credential_slots(monkeypatch):
    client = ProviderClient(_Cache())
    monkeypatch.setattr("experiments.provider_retrieval_matrix._load_env", lambda name: {"VOYAGE_API_KEY": "first", "VOYAGE_API_KEY_SECONDARY": "second", "VOYAGE_API_KEY_TERTIARY": "third", "VOYAGE_API_KEY_QUATERNARY": "fourth"}[name])
    assert client._next_secret("voyage") == ("VOYAGE_API_KEY_TERTIARY", "third")
    assert client._next_secret("voyage") == ("VOYAGE_API_KEY_QUATERNARY", "fourth")
    assert client._next_secret("voyage") == ("VOYAGE_API_KEY_SECONDARY", "second")
    assert client._next_secret("voyage") == ("VOYAGE_API_KEY", "first")


def test_voyage_rate_limit_fails_over_to_the_other_ready_organization(monkeypatch):
    client = ProviderClient(_Cache())
    monkeypatch.setattr("experiments.provider_retrieval_matrix._load_env", lambda name: {"VOYAGE_API_KEY": "first", "VOYAGE_API_KEY_SECONDARY": "second", "VOYAGE_API_KEY_TERTIARY": "third", "VOYAGE_API_KEY_QUATERNARY": "fourth"}[name])
    responses = iter([
        type("Response", (), {"status_code": 429, "content": b"x", "headers": {}, "json": lambda self: {"detail": "rate limited"}})(),
        type("Response", (), {"status_code": 200, "content": b"x", "headers": {}, "json": lambda self: {"data": []}})(),
    ])
    headers = []
    def post(*_args, **kwargs):
        headers.append(kwargs["headers"]["Authorization"])
        return next(responses)
    monkeypatch.setattr("experiments.provider_retrieval_matrix.requests.post", post)
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.monotonic", lambda: 0.0)
    waits = []
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.sleep", waits.append)
    assert client._post(provider="voyage", url="https://example.invalid", payload={}, cache_kind="test", binding={})["status"] == 200
    assert headers == ["Bearer third", "Bearer fourth"]
    assert waits == []


def test_voyage_network_timeout_fails_over_to_the_other_ready_organization(monkeypatch):
    client = ProviderClient(_Cache())
    monkeypatch.setattr("experiments.provider_retrieval_matrix._load_env", lambda name: {"VOYAGE_API_KEY": "first", "VOYAGE_API_KEY_SECONDARY": "second", "VOYAGE_API_KEY_TERTIARY": "third", "VOYAGE_API_KEY_QUATERNARY": "fourth"}[name])
    responses = iter([requests.ConnectionError("timed out"), type("Response", (), {"status_code": 200, "content": b"x", "headers": {}, "json": lambda self: {"data": []}})()])
    headers = []
    def post(*_args, **kwargs):
        response = next(responses)
        if isinstance(response, requests.RequestException):
            raise response
        headers.append(kwargs["headers"]["Authorization"])
        return response
    monkeypatch.setattr("experiments.provider_retrieval_matrix.requests.post", post)
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.monotonic", lambda: 0.0)
    waits = []
    monkeypatch.setattr("experiments.provider_retrieval_matrix.time.sleep", waits.append)
    assert client._post(provider="voyage", url="https://example.invalid", payload={}, cache_kind="test", binding={})["status"] == 200
    assert headers == ["Bearer fourth"]
    assert waits == []


def test_provider_requests_use_a_short_connect_and_read_timeout(monkeypatch):
    client = ProviderClient(_Cache())
    captured = {}
    def post(*_args, **kwargs):
        captured.update(kwargs)
        return type("Response", (), {"status_code": 200, "content": b"x", "headers": {}, "json": lambda self: {"data": []}})()
    monkeypatch.setattr("experiments.provider_retrieval_matrix.requests.post", post)
    client._post(provider="test", url="https://example.invalid", secret="test", payload={}, cache_kind="test", binding={})
    assert captured["timeout"] == (10, 30)
    assert captured["headers"]["Connection"] == "close"


def test_voyage_embedding_batches_stay_under_the_safe_account_budget():
    documents = [Document(page_content="x" * 4000) for _ in range(10)]
    batches = _embedding_batches(documents, "voyage")
    assert len(batches) == 2
    assert all(sum(len(item.page_content.encode("utf-8")) for item in batch) <= VOYAGE_SAFE_REQUEST_BYTES for batch in batches)


def test_mcnemar_and_holm_are_paired_and_monotone():
    control = {"r1": {"candidate_page_rank": 6}, "r2": {"candidate_page_rank": 1}}
    records = [
        {"id": "r1", "relevant": True, "result": {"page_rank": 1}},
        {"id": "r2", "relevant": True, "result": {"page_rank": 8}},
    ]
    paired = _exact_mcnemar(records, control)
    assert paired["repairs"] == paired["regressions"] == 1
    adjusted = _holm({"a": paired, "b": {**paired, "p_value": 0.01}})
    assert adjusted["a"] >= adjusted["b"]
