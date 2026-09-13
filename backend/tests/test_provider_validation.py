import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import runtime_retrieval

from runtime_retrieval import VoyageRuntimeClient


class Response:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


def client(tmp_path, monkeypatch, body):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    return VoyageRuntimeClient(tmp_path, request_post=lambda *_args, **_kwargs: Response(body))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0])
def test_query_embedding_rejects_nonfinite_or_zero_vectors(tmp_path, monkeypatch, value):
    provider = client(tmp_path, monkeypatch, {"data": [{"data": [{"embedding": [value] * 1024}]}]})
    with pytest.raises(ValueError):
        provider.embed_query("question")
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("data", [
    [{"index": 0, "relevance_score": float("nan")}],
    [{"index": False, "relevance_score": 0.8}],
    [{"index": 0, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}],
])
def test_reranker_rejects_nonfinite_boolean_or_duplicate_results(tmp_path, monkeypatch, data):
    provider = client(tmp_path, monkeypatch, {"data": data})
    with pytest.raises(ValueError):
        provider.rerank("question", ["evidence"])
    assert not list(tmp_path.glob("*.json"))


def test_concurrent_identical_requests_write_one_valid_cache(tmp_path, monkeypatch):
    ready = Barrier(2)
    replace = runtime_retrieval.os.replace
    def synchronized_replace(source, destination):
        ready.wait(timeout=5)
        replace(source, destination)
    monkeypatch.setattr(runtime_retrieval.os, "replace", synchronized_replace)
    provider = client(tmp_path, monkeypatch, {"data": [{"index": 0, "relevance_score": 0.8}]})
    with ThreadPoolExecutor(max_workers=2) as workers:
        calls = [workers.submit(provider.rerank, "question", ["evidence"]) for _ in range(2)]
        assert [call.result(timeout=10) for call in calls] == [[0.8], [0.8]]
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert provider.rerank("question", ["evidence"]) == [0.8]
