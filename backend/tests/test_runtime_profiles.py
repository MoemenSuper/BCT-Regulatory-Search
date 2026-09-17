from types import SimpleNamespace

import pytest

from runtime_profiles import (
    RuntimeProfile,
    RuntimeProfileManager,
    profile_options,
)


def test_profile_options_describe_the_three_coherent_deployment_choices():
    options = {item["value"]: item for item in profile_options()}

    assert set(options) == {"local", "local_hybrid", "cloud"}
    assert options["local"]["retrieval"] == "local_e5_bge"
    assert options["local"]["answer"] == "ollama"
    assert options["local"]["qualification"] == "experimental_rejected"
    assert options["local_hybrid"]["retrieval"] == "local_e5_bge"
    assert options["local_hybrid"]["answer"] == "groq"
    assert options["cloud"]["retrieval"] == "cloud_embed_rerank"
    assert options["cloud"]["answer"] == "groq"


def test_manager_shares_local_retrieval_and_lazily_builds_cloud_retrieval():
    local_backend = object()
    cloud_backend = object()
    calls = []

    def cloud_factory():
        calls.append("cloud")
        return cloud_backend

    manager = RuntimeProfileManager(local_backend, cloud_factory)

    local = manager.get(RuntimeProfile.LOCAL)
    hybrid = manager.get("local_hybrid")
    first_cloud = manager.get("cloud")
    second_cloud = manager.get("cloud")

    assert local.retrieval_backend is local_backend
    assert local.answer_provider == "ollama"
    assert hybrid.retrieval_backend is local_backend
    assert hybrid.answer_provider == "groq"
    assert first_cloud.retrieval_backend is cloud_backend
    assert second_cloud.retrieval_backend is cloud_backend
    assert calls == ["cloud"]


def test_manager_rejects_an_unknown_profile():
    manager = RuntimeProfileManager(object(), lambda: object())

    with pytest.raises(ValueError, match="Unknown runtime profile"):
        manager.get("half-cloud")
