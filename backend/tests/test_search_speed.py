"""CPU speed: each passage reranked once, and the 8-bit reranker only when the admin picks "fast"."""
from langchain_core.documents import Document
import pytest

from rag import reranker
from rag.runtime_retrieval import script_scores


def test_each_passage_is_scored_once_against_a_wording_in_its_own_script():
    calls = []

    def score(wording, chunks):
        calls.append((wording, [c.page_content for c in chunks]))
        return [1.0] * len(chunks)

    documents = [Document(page_content="Le ratio de solvabilité est de 10 %."),
                 Document(page_content="نسبة الملاءة لا تقل عن 10 بالمائة.")]
    wordings = ["ratio de solvabilité minimum", "minimum solvency ratio", "النسبة الدنيا للملاءة"]

    assert script_scores(score, wordings[0], wordings[1:], documents) == [1.0, 1.0]
    assert calls == [
        ("ratio de solvabilité minimum", ["Le ratio de solvabilité est de 10 %."]),
        ("النسبة الدنيا للملاءة", ["نسبة الملاءة لا تقل عن 10 بالمائة."]),
    ]


@pytest.mark.parametrize("mode,device,quantized", [("fast", "cpu", True), ("auto", "cpu", False), ("fast", "cuda", False)])
def test_fast_mode_quantizes_the_reranker_on_cpu_only(monkeypatch, mode, device, quantized):
    seen = []
    monkeypatch.setenv("BCT_SPEED_MODE", mode)
    monkeypatch.setattr(reranker, "torch_device", lambda: device)
    monkeypatch.setattr("sentence_transformers.CrossEncoder", lambda name, device: "model")
    monkeypatch.setattr("torch.ao.quantization.quantize_dynamic", lambda model, *a, **k: seen.append(model))
    reranker.create_reranker.cache_clear()
    try:
        assert reranker.create_reranker() == "model"
    finally:
        reranker.create_reranker.cache_clear()
    assert seen == (["model"] if quantized else [])


def test_the_speed_mode_only_takes_known_values(tmp_path):
    from extras.app_settings import AppSettingsStore

    store = AppSettingsStore(tmp_path / "settings.sqlite3")
    try:
        with pytest.raises(ValueError, match="BCT_SPEED_MODE"):
            store.update_secrets({"BCT_SPEED_MODE": "turbo"})
        config = store.update_secrets({"BCT_SPEED_MODE": "Fast"})
        mode = next(s for s in config["secrets"] if s["key"] == "BCT_SPEED_MODE")
        assert mode["value"] == "fast" and mode["choices"] == ["auto", "fast"]
    finally:
        store.update_secrets({"BCT_SPEED_MODE": None})
        store.close()
