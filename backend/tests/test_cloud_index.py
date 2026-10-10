"""The cloud profile's Voyage index is built on demand from the admin screen, beside the live
version, and only appears once complete; the profile cannot be chosen before it exists."""
import numpy as np
import pytest
from langchain_core.documents import Document

from cloud import voyage_index
from cloud.voyage_client import CloudEmbedSpec
from ingestion.index import _jsonl


class FakeClient:
    dimension = 4

    def embed_document_chunks(self, texts, **_kwargs):
        return np.ones((len(texts), self.dimension), dtype=np.float32)


class BrokenClient(FakeClient):
    def embed_document_chunks(self, texts, **_kwargs):
        raise RuntimeError("Voyage is down")


@pytest.fixture
def active(tmp_path, monkeypatch):
    spec = CloudEmbedSpec(key="voyage", provider="voyage", model="fake", dimension=4, contextual=False)
    monkeypatch.setattr(voyage_index, "VOYAGE_SPEC", spec)
    version = tmp_path / "versions" / "v1"
    _jsonl(version / "native.jsonl", [Document(page_content="Article 3 : le ratio", metadata={"source": "a.pdf", "page": 1})])
    _jsonl(version / "arabic_ocr_secondary.jsonl", [Document(page_content="الفصل 3", metadata={"source": "b.pdf", "page": 2})])
    return version


def test_the_cloud_index_is_ready_only_after_a_build(active, monkeypatch):
    monkeypatch.setattr(voyage_index, "create_cloud_runtime_client", lambda *a, **k: FakeClient())
    assert not voyage_index.cloud_index_ready(active)

    voyage_index.build_cloud_index(active.parent.parent, active)

    assert voyage_index.cloud_index_ready(active)
    assert not list(active.glob(".cloud-index-*"))  # the temporary folder is gone


def test_a_failed_build_leaves_the_live_version_untouched(active, monkeypatch):
    monkeypatch.setattr(voyage_index, "create_cloud_runtime_client", lambda *a, **k: BrokenClient())

    with pytest.raises(RuntimeError):
        voyage_index.build_cloud_index(active.parent.parent, active)

    assert not (active / "indexes").exists()
    assert not voyage_index.cloud_index_ready(active)



def _stage_one_upload(active):
    staging = active.parent / "v2"
    new = [Document(page_content="Article 4 : nouveau", metadata={"source": "c.pdf", "page": 1})]
    _jsonl(staging / "native.jsonl", voyage_index._read_chunks(active / "native.jsonl") + new)
    _jsonl(staging / "arabic_ocr_secondary.jsonl", voyage_index._read_chunks(active / "arabic_ocr_secondary.jsonl"))
    voyage_index.stage_voyage_indexes(root=active.parent.parent, active=active, staging=staging,
                                      source_keys=set(), new_primary=new, new_visual=[])
    return staging


def test_once_built_every_upload_keeps_the_cloud_index_current(active, monkeypatch):
    monkeypatch.delenv("BCT_INGEST_CLOUD_INDEX", raising=False)
    monkeypatch.delenv("BCT_DEFAULT_PROFILE", raising=False)
    monkeypatch.setattr(voyage_index, "create_cloud_runtime_client", lambda *a, **k: FakeClient())
    voyage_index.build_cloud_index(active.parent.parent, active)

    assert voyage_index.cloud_index_ready(_stage_one_upload(active))


def test_a_voyage_outage_does_not_block_an_upload_for_the_local_profiles(active, monkeypatch):
    monkeypatch.delenv("BCT_INGEST_CLOUD_INDEX", raising=False)
    monkeypatch.delenv("BCT_DEFAULT_PROFILE", raising=False)
    monkeypatch.setattr(voyage_index, "create_cloud_runtime_client", lambda *a, **k: FakeClient())
    voyage_index.build_cloud_index(active.parent.parent, active)
    monkeypatch.setattr(voyage_index, "create_cloud_runtime_client", lambda *a, **k: BrokenClient())

    staged = _stage_one_upload(active)  # no exception: the upload goes live

    assert not voyage_index.cloud_index_ready(staged)  # the cloud index waits for a rebuild
