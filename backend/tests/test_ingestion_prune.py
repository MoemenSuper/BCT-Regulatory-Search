import json

import chromadb

from ingestion.index import prune_versions


def _version(root, name, collection=None):
    path = root / "versions" / name
    path.mkdir(parents=True)
    snapshot = {"version": name}
    if collection:
        snapshot.update(
            local_collection=f"bct_regulations_{collection}",
            local_visual_collection=f"bct_arabic_visual_{collection}",
            local_chroma_db=str(root / "local_chroma"),
        )
    (path / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    return path


def test_prune_keeps_active_recent_and_served_collections(tmp_path, monkeypatch):
    monkeypatch.delenv("BCT_CHROMA_COLLECTION", raising=False)
    monkeypatch.delenv("BCT_OCR_CHROMA_COLLECTION", raising=False)
    client = chromadb.PersistentClient(path=str(tmp_path / "local_chroma"))
    for name in ("bct_regulations", "bct_regulations_a", "bct_arabic_visual_a",
                 "bct_regulations_b", "bct_arabic_visual_b", "bct_regulations_c", "bct_arabic_visual_c"):
        client.create_collection(name)
    (tmp_path / "versions" / "baked-active").mkdir(parents=True)  # hand-seeded: never pruned
    _version(tmp_path, "20260101T000000Z-aaa", "a")
    _version(tmp_path, "20260102T000000Z-bbb", "b")
    _version(tmp_path, "20260103T000000Z-ccc", "c")  # served index of the active version below
    _version(tmp_path, "20260104T000000Z-ddd")  # activated without its own local index
    _version(tmp_path, "20260105T000000Z-eee")
    active = _version(tmp_path, "20260106T000000Z-fff")
    (tmp_path / "ACTIVE.json").write_text(json.dumps({"active_version": f"versions/{active.name}"}), encoding="utf-8")

    report = prune_versions(tmp_path, keep=2)

    remaining = sorted(p.name for p in (tmp_path / "versions").iterdir())
    assert remaining == ["20260105T000000Z-eee", "20260106T000000Z-fff", "baked-active"]
    assert report == {"removed_versions": 4, "removed_collections": 4}
    names = {c.name for c in chromadb.PersistentClient(path=str(tmp_path / "local_chroma")).list_collections()}
    # c is still served (the newest version with a local index); unversioned names are never touched.
    assert names == {"bct_regulations", "bct_regulations_c", "bct_arabic_visual_c"}


def test_prune_without_versions_is_a_no_op(tmp_path):
    assert prune_versions(tmp_path) == {"removed_versions": 0, "removed_collections": 0}
