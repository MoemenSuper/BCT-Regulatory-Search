"""Removal staging survives Chroma pages whose embeddings cannot be read."""

import embedding as embedding_module
from ingestion.index import _read_collection


class _Collection:
    name = "old"

    def __init__(self, rows, broken_from):
        self.rows = rows
        self.broken_from = broken_from

    def count(self):
        return len(self.rows)

    def get(self, *, limit, offset, include):
        page = self.rows[offset : offset + limit]
        if "embeddings" in include and offset + limit > self.broken_from:
            raise RuntimeError("Error executing plan: Internal error: Error finding id")
        result = {
            "ids": [row[0] for row in page],
            "documents": [row[1] for row in page],
            "metadatas": [row[2] for row in page],
        }
        if "embeddings" in include:
            result["embeddings"] = [row[3] for row in page]
        return result


class _Embedder:
    def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]


def test_unreadable_embeddings_are_re_embedded_and_removed_pdf_is_skipped(monkeypatch):
    monkeypatch.setattr(embedding_module, "create_embedding_model", lambda: _Embedder())
    rows = [(f"id{i}", "x" * (i + 1), {"source": "keep.pdf" if i != 3 else "drop.pdf"}, [0.5]) for i in range(5)]
    got = _read_collection(_Collection(rows, broken_from=2), batch_size=2, exclude_sources=["drop.pdf"])
    assert [row[0] for row in got] == ["id0", "id1", "id2", "id4"]
    assert [row[3] for row in got] == [[0.5], [0.5], [3.0], [5.0]]
