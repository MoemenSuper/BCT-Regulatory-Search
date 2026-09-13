from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def load_documents_from_chroma(vector_store):
    results = vector_store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(results["documents"], results["metadatas"])
    ]


def create_bm25(documents):
    return BM25Okapi([document.page_content.lower().split() for document in documents])


def retrieve_bm25(query, bm25, documents, k=15):
    return bm25.get_top_n(query.lower().split(), documents, n=k)
