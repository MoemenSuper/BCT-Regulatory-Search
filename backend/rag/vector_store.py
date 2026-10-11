"""Chroma, the vector database: it stores one vector per chunk and returns the chunks whose
vectors are closest to the question's vector (meaning search).
"""
import os


def retrieve_relevant_chunks(user_query, vector_store, k=20):
    """The k chunks closest in meaning to the question (Chroma embeds the question itself)."""
    results = vector_store.similarity_search_with_score(user_query, k=k)
    return [document for document, _ in results]  # the distance score is not used further


def load_vector_store(embedding_model, *, persist_directory=None, collection_name=None):
    """Open an existing Chroma collection on disk. The folder and collection name come from the
    live index (set by ingestion/index.py configure_runtime_assets) unless given explicitly."""
    from langchain_chroma import Chroma  # imported here: loading Chroma is slow and not always needed

    return Chroma(
        collection_name=collection_name or os.environ.get(
            "BCT_CHROMA_COLLECTION", "bct_regulations"
        ),
        embedding_function=embedding_model,
        persist_directory=persist_directory or os.environ.get("BCT_CHROMA_DB", "./chroma_db"),
    )
