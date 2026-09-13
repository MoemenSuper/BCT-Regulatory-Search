import os


def retrieve_relevant_chunks(user_query, vector_store, k=20):
    results = vector_store.similarity_search_with_score(user_query, k=k)
    return [document for document, _ in results]


def load_vector_store(embedding_model, *, persist_directory=None, collection_name=None):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name or os.environ.get(
            "BCT_CHROMA_COLLECTION", "bct_regulations"
        ),
        embedding_function=embedding_model,
        persist_directory=persist_directory or os.environ.get("BCT_CHROMA_DB", "./chroma_db"),
    )
