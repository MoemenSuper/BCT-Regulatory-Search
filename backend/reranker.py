from functools import lru_cache


@lru_cache(maxsize=1)
def create_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("BAAI/bge-reranker-v2-m3")


def score_documents(reranker, user_query, candidate_docs):
    pairs = [(user_query, document.page_content) for document in candidate_docs]
    return reranker.predict(pairs)
