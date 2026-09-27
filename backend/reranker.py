from functools import lru_cache


def _torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=1)
def create_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=_torch_device())


def score_documents(reranker, user_query, candidate_docs):
    pairs = [(user_query, document.page_content) for document in candidate_docs]
    return reranker.predict(pairs)
