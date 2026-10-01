from functools import lru_cache

from hardware import batch_size, torch_device


@lru_cache(maxsize=1)
def create_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=torch_device())


def score_documents(reranker, user_query, candidate_docs):
    pairs = [(user_query, document.page_content) for document in candidate_docs]
    scores = reranker.predict(pairs, batch_size=batch_size(16))
    if torch_device() == "cuda":
        import torch

        # PyTorch keeps the GPU memory of the largest batch it has seen. Over a long session that
        # filled an 8 GB card (3.9 -> 7.4 GB in 60 questions) and Windows started swapping, so
        # some answers took two minutes. Give it back after every question; the scores are the same.
        torch.cuda.empty_cache()
    return scores
