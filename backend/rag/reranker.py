from functools import lru_cache
import os

from rag.hardware import batch_size, torch_device


def speed_mode() -> str:
    """"auto" (default): full precision. "fast": 8-bit reranker on a CPU server (set in Admin)."""
    return (os.environ.get("BCT_SPEED_MODE") or "auto").strip().casefold()


@lru_cache(maxsize=1)
def create_reranker():
    """The reranker, loaded once (admin.set_secrets clears this when the speed mode changes)."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=torch_device())
    if speed_mode() == "fast" and torch_device() == "cpu":
        import torch

        # 8-bit weights: 1.9x faster per passage on CPU, but 5 points lower on the retrieval
        # benchmark (82 -> 77 % top-5), so only when the administrator picks "fast".
        # ponytail: torch's dynamic quantization (deprecated in favour of torchao, still works).
        torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return model


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
