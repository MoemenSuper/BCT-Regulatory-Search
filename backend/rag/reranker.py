"""The reranker: a cross-encoder that reads the question and one chunk together and gives the pair
a relevance score. Search first gathers about 30-40 candidates cheaply (meaning search + keyword
search); the reranker then reorders them precisely. It is too slow to read the whole corpus,
which is why it only sees the shortlist.

Model: bge-reranker-v2-m3 (multilingual: French, Arabic, English).
"""
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

        # Store the weights as 8-bit integers instead of 32-bit floats: about twice as fast on a
        # CPU, slightly less precise (the right page reached the top 5 a bit less often), so it is
        # only used when the administrator chooses "fast".
        # ponytail: torch's dynamic quantization (deprecated in favour of torchao, still works).
        torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return model


def score_documents(reranker, user_query, candidate_docs):
    """One relevance score per candidate, in the same order (higher = more relevant)."""
    # Each pair is (question, chunk): the cross-encoder reads both in one input.
    pairs = [(user_query, document.page_content) for document in candidate_docs]
    scores = reranker.predict(pairs, batch_size=batch_size(16))
    if torch_device() == "cuda":
        import torch

        # PyTorch keeps the GPU memory of the largest batch it has seen. Over a long session that
        # filled an 8 GB card and Windows started swapping, so answers slowed down. Give the
        # memory back after every question; the scores do not change.
        torch.cuda.empty_cache()
    return scores
