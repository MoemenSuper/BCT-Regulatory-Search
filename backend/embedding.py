from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


def _torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=1)
def create_embedding_model():
    device = _torch_device()
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": device},
        encode_kwargs={"prompt": "passage: "},
        query_encode_kwargs={"prompt": "query: "},
    )
